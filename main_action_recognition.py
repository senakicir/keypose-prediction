#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import os
import numpy as np
import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import utils as utils
from utils.action_recognition.model import HCN 
from utils.log import Logger_TB, Logger_CSV, save_ckpt
from utils.action_recognition.run_ar import run_action_recognition as run
from utils.action_recognition.loss_funcs_ar import Loss_Function_Manager_Action_Recognition
from utils.opt import Options, save_opt
from utils.visualization_client import display_confusion_matrix
import utils.data_utils as data_utils
from utils.rng import Random_Number_Generator
from utils.h36motion3d import H36motion3D_SeqOnly
from utils.cmu_motion_3d import CMU_Motion3D_SeqOnly

torch.set_num_threads(8)


def return_dataset_seq_only(dataset):
    if dataset == "h36m":
        return H36motion3D_SeqOnly
    elif dataset == "cmu":
        return CMU_Motion3D_SeqOnly

def main(opt):
    start_epoch = 0
    acc_best = -10000
    is_best = False
    update_losses_epoch = 1
    lr_now = opt.ar_lr
    is_cuda = torch.cuda.is_available()

    date_time= datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    ckpt_datetime = opt.ckpt+'/action_recognition/'+date_time
    while os.path.exists(ckpt_datetime):
        ckpt_datetime += "_x"
    os.makedirs(ckpt_datetime)
    opt.ckpt_datetime = ckpt_datetime
    logger_tb = Logger_TB(opt.ckpt_datetime)
    logger_tr = Logger_CSV(opt.ckpt_datetime, file_name="log_tr")
    logger_val = Logger_CSV(opt.ckpt_datetime, file_name="log_val")
    logger_test = Logger_CSV(opt.ckpt_datetime, file_name="log_test")
    save_opt(opt, logger_tb.writer_tr)

    # create model
    print(">>> creating model")

    # save option in log
    ar_type = "OMAC" if opt.ar_motion_only else "FAC"
    run_info = ar_type + "_" + opt.dataset + '_seq_len{}'.format(opt.ar_seq_len)

    acts = data_utils.define_actions(opt.actions, opt.dataset, opt.overfitting_exp)

    model = HCN( in_channel=3,
                 num_joint= data_utils.joint_num(opt.dataset)-len(data_utils.nonmoving_joints(opt.dataset)),
                 num_person=1,
                 out_channel=64,
                 window_size=opt.ar_window_size,
                 seq_len = opt.ar_seq_len,
                 num_class = len(acts),
                 motion_only=opt.ar_motion_only)

    if is_cuda:
        model.cuda()

    rng = Random_Number_Generator(seed=opt.seed, input_noise_magn=opt.input_noise_magnitude)
    loss_function_manager = Loss_Function_Manager_Action_Recognition(actions=acts, ar_seq_len=opt.ar_seq_len)

    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.ar_lr, betas=(0.9, 0.999), eps=1e-8,
                weight_decay=opt.ar_weight_decay)

    ## load parameters of the model and the optimizer
    if opt.is_load or opt.is_eval:
        if opt.is_eval:
            model_path_len = opt.ar_model_path + "/model_" + run_info + '_best.pth.tar'
        else:
            model_path_len = opt.ar_model_path + "/model_" + run_info + '_last.pth.tar'
        model.load_weights(model_path_len)

        print(">>> loading ckpt len from '{}'".format(model_path_len))
        ckpt = torch.load(model_path_len)
        start_epoch = ckpt['epoch']
        acc_best = ckpt['acc']
        lr_now = ckpt['lr']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(">>> ckpt len loaded (epoch: {} | acc: {})".format(start_epoch, acc_best))

    ### DATA LOADING
    print(">>> Loading data")
    My_Dataset = return_dataset_seq_only(opt.dataset)
    data_dir = opt.cmu_data_dir if opt.dataset == "cmu" else opt.data_dir
    if not opt.is_eval:
        train_dataset = My_Dataset(path_to_data=data_dir, actions=acts, 
                                            seq_len=opt.ar_seq_len, split="train", sample_rate=opt.sample_rate, 
                                            overfitting_exp=opt.overfitting_exp)
        # load dadasets for training
        train_loader = DataLoader(dataset=train_dataset, batch_size=opt.train_batch,
                                    shuffle=True,num_workers=opt.job,pin_memory=True)

        print("Training data is loaded")
    
        #load validation set
        val_dataset = My_Dataset(path_to_data=data_dir, actions=acts, 
                                        seq_len=opt.ar_seq_len, split="val", sample_rate=opt.sample_rate, 
                                        overfitting_exp=opt.overfitting_exp)
        val_loader = DataLoader(dataset=val_dataset, batch_size=opt.test_batch, shuffle=False,
                                    num_workers=opt.job, pin_memory=True)
    
        print("Validation data is loaded")

    #load test set
    if opt.run_test:
        test_dataset = My_Dataset(path_to_data=data_dir, actions=acts, 
                            seq_len=opt.ar_seq_len, split="test", sample_rate=opt.sample_rate, 
                            overfitting_exp=opt.overfitting_exp)
        test_loader = DataLoader(dataset=test_dataset, batch_size=opt.test_batch,
                                    shuffle=False, num_workers=opt.job, pin_memory=True)
        print("Test data is loaded")


    print(">>> data loaded !")
    if not opt.is_eval:
        print(">>> train data {}".format(train_dataset.__len__()))
        print(">>> validation data {}".format(val_dataset.__len__()))
    if opt.run_test:
        print(">>> test data {}".format(test_dataset.__len__()))

    #####
    ## LOOP
    epoch_list = [start_epoch+1] if opt.is_eval else range(start_epoch, opt.ar_epochs)
    for epoch in epoch_list:
        logger_tb.update_epoch(epoch)

        # if we are only evaluating, we only need to run test (so skip training and validation)
        if not opt.is_eval:
            logger_tr.update_epoch(epoch)
            if (epoch + 1) % opt.lr_decay == 0:
                lr_now = utils.lr_decay(optimizer, lr_now, opt.lr_gamma)

            print('==========================')
            print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))

            # training
            fig_location = run(mode="train", data_loader=train_loader, 
                    model=model, loss_function_manager=loss_function_manager, optimizer=optimizer,
                    epoch=epoch,  save_loc=ckpt_datetime, writer=logger_tb.writer_tr, rng=rng,
                    is_cuda=is_cuda, dim_used=train_dataset.dim_used)


            #log training losses
            loss_dict_tr = loss_function_manager.return_average_losses()
            if (epoch+1)%update_losses_epoch == 0:
                logger_tb.write_average_losses_tb("train", loss_dict_tr, omit_keys="conf_mat")
            logger_tr.update_values_from_dict(dict(lr=lr_now, ce_loss=loss_dict_tr["ce_loss"]))
            logger_tr.save_csv_log()

            # validation 
            if (epoch+1)%1==0 and not opt.overfitting_exp:
                logger_val.update_epoch(epoch)
                fig_location= run(mode="val", data_loader=val_loader, 
                                model=model, loss_function_manager=loss_function_manager, optimizer=None,
                                epoch=epoch,  save_loc=ckpt_datetime, writer=logger_tb.writer_val, rng=rng,
                                is_cuda=is_cuda, dim_used=val_dataset.dim_used)

                #logging validation errors:
                loss_dict_val = loss_function_manager.return_average_losses()
                logger_tb.write_average_losses_tb("val", loss_dict_val, omit_keys=["conf_mat"])
                logger_val.update_values_from_dict(loss_dict_val, omit_keys=["conf_mat"])
                sum_v_acc = loss_dict_val["accuracy"] 

                display_confusion_matrix(loss_dict_val["conf_mat"], logger_tb.writer_val, epoch, loss_function_manager.actions, plot_loc=fig_location+"/",  ar_type=ar_type)

                if not np.isnan(sum_v_acc):
                    is_best = sum_v_acc > acc_best
                    acc_best = max(sum_v_acc, acc_best)
                else:
                    is_best = False

                ## save model we just trained
                file_name = ['model_' + run_info + '_best.pth.tar', "model_" + run_info + '_last.pth.tar']
                save_ckpt({'epoch': epoch + 1,
                         'lr': lr_now,
                         'acc': sum_v_acc,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()},
                        ckpt_path=ckpt_datetime,
                        is_best=is_best,
                        file_name=file_name)
                logger_val.save_csv_log()

        ##test
        if opt.run_test and (epoch+1)%1==0:
            logger_test.update_epoch(epoch)
            fig_location = run(mode="test", data_loader=test_loader, 
                                    model=model, loss_function_manager=loss_function_manager, optimizer=None,
                                    epoch=epoch,  save_loc=ckpt_datetime, writer=logger_tb.writer_test, rng=rng,
                                    is_cuda=is_cuda, dim_used=test_dataset.dim_used)
            #logging validation errors:
            loss_dict_test = loss_function_manager.return_average_losses()
            logger_tb.write_average_losses_tb("test", loss_dict_test, omit_keys=["conf_mat"])
            logger_test.update_values_from_dict(loss_dict_test, omit_keys=["conf_mat"])
            display_confusion_matrix(loss_dict_test["conf_mat"], logger_tb.writer_test, epoch, loss_function_manager.actions, plot_loc=fig_location+"/", ar_type=ar_type)
            logger_test.save_csv_log()

if __name__ == "__main__":
    option = Options().parse()
    main(option)
