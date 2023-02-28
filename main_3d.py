#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import os
import time
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from utils import utils as utils
from utils.opt import Options, save_opt, setup_folder
from utils.loss_funcs import Loss_Function_Manager
from utils.model.model_combined import Combined_Model
from utils.run_value import run as run
from utils.log import Logger_TB, Logger_CSV
from utils.rng import Random_Number_Generator
import utils.keypose_extract as keypose_module
from utils.model.optimizer import My_Optimizer
from utils.keypose_extract.keypose_utils import keypose_directory
from utils.action_recognition.loss_funcs_ar import Loss_Function_Manager_Action_Recognition
from utils.data_utils import define_actions, all_bone_connections
from utils.h36motion3d import H36motion3D
from utils.cmu_motion_3d import CMU_Motion3D


def return_dataset(dataset):
    if dataset == "h36m":
        return H36motion3D
    elif dataset == "cmu":
        return CMU_Motion3D

def main(opt):
    timer_begin = time.time()
    is_best = False
    no_improvement_count = 0
    is_cuda = torch.cuda.is_available()

    # set up folders and loggers
    setup_folder(opt)
    logger_tb = Logger_TB(opt.ckpt_datetime)
    logger_tr = Logger_CSV(opt.ckpt_datetime, file_name="log_tr")
    logger_val = Logger_CSV(opt.ckpt_datetime, file_name="log_val")
    logger_test = Logger_CSV(opt.ckpt_datetime, file_name="log_test")

    save_opt(opt, logger_tb.writer_tr)

    # create random number generator and loss function manager
    opt.keypose_dir += keypose_directory(opt.dataset, opt.kp_threshold, opt.kp_suffix)
    initial_cluster_centers, cluster_num = keypose_module.load_cluster_centers(opt.keypose_dir, opt.cluster_n)
    rng = Random_Number_Generator(seed=opt.seed, input_noise_magn=opt.input_noise_magnitude)
    loss_function_manager = Loss_Function_Manager(opt=opt, cluster_centers=initial_cluster_centers)
    run_info = 'keypose_{}_in{:d}_out{:d}_clusters{}'.format(opt.dataset,opt.input_kp,opt.output_kp, str(cluster_num))

    # create model and optimizer
    print(">>> creating model")
    model = Combined_Model(opt, cluster_num)
    model.cuda()
    print(">>> model total params: {:.2f}M".format(sum(p.numel() for p in model.trainable_model.parameters() if p.requires_grad) / 1000000.0))
    optimizer = My_Optimizer(opt, model.trainable_model, loss_function_manager)
    acc_best,start_epoch=model.load_weights(run_info, opt, optimizer)
    acc_best = 0
    sum_v_acc = acc_best
    prev_val = 0

    # create action recognition loss function managers (they keep track of accuracy according to action recognition model)
    all_actions = define_actions("all", opt.dataset, opt.overfitting_exp)
    ar_loss_function_manager = Loss_Function_Manager_Action_Recognition(actions=(all_actions), ar_seq_len=opt.ar_seq_len, dataset=opt.dataset)
    omac_ar_loss_function_manager = Loss_Function_Manager_Action_Recognition(actions=(all_actions), ar_seq_len=opt.ar_seq_len, only_motion=True)
    

    # DATA LOADING
    print(">>> Loading data")
    acts = define_actions(opt.actions, opt.dataset, opt.overfitting_exp)
    My_Dataset = return_dataset(opt.dataset)
    if not opt.is_eval:
        train_dataset = My_Dataset(opt=opt, cluster_n=cluster_num, cluster_centers=initial_cluster_centers, actions=acts, 
                                    split="train")
        train_loader = DataLoader(dataset=train_dataset,batch_size=opt.train_batch,
                                    shuffle=True,num_workers=opt.job, pin_memory=True,  drop_last=opt.drop_last)
        print("Training data is loaded")
        print(">>> train data {}".format(train_dataset.__len__()))

        val_data = dict()
        for act in acts:
            val_dataset = My_Dataset(opt=opt, cluster_n=cluster_num, cluster_centers=initial_cluster_centers, actions=[act], 
                                        split="val")
            val_data[act] = DataLoader(dataset=val_dataset,batch_size=opt.test_batch,
                                        shuffle=False,num_workers=opt.job,pin_memory=True)
            
        print("Validation data is loaded")
        print(">>> validation data {}".format(val_dataset.__len__()))

    #load test set
    if opt.run_test:
        test_data = dict()
        for act in acts:
            test_dataset = My_Dataset(opt=opt, cluster_n=cluster_num, cluster_centers=initial_cluster_centers, actions=[act], 
                                        split="test")
            test_data[act] = DataLoader(dataset=test_dataset,batch_size=opt.test_batch,
                                        shuffle=False,num_workers=opt.job,pin_memory=True)
        print("Test data is loaded")
        print(">>> test data {}".format(test_dataset.__len__()))
        dummy_pose = test_dataset.p3d[0][0]

    print(">>> data loaded !")

    ## MAIN LOOP
    epoch_list = [1] if opt.is_eval else range(start_epoch, opt.epochs+1)
    for epoch in epoch_list:
        update_losses = (epoch+1)%opt.update_losses_epoch==0
        update_figures = (epoch+1)%opt.update_figures_epoch==0
        run_validation = (epoch+1)%opt.val_epoch==0 
        interpolate_training = (epoch+1)%opt.train_mse_epoch==0

        logger_tb.update_epoch(epoch)
        is_best=False

        # if we are only evaluating, we only need to run test (skips training and validation)
        if not opt.is_eval :
            logger_tr.update_epoch(epoch)

            if opt.use_lr_decay:
                if no_improvement_count==10:
                    optimizer.lr_decay()
                    no_improvement_count = 0

            # update probabilities dependent on epoch number
            if opt.use_tf_mode == "ss":
                ss_prob = opt.ss_k/(opt.ss_k+np.exp(max(0, epoch-opt.ss_offset) /opt.ss_k)) 
            elif opt.use_tf_mode == "alwaystf":
                ss_prob = 0.0
            elif opt.use_tf_mode == "nevertf":
                ss_prob = 1.0

            shuffle_prob = opt.shuffle_prob if opt.random_shuffle else 0.0
            rng.update_prob(ss_prob, shuffle_prob)

            print('==========================')
            print('>>> epoch: {} | lr: {:.5f}, noise:{:.4f}, kd_t:{:.3f}, ss_prob:{:.3f}'.format(epoch + 1, 
                            optimizer.lr_model, rng.input_noise_magn, loss_function_manager.kd_t, ss_prob))

            # TRAINING
            run(opt=opt, mode="train", data_loader=train_loader, interpolate=interpolate_training,
                        model=model, loss_function_manager=loss_function_manager, optimizer=optimizer,
                        epoch=epoch, rng=rng, writer=logger_tb.writer_tr, action="all", 
                        update_losses=update_losses, update_figures=update_figures, is_cuda=is_cuda,
                        dim_used=train_dataset.dim_used)

            #log training losses
            loss_dict_tr = loss_function_manager.return_average_losses()
            if update_losses:
                logger_tb.write_average_losses_tb("train", loss_dict_tr, omit_keys=loss_function_manager.eval_frame_keys)
            logger_tr.update_values_from_dict(dict(lr_model=optimizer.lr_model, mse_upto_1sec=loss_dict_tr["mse_upto_1sec"]))
            logger_tr.save_csv_log()

            # VALIDATION             
            val_errors = {}
            if run_validation:
                logger_val.update_epoch(epoch)
                ar_loss_function_manager.reset_all() 
                omac_ar_loss_function_manager.reset_all() 
                
                for act in acts:
                    
                    with torch.no_grad():
                        run(opt=opt, mode="val", data_loader=val_data[act], interpolate=True,
                            model=model, loss_function_manager=loss_function_manager,
                            epoch=epoch, rng=rng, writer=logger_tb.writer_val, action=act, 
                            update_losses=update_losses, update_figures=update_figures,
                            is_cuda=is_cuda, dim_used=train_dataset.dim_used,
                            ar_loss_function_manager=ar_loss_function_manager,omac_ar_loss_function_manager=omac_ar_loss_function_manager)
                    val_errors[act] = loss_function_manager.return_average_losses()
                    logger_val.update_values_from_dict(val_errors[act], header_prefix=act+"_", omit_keys=loss_function_manager.training_loss_keys)

                #logging validation errors and AR accuracy
                loss_dict_val = loss_function_manager.return_average_losses()
                logger_val.update_values_from_dict(loss_dict_val, header_prefix=act+"_", omit_keys=loss_function_manager.training_loss_keys)
                logger_tb.write_average_losses_tb("val", loss_dict_val, omit_keys=loss_function_manager.eval_frame_keys)

                omac_ar_loss_dict_val = omac_ar_loss_function_manager.return_average_losses()
                logger_tb.write_average_losses_tb("val", omac_ar_loss_dict_val, omit_keys=["conf_mat", "ce_loss"])
                logger_val.update_values_from_dict(omac_ar_loss_dict_val, omit_keys=["conf_mat", "ce_loss"])
                logger_val.save_csv_log()

                #determine if best model according to validation acc
                sum_v_acc=omac_ar_loss_dict_val["_omacaccuracy"]+omac_ar_loss_dict_val["_omacaccuracy2"]  \
                            +omac_ar_loss_dict_val["_omacaccuracy3"] +omac_ar_loss_dict_val["_omacaccuracy5"]
                if not np.isnan(sum_v_acc):
                    is_best = sum_v_acc > acc_best
                    acc_best = max(sum_v_acc, acc_best)
  
                #for learning rate decay
                no_improvement_count = no_improvement_count+1 if not is_best else 0
                prev_val = sum_v_acc

                ## save model we just trained
                model.save_model(run_info, epoch, optimizer, sum_v_acc, opt.ckpt_datetime, is_best)
                
                
        # TEST
        if opt.is_eval or (opt.run_test and (is_best or (epoch+1)%opt.test_epoch==0 )):
            logger_test.update_epoch(epoch)
            test_errors = {}
            omac_ar_loss_function_manager.reset_all() 
            for act in acts:
                with torch.no_grad():
                    run(opt=opt, mode="test", data_loader=test_data[act], interpolate=True,
                        model=model, loss_function_manager=loss_function_manager, 
                        epoch=epoch, rng=rng, writer=logger_tb.writer_test, action=act, mean_pose=test_dataset.mean_poses,
                        update_losses=True, update_figures=True, is_cuda=is_cuda, dim_used=test_dataset.dim_used,
                        ar_loss_function_manager=ar_loss_function_manager,omac_ar_loss_function_manager=omac_ar_loss_function_manager)

                test_errors[act] = loss_function_manager.return_average_losses()
            
            # log ar accuracy
            omac_ar_loss_dict_test = omac_ar_loss_function_manager.return_average_losses()
            logger_tb.write_average_losses_tb("test", omac_ar_loss_dict_test, omit_keys=["conf_mat", "ce_loss"])

            loss_dict_test = utils.average_errs_across_actions(test_errors, acts)
            logger_test.update_values_from_dict(loss_dict_test, header_prefix="ave_", omit_keys=loss_function_manager.training_loss_keys)
            logger_tb.write_average_losses_tb("test", loss_dict_test, omit_keys=loss_function_manager.eval_frame_keys)
            
            logger_test.update_values_from_dict(omac_ar_loss_dict_test, omit_keys=["conf_mat", "ce_loss"])
            logger_test.save_csv_log()
 
            if is_best:
                other_vals_dict ={"val acc": acc_best.item(), "noise": rng.input_noise_magn, "kd_t": loss_function_manager.kd_t, "ss_prob": ss_prob}
                logger_tb.write_best_epoch_acc(epoch+1, omac_ar_loss_dict_test, other_vals_dict)

        if opt.is_eval:
            break
    timer_end = time.time()
    print("Total training time:", timer_end-timer_begin)
    logger_tb.write_training_time(timer_end-timer_begin)
    
if __name__ == "__main__":
    option = Options().parse()
    main(option)
