import os
import numpy as np
import datetime

import torch 
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from utils import utils as utils
from utils.opt import Options, save_opt
import utils.data_utils as data_utils
from utils.data_utils import define_actions
import utils.keypose_extract as keypose_module
from utils.keypose_extract.keypose_utils import keypose_directory

from utils.h36motion3d import H36motion3D
from utils.cmu_motion_3d import CMU_Motion3D

from utils.interpolator.model import KP_Interpolator
from utils.interpolator.run import run

from utils.log import Logger_TB, Logger_CSV, save_ckpt

torch.set_num_threads(8)
    

def return_dataset(dataset):
    if dataset == "h36m":
        return H36motion3D
    elif dataset == "cmu":
        return CMU_Motion3D

def train_interpolator(opt):

    opt.input_seq_n = 0
    opt.output_seq_n = 250
    opt.input_kp = 1
    opt.output_kp = opt.interpolator_kp_num-1

    err_best = 10000
    is_best = False

    update_figures_epoch = opt.update_figures_epoch

    date_time= datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    ckpt_datetime = opt.ckpt+'/interpolator/'+date_time
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

    opt.keypose_dir += keypose_directory(opt.dataset, opt.kp_threshold, opt.kp_suffix)

    initial_cluster_centers, cluster_contents, cluster_num = keypose_module.load_cluster_centers(opt.keypose_dir, opt.cluster_n, (opt.cluster_n_arms, opt.cluster_n_legs), (opt.cluster_n_outer, opt.cluster_n_inner), opt.clustering_method)

    run_info = 'interpolate_'+opt.dataset
    all_actions = define_actions("all", opt.dataset, opt.overfitting_exp)

    num_joint= data_utils.joint_num(opt.dataset)-len(data_utils.nonmoving_joints(opt.dataset))
     
    input_feature=opt.interpolator_kp_num if not opt.preprocess_interpolator else opt.interpolator_seq_len
    model = KP_Interpolator(input_feature=input_feature, hidden_feature=opt.interpolator_hidden_nodes, output_feature=opt.interpolator_seq_len, p_dropout=0.2, num_stage=opt.interpolator_num_stage, node_n=num_joint*3, pred_duration=opt.interpolator_pred_duration, use_cnn=opt.interpolator_cnn)
    model.cuda()

    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=opt.lr_gamma)

    #data loaders
    My_Dataset = return_dataset(opt.dataset)

    train_dataset = My_Dataset(opt=opt, cluster_n=cluster_num, cluster_centers=initial_cluster_centers, 
                                actions=all_actions, split="train", nosplits=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.train_batch,
                                shuffle=True,num_workers=opt.job,pin_memory=True)
    print("Training data is loaded")

    val_dataset = My_Dataset(opt=opt, cluster_n=cluster_num, cluster_centers=initial_cluster_centers,
                                actions=all_actions, split="val", nosplits=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=opt.test_batch,
                                shuffle=True,num_workers=opt.job,pin_memory=True)
    print("Validation data is loaded")
    if opt.run_test:
        test_loader = dict()
        for act in all_actions:
            test_dataset = My_Dataset(opt=opt, cluster_n=cluster_num, cluster_centers=initial_cluster_centers,
                                        actions=[act], split="test", nosplits=True)
            test_loader[act] = DataLoader(dataset=test_dataset, batch_size=opt.test_batch,
                                        shuffle=True,num_workers=opt.job,pin_memory=True)
    print("Test data is loaded")

    lr_now = opt.lr
    for epoch in range(opt.interpolator_epochs):
        logger_tb.update_epoch(epoch)
        logger_tr.update_epoch(epoch)
        if (epoch + 1) % opt.lr_decay == 0 and scheduler.get_last_lr()[0]>opt.lr*1e-4:
            scheduler.step()

        update_figures = (epoch+1)%update_figures_epoch==0

        print('==========================')
        print('>>> epoch: {}'.format(epoch + 1))

        # training
        if not opt.is_eval:
            _, loss_dict_tr = run(mode="train", update_figures=update_figures,
                                            opt=opt, data_loader=train_loader, 
                                            model=model, optimizer=optimizer,
                                            epoch=epoch,  writer=logger_tb.writer_tr,
                                            dim_used=train_dataset.dim_used,
                                            action="all")



            #log training losses
            logger_tb.write_average_losses_tb("train", loss_dict_tr)
            logger_tr.update_values_from_dict(loss_dict_tr)
            logger_tr.save_csv_log()

            # validation 
            logger_val.update_epoch(epoch)
            with torch.no_grad():
                _, loss_dict_val = run(mode="val", opt=opt, 
                                    update_figures=update_figures, data_loader=val_loader, 
                                    model=model, optimizer=None,
                                    epoch=epoch,  writer=logger_tb.writer_val,
                                    dim_used=train_dataset.dim_used,
                                    action="all")

            #logging validation errors:
            logger_tb.write_average_losses_tb("val", loss_dict_val)
            logger_val.update_values_from_dict(loss_dict_val)
            sum_v_err = loss_dict_val["loss"]

            if not np.isnan(sum_v_err):
                is_best = sum_v_err < err_best
                err_best = min(sum_v_err, err_best)
            else:
                is_best = False

            ## save model we just trained
            file_name = ['model_' + run_info + '_best.pth.tar', "model_" + run_info + '_last.pth.tar']
            save_ckpt({'epoch': epoch + 1,
                        'err': sum_v_err,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                    ckpt_path=opt.ckpt_datetime,
                    is_best=is_best,
                    file_name=file_name)
            logger_val.save_csv_log()

        logger_test.update_epoch(epoch)
        loss_dict_test = {"loss":0, "bone":0, "mse":0, "velocity":0}
        for act in all_actions:
            with torch.no_grad():
                _, loss_dict_test_act = run(mode="test", opt=opt, 
                                        update_figures=update_figures,data_loader=test_loader[act], 
                                        model=model, optimizer=None,
                                        epoch=epoch,  writer=logger_tb.writer_test, 
                                        dim_used=train_dataset.dim_used, action=act)
            for key, _ in loss_dict_test.items():
                loss_dict_test[key] +=  loss_dict_test_act[key]/len(all_actions)



            
        #logging validation errors:
        logger_tb.write_average_losses_tb("test", loss_dict_test)
        logger_test.update_values_from_dict(loss_dict_test)
        logger_test.save_csv_log()


if __name__ == "__main__":
    option = Options().parse()
    train_interpolator(option)
