#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import math
from pprint import pprint
import datetime


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================/Users/kicirogl/workspace
        self.parser.add_argument('--data_dir', type=str, default='/cvlabsrc1/cvlab/Human36m/H36M_exp_map/h3.6m/dataset', help='path to H36M dataset')
        self.parser.add_argument('--cmu_data_dir', type=str, default='/cvlabsrc1/cvlab/dataset_cmu_mocap/', help='path to CMU-Mocap dataset')

        self.parser.add_argument('--dataset', type=str, default='h36m', help='h36m or cmu')

        self.parser.add_argument('--exp', type=str, default='test', help='ID of experiment')
        self.parser.add_argument('--ckpt', type=str, default='checkpoint/', help='path to save checkpoint')
        self.parser.add_argument('--seed', type=int, default=0, help='')
        self.parser.add_argument('--use_gt_durations', type=str2bool, default=False)

        self.parser.add_argument('--oracle', type=str2bool, default=False)
        self.parser.add_argument('--fixed_time', type=int, default=15)


        # ===============================================================
        #                     Keypose options
        # ===============================================================
        self.parser.add_argument('--keypose_dir', type=str, default='kp_datasets', help='path to processed keyposes of H36M dataset')
        self.parser.add_argument('--kp_threshold', type=int, default=500)
        self.parser.add_argument('--kp_suffix', type=str, default="_3dv2022")
        self.parser.add_argument('--cluster_n', type=int, default=1000, help='number of keypose clusters')
        self.parser.add_argument('--reevaluate_keyposes', type=str2bool, default=False)
        self.parser.add_argument('--load_clusters', type=str2bool, default=False)

        # ===============================================================
        #                     Action Recognition options
        # ===============================================================
        self.parser.add_argument('--ar_lr', type=float, default=1e-5)
        self.parser.add_argument('--ar_seq_len', type=int, default=125)
        self.parser.add_argument('--ar_epochs', type=int, default=500)
        self.parser.add_argument('--ar_window_size', type=int, default=32)
        self.parser.add_argument('--ar_weight_decay', type=float, default=1e-4)
        self.parser.add_argument('--ar_motion_only', type=str2bool, default=True)

        # 125 seq
        self.parser.add_argument('--ar_model_path', type=str, default='pretrained/action_recognition/fac')
        # motion only:
        self.parser.add_argument('--omac_ar_model_path', type=str, default='pretrained/action_recognition/omac')


        # ===============================================================
        #                     Interpolator options
        # ===============================================================
        self.parser.add_argument('--interpolator_num_stage', type=int, default=10, help='')
        self.parser.add_argument('--interpolator_hidden_nodes', type=int, default=512, help='')
        self.parser.add_argument('--interpolator_duration_lambda', type=float, default=100.0)
        self.parser.add_argument('--interpolator_epochs', type=int, default=50)
        self.parser.add_argument('--interpolator_kp_num', type=int, default=10, help='')
        self.parser.add_argument('--interpolator_seq_len', type=int, default=125, help='')

        self.parser.add_argument('--interpolator_model_path', type=str, default='pretrained/interpolator')
        self.parser.add_argument('--use_interpolator', type=str2bool, default=False, help='')
        self.parser.add_argument('--interpolator_dur', type=int, default=42, help='')
        self.parser.add_argument('--interpolator_sample_seq_len', type=str2bool, default=True, help='')
        self.parser.add_argument('--interpolator_bone_loss', type=float, default=1e-6, help='')
        self.parser.add_argument('--interpolator_velocity_loss', type=float, default=100, help='')


        # ===============================================================
        #                     Model options
        # ===============================================================
        self.parser.add_argument('--kp_hidden_size', type=int, default=512)
        self.parser.add_argument('--kp_num_gru_layers', type=int, default=2)
        self.parser.add_argument('--kp_num_fc_layers', type=int, default=1)
        self.parser.add_argument('--augment_output', type=str2bool, default=True, help='whether or not to augment output of model')
        self.parser.add_argument('--kp_model_path',type=str, default='pretrained/kp_model', help='path to pretrained model')

        self.parser.add_argument('--duration_lambda', type=float, default=1e-1)
        self.parser.add_argument('--KL_lambda', type=float, default=1.0)
        self.parser.add_argument('--crossentropy_lambda', type=float, default=0.0)
        self.parser.add_argument('--oracle_fixed_time', type=str2bool, default=False)

        self.parser.add_argument('--is_diverse', type=str2bool, default=False)
        self.parser.add_argument('--diverse_seq_num', type=int, default=100, help='num of diverse seq to generate')
        self.parser.add_argument('--use_lr_decay', type=str2bool, default=False)
        self.parser.add_argument('--use_ss', type=str2bool, default=True)

        self.parser.add_argument('--smoothing', type=float, default=0.0)
        self.parser.add_argument('--kp_model_type', type=str, default="v1", help='choices: oracle, v1, v2')
        self.parser.add_argument('--final_frame_kp', type=str2bool, default=True)

        self.parser.add_argument('--ss_k', type=float, default=10, help='bigger k means more tf')
        self.parser.add_argument('--ss_offset', type=float, default=5, help='when to start scheduling')
        self.parser.add_argument('--kd_t', type=float, default=0.03, help='t of knowledge distillation, smaller means more exaggeration')
        self.parser.add_argument('--sampling_kd_t', type=float, default=0.3, help='t of knowledge distillation, smaller means more exaggeration')
        self.parser.add_argument('--use_tf_mode', type=str, default="ss", help="alwaystf, nevertf, ss")

        self.parser.add_argument('--kd_cutoff', type=float, default=0.01, help='how few labels we would like (higher number means fewer labels)')
        self.parser.add_argument('--scale', type=str, default="scale_sum", help='scale_sum or scale_max')
        
        self.parser.add_argument('--duration_category_num', type=int, default=5)
        
        # ===============================================================
        #                     Evaluation options
        # ===============================================================
        self.parser.add_argument('--is_eval',type=str2bool, default=False, help='whether to evaluate existing model')
        self.parser.add_argument('--save_figs', type=str2bool, default=False)
        self.parser.add_argument('--run_test', type=str2bool, default=True)
        self.parser.add_argument('--overfitting_exp', type=str2bool, default=False) 
        self.parser.add_argument('--run_static', type=str2bool, default=False) 
        self.parser.add_argument('--train_batch', type=int, default=64)
        self.parser.add_argument('--test_batch', type=int, default=64)
        self.parser.add_argument('--drop_last', type=str2bool, default=True)
        self.parser.add_argument('--job', type=int, default=10, help='subprocesses to use for data loading')
        self.parser.add_argument('--is_load', type=str2bool, default=False, help='whether to load existing model')

        # ===============================================================
        #                     Optimizer options
        # ===============================================================
        self.parser.add_argument('--lr', type=float, default=1e-4)
        self.parser.add_argument('--momentum', type=float, default=0.99)
        self.parser.add_argument('--lr_decay', type=int, default=200, help='every lr_decay epoch do lr decay')
        self.parser.add_argument('--lr_gamma', type=float, default=0.80)
        self.parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
        self.parser.add_argument('--optimizer', type=str, default="adam", help='options: sgd, adam')
        self.parser.add_argument('--max_norm', type=float, default=500, help='')


        self.parser.add_argument('--cc_lr', type=float, default=1e-5)
        self.parser.add_argument('--cc_momentum', type=float, default=0.99)
        self.parser.add_argument('--cc_weight_decay', type=float, default=0.001, help='weight decay')
        self.parser.add_argument('--cc_opt_freq', type=int, default=10, help='')
        self.parser.add_argument('--cc_opt_len', type=int, default=3, help='')
        self.parser.add_argument('--cc_max_norm', type=float, default=500, help='')

        self.parser.add_argument('--cc_optimizer', type=str, default="adam", help='options: sgd, adam')
        self.parser.add_argument('--optimize_cluster_centers', type=str2bool, default=False, help='whether we optimize cluster centers') 

        # ===============================================================
        #                     Running options
        # ===============================================================
        self.parser.add_argument('--epochs', type=int, default=100)
        self.parser.add_argument('--test_subj', type=int, default=5, help='test subject')

        ## sequence size options
        self.parser.add_argument('--input_kp', type=int, default=7, help='observed num of keyposes')
        self.parser.add_argument('--output_kp', type=int, default=12, help='future keyposes')
        self.parser.add_argument('--output_seq_n', type=int, default=125, help='future sequence length')
        self.parser.add_argument('--input_seq_n', type=int, default=100, help='observed sequence length (estimate)')

        ## data options
        self.parser.add_argument('--actions', type=str, default='all', help='path to save checkpoint')
        self.parser.add_argument('--sample_rate', type=int, default=2, help='frame sampling rate')

        self.parser.add_argument('--supervise_past', type=str2bool, default=True, help='')        
        self.parser.add_argument('--input_noise_magnitude', type=float, default=0.1, help='magnitude of noise we will add to the data')
        self.parser.add_argument('--random_shuffle', type=str2bool, default=False, help='parameter on whether to add a random shuffle (for data aug)')
        self.parser.add_argument('--shuffle_prob', type=float, default=0.20)
        self.parser.add_argument('--is_norm', dest='is_norm', action='store_true', help='whether to normalize the angles/3d coordinates')

        self.parser.add_argument('--update_losses_epoch', type=int, default=1)
        self.parser.add_argument('--update_figures_epoch', type=int, default=20)
        self.parser.add_argument('--train_mse_epoch', type=int, default=100)
        self.parser.add_argument('--test_epoch', type=int, default=1000)
        self.parser.add_argument('--val_epoch', type=int, default=1)
        
        self.parser.add_argument('--note', type=str, default="", help='any other notes')

    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()
        # do some pre-check
        ckpt = os.path.join(self.opt.ckpt, self.opt.exp)
        if not os.path.isdir(ckpt):
            os.makedirs(ckpt)
        self.opt.ckpt = ckpt
        self._print()
        return self.opt

def setup_folder(opt):
    date_time= datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    ckpt_datetime = opt.ckpt+'/'+date_time
    while os.path.exists(ckpt_datetime):
        ckpt_datetime += "_x"
    os.makedirs(ckpt_datetime)
    opt.ckpt_datetime = ckpt_datetime

def save_opt(opt, writer):
    with open(opt.ckpt_datetime+'/args.txt', 'w') as f:
        my_str = ""
        for key, value in vars(opt).items():
            if not key == "note":
                my_str += str(key)+": "+str(value)+"\n"
            elif value != "":
                my_str += "\n********\nNOTE: "+str(value)+"\n"
        f.write(my_str)
        writer.add_text("notes/", my_str)