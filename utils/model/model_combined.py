import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
from .model_kp import KP_GRU_Prob,KP_GRU_Prob_v2, KP_Oracle, KP_GRU_Prob_v3

import utils.data_utils as data_utils
from utils.action_recognition.model import HCN
from utils.interpolator.model import KP_Interpolator
from utils.log import save_ckpt


class Combined_Model(nn.Module):
    def __init__(self, opt=None, cluster_n=None):
        super(Combined_Model, self).__init__()

        self.kp_hidden_size = opt.kp_hidden_size
        self.kp_gru_layers = opt.kp_num_gru_layers
        self.kp_fc_layers = opt.kp_num_fc_layers

        self.output_seq_n = opt.output_seq_n
        self.input_seq_n = opt.input_seq_n

        self.oracle = opt.oracle
        nonmoving_joint_num = data_utils.joint_num(opt.dataset)-len(data_utils.nonmoving_joints(opt.dataset))

        if opt.use_interpolator:
            input_feature=opt.interpolator_seq_len
            self.interpolator_model = KP_Interpolator(input_feature=input_feature, hidden_feature=opt.interpolator_hidden_nodes, output_feature=opt.interpolator_seq_len, p_dropout=0.2, num_stage=opt.interpolator_num_stage, node_n=nonmoving_joint_num*3)
            for param in self.interpolator_model.parameters():
                param.requires_grad = False
            self.interpolator_model.eval() 
        
        if opt.oracle:
            self.kp_model = KP_Oracle()
        else:
            if opt.kp_model_type == "v1":
                self.kp_model = KP_GRU_Prob(cluster_n, self.kp_hidden_size,  opt.duration_category_num)
            if opt.kp_model_type == "v2":
                self.kp_model = KP_GRU_Prob_v2(cluster_n,self.kp_hidden_size,  opt.duration_category_num)
            if opt.kp_model_type == "v3":
                self.kp_model = KP_GRU_Prob_v3(cluster_n,self.kp_hidden_size, opt.duration_category_num)
            
        self.trainable_model = self.kp_model

        all_actions = data_utils.define_actions("all", opt.dataset, False)
        self.ar_model_full = HCN(in_channel=3,
                num_joint= nonmoving_joint_num,
                out_channel=64,window_size=opt.ar_window_size,seq_len = opt.ar_seq_len,
                num_class = len(all_actions), motion_only=False)
        self.ar_model_omac = HCN(in_channel=3,
                num_joint= nonmoving_joint_num,
                out_channel=64,window_size=opt.ar_window_size,seq_len = opt.ar_seq_len,
                num_class = len(all_actions), motion_only=True)
        for param in self.ar_model_full.parameters():
            param.requires_grad = False
        self.ar_model_full.eval() 
        for param in self.ar_model_omac.parameters():
            param.requires_grad = False
        self.ar_model_omac.eval() 

        

    def load_weights(self, model_pth, opt, optimizer):
        lr_now = opt.lr
        acc_best = 0
        start_epoch = 0
        self.ar_model_full.load_weights('FAC_{}_seq_len{}'.format(opt.dataset, opt.ar_seq_len), opt.ar_model_path)
        self.ar_model_omac.load_weights('OMAC_{}_seq_len{}'.format(opt.dataset, opt.ar_seq_len), opt.omac_ar_model_path)
        if opt.use_interpolator:
            self.interpolator_model.load_weights('interpolate_{}'.format(opt.dataset), opt)
        if not self.oracle:
            if (opt.is_load or opt.is_eval):
                #load best model
                model_path_len = opt.kp_model_path + "/model_" + model_pth + '_best.pth.tar'
                print(">>> loading ckpt kp from '{}'".format(model_path_len))
                ckpt = torch.load(model_path_len)
                start_epoch = ckpt['epoch']
                acc_best = ckpt['acc']
                self.kp_model.load_state_dict(ckpt['state_dict'])
                optimizer.load_state_dict(ckpt['optimizer'])
                optimizer.load_lr(ckpt['lr'])
                if opt.is_eval:
                    print(">>> ckpt kp loaded (epoch: {} | acc: {})".format(start_epoch, acc_best))

                #resave loaded best model
                if opt.is_load:
                    self.save_model(model_pth, start_epoch, optimizer, acc_best, opt.ckpt_datetime, True)

                    #save last model
                    model_path_len = opt.kp_model_path + "/model_" + model_pth + '_last.pth.tar'
                    print(">>> loading ckpt kp from '{}'".format(model_path_len))
                    ckpt = torch.load(model_path_len)
                    start_epoch = ckpt['epoch']
                    last_acc = ckpt['acc']
                    self.kp_model.load_state_dict(ckpt['state_dict'])
                    optimizer.load_state_dict(ckpt['optimizer'])
                    optimizer.load_lr(ckpt['lr'])

        return acc_best, start_epoch


    def save_model(self, run_info, epoch, optimizer, curr_acc, ckpt_datetime, is_best):
        file_name = ['model_' + run_info + '_best.pth.tar', "model_" + run_info + '_last.pth.tar']
        save_ckpt({'epoch': epoch + 1,
                'lr': optimizer.lr_state(),
                'acc': curr_acc,
                'state_dict': self.trainable_model.state_dict(),
                'optimizer': optimizer.state_dict()},
                ckpt_path=ckpt_datetime,
                is_best=is_best,
                file_name=file_name)

    def switch_mode(self, mode):
        if mode=="val" or mode=="test":
            self.eval()
            self.kp_model.eval()

        elif mode=="train":
            self.kp_model.train()
            self.train()


    def forward_kp(self, kp_inputs, hidden_states, kp_outputs=None):
        return self.kp_model(kp_inputs, hidden_states, kp_outputs)

    def is_oracle(self):
        return self.kp_model.is_oracle()

    def initHidden(self, is_cuda, batch_size):
        val_hidden = self.kp_model.initHidden( is_cuda, batch_size)
        return val_hidden
