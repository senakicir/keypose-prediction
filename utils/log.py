#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import json
import os
import torch
import pandas as pd
import numpy as np

from torch.utils.tensorboard import SummaryWriter


class Logger_TB(object):
    def __init__(self, loc):
        self.writer_tr = SummaryWriter(loc+'/tr')
        self.writer_val = SummaryWriter(loc+'/val')
        self.writer_test = SummaryWriter(loc+'/test')
        self.writer_dict={"train":self.writer_tr, "val":self.writer_val, "test":self.writer_test}

    def update_epoch(self, epoch):
        self.epoch = epoch

    def write_average_losses_tb(self, mode, losses, omit_keys=[]):
        writer = self.writer_dict[mode]
        for loss_name, loss in losses.items():
            if (loss_name not in omit_keys):
                if loss != 0:
                    writer.add_scalar('loss/'+str(loss_name), loss, self.epoch+1)

    def write_best_epoch_acc(self, epoch, OMAC, record_vals):
        my_text = "best epoch: {}.".format(epoch) 
        my_text += ". Accuracies OMAC:" 
        for key, value in OMAC.items():
            if key != "conf_mat" and key!= "ce_loss":
                if type(value)==int:
                    my_text += "{:.3f}, ".format(value)
                else:
                    my_text += "{:.3f}, ".format(value.item())
        my_text += ". Other values:"
        for key, value in record_vals.items():
            my_text += key + ": {:.3f},".format(value)
        
        self.writer_test.add_text('best_model', my_text, epoch)

    def write_training_time(self, tt):
        self.writer_test.add_text('training_time', str(tt), 100)

class Logger_CSV(object):
    
    def __init__(self, save_loc, file_name="test"):
        self.values = np.array([])
        self.head = np.array([])
        self.file_name = file_name
        self.save_loc = save_loc

    def update_epoch(self, epoch):
        self.values  = np.array([epoch + 1])
        self.head = np.array(['epoch'])

    def update_values_from_dict(self, value_dict, header_prefix="", omit_keys=[]):
        for key, value in value_dict.items():
            if key not in omit_keys:
                self.values  = np.append(self.values, [value])
                self.head = np.append(self.head, [header_prefix+str(key)])

    def update_value(self, head, value):
        self.values  = np.append(self.values, [value])
        self.head = np.append(self.head, [head])

    def save_csv_log(self):
        if len(self.values.shape) < 2:
            self.values = np.expand_dims(self.values, axis=0)
        df = pd.DataFrame(self.values)
        file_path = self.save_loc + '/{}.csv'.format(self.file_name)
        if not os.path.exists(file_path):
            df.to_csv(file_path, header=self.head, index=False)
        else:
            with open(file_path, 'a') as f:
                df.to_csv(f, header=False, index=False)



def save_ckpt(state, ckpt_path, is_best=True, file_name=['ckpt_best.pth.tar', 'ckpt_last.pth.tar']):
    file_path = os.path.join(ckpt_path, file_name[1])
    torch.save(state, file_path)
    if is_best:
        file_path = os.path.join(ckpt_path, file_name[0])
        torch.save(state, file_path)


def save_options(opt):
    with open(opt.ckpt + '/option.json', 'w') as f:
        f.write(json.dumps(vars(opt), sort_keys=False, indent=4))
