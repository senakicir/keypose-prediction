#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import os
import numpy as np
import time
import math

class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def lr_decay(optimizer, lr_now, gamma):
    lr = lr_now * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def average_errs_across_actions(errors, acts):
    loss_dict = {}
    for action, loss_dict_action in errors.items():
        for loss_name, loss_val in loss_dict_action.items():
            if loss_name not in loss_dict:
                loss_dict[loss_name] = 0
            loss_dict[loss_name] += loss_val/len(acts) 
    return loss_dict

##############