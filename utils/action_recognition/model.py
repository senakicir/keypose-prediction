"""
@author: huguyuehuhu, senakicir (modified)
@time: 18-4-16 下午6:51
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import os

from . import utils_ar as utils


class HCN(nn.Module):
    '''
    Input shape:
    Input shape should be (N, C, T, V, M)
    where N is the number of samples,
          C is the number of input channels,
          T is the length of the sequence,
          V is the number of joints
      and M is the number of people.
    '''
    def __init__(self,
                 in_channel=3,
                 num_joint=25,
                 num_person=1,
                 out_channel=64,
                 window_size=64,
                 seq_len = 25,
                 num_class = 60, 
                 motion_only=True):

        super(HCN, self).__init__()
        self.num_person = num_person
        self.num_class = num_class
        self.motion_only = motion_only

        # position
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
        )
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3,1), stride=1, padding=(1,0))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2))

        # motion
        self.conv1m = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
        )
        self.conv2m = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3,1), stride=1, padding=(1,0))

        self.conv3m = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        self.conv4m = nn.Sequential(
            nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2))

        # concatenate motion & position
        if self.motion_only:
            in_conv5 = out_channel
        else:
            in_conv5 = out_channel*2

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=in_conv5, out_channels=out_channel*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )

        self.fc7= nn.Sequential(
            nn.Linear((out_channel * 4)*(window_size//16)*(seq_len//16),256*2), # 4*4 for window=64; 8*8 for window=128
            nn.ReLU(),
            nn.Dropout2d(p=0.5))
        self.fc8 = nn.Linear(256*2,num_class)

        # initial weight
        utils.initial_model_weight(layers = list(self.children()))
        print('weight initial finished!')


    def forward(self, x, target=None):
        N, C, T, V, M = x.size()  # N0, C1, T2, V3, M4
        motion = x[:,:,1::,:,:]-x[:,:,0:-1,:,:]
        motion = motion.permute(0,1,4,2,3).contiguous().view(N,C*M,T-1,V)
        motion = F.interpolate(motion, size=(T,V), mode='bilinear',align_corners=False).contiguous().view(N,C,M,T,V).permute(0,1,3,4,2)

        logits = []
        for i in range(self.num_person):
            # position
            if not self.motion_only:
                # N0,C1,T2,V3 point-level
                out = self.conv1(x[:,:,:,:,i])

                out = self.conv2(out)
                # N0,V1,T2,C3, global level
                out = out.permute(0,3,2,1).contiguous()
                out = self.conv3(out)
                out_p = self.conv4(out)


            # motion
            # N0,T1,V2,C3 point-level
            out = self.conv1m(motion[:,:,:,:,i])
            out = self.conv2m(out)
            # N0,V1,T2,C3, global level
            out = out.permute(0, 3, 2, 1).contiguous()
            out = self.conv3m(out)
            out_m = self.conv4m(out)

            # concat
            if not self.motion_only:
                out = torch.cat((out_p,out_m),dim=1)
            else:
                out = out_m
            out = self.conv5(out)
            out = self.conv6(out)

            logits.append(out)

        # max out logits
        if len(logits) > 1:
            out = torch.max(logits[0],logits[1])
        else:
            out = logits[0]
        

        out = out.view(out.size(0), -1)
        out = self.fc7(out)
        out = self.fc8(out)

        t = out
        assert not ((t != t).any())# find out nan in tensor
        assert not (t.abs().sum() == 0) # find out 0 tensor

        return out

    def load_weights(self, run_info, ar_model_path):
        ar_ckpt = torch.load(ar_model_path+"/model_"+run_info+"_best.pth.tar")
        self.cuda()
        acc_best= ar_ckpt['acc']if "acc" in ar_ckpt else ar_ckpt['err']
        self.load_state_dict(ar_ckpt['state_dict'])
        print(">>> Ar model loaded (best acc: {})".format(acc_best))