import os
import time
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
import numpy as np
from progress.bar import Bar
import random

from .. import data_utils as data_utils
from .. import visualization_client as vis_client
from ..utils import AccumLoss 
from .interpolator_utils import resample_sequence_from_dur, interpolator_loss_function, linearly_interpolate_values2

def run(mode, opt, update_figures, data_loader, model, epoch, writer, optimizer, dim_used=[], action="all"):
    save_loc = opt.ckpt_datetime

    fig_location = save_loc+'/figs/'+str(epoch+1)+"/"+mode+"/"+action
    if update_figures:
        if not os.path.exists(fig_location):
            os.makedirs(fig_location)

    if mode=="train":
        model.train()
    else: #val or test
        model.eval()
    
    st = time.time()
    loss_dict_batch = {}

    bar = Bar('>>>', fill='>', max=len(data_loader))

    for i, (all_seq, keypose_vals, keypose_durations, _, _) in enumerate(data_loader):

        bt = time.time()
        all_seq = all_seq.cuda()
        keypose_vals = keypose_vals.cuda()
        batch_size = all_seq.shape[0]
        orig_keypose_durations = keypose_durations.clone()
        interp_net_input = keypose_vals


        all_seq_input = all_seq[:, :, dim_used]
    
        # linear interpolator
        max_duration = torch.max(torch.sum(keypose_durations[:,1:], dim=1))
        min_duration = torch.min(torch.sum(keypose_durations[:,1:], dim=1))
        durations = torch.sum(keypose_durations[:,1:], dim=1)

        linear_interp_tmp = linearly_interpolate_values2(interp_net_input, keypose_durations, max_duration+1)

        interp_net_input = torch.zeros([batch_size, opt.interpolator_seq_len, linear_interp_tmp.shape[2]]).float().cuda()
        if min_duration < opt.interpolator_seq_len:
            interp_net_input[durations<opt.interpolator_seq_len] = resample_sequence_from_dur(linear_interp_tmp[durations<opt.interpolator_seq_len], keypose_durations[durations<opt.interpolator_seq_len], opt.interpolator_seq_len)
        if max_duration >= opt.interpolator_seq_len:
            interp_net_input[durations>=opt.interpolator_seq_len] = linear_interp_tmp[durations>=opt.interpolator_seq_len][:,:opt.interpolator_seq_len]


        residual_reconst, _ = model(interp_net_input.permute(0,2,1))
        reconst = residual_reconst.permute(0,2,1)        

        resampled_interval = torch.zeros([batch_size, opt.interpolator_seq_len, len(dim_used)]).float().cuda()
        max_duration = torch.max(torch.sum(orig_keypose_durations[:,1:], dim=1))
        min_duration = torch.min(torch.sum(orig_keypose_durations[:,1:], dim=1))
        durations = torch.sum(orig_keypose_durations[:,1:], dim=1)
        if min_duration < opt.interpolator_seq_len:
            resampled_interval[durations<opt.interpolator_seq_len] = resample_sequence_from_dur(all_seq_input[durations<opt.interpolator_seq_len], orig_keypose_durations[durations<opt.interpolator_seq_len], opt.interpolator_seq_len)
        if max_duration >= opt.interpolator_seq_len:
            resampled_interval[durations>=opt.interpolator_seq_len] = all_seq_input[durations>=opt.interpolator_seq_len][:,:opt.interpolator_seq_len] 
        resampled_interval =  Variable(resampled_interval).float()

        loss_dict = interpolator_loss_function(opt, resampled_interval, reconst)

        if mode == "train":
            optimizer.zero_grad()
            loss = loss_dict["loss"] 
            loss.backward()
            optimizer.step()

        for key, val in loss_dict.items():
            if key not in loss_dict_batch:
                loss_dict_batch[key] = AccumLoss()
            loss_dict_batch[key].update(val.item(), 1)

        bar.suffix = '{}/{}|loss{:.3f}|batch time {:.3f}s|total time{:.2f}s.'.format(i + 1, len(data_loader), loss_dict_batch["loss"].avg,
                                                                            time.time() - bt,time.time() - st)
        bar.next()

    bar.finish() 

    # figures
    if update_figures:
        batch_size, seq_len = reconst.shape[0], reconst.shape[1]
        gt_p3d = resampled_interval.reshape(batch_size, seq_len ,-1,3).permute(0,1,3,2)
        output_p3d = reconst.reshape(batch_size, seq_len ,-1,3).permute(0,1,3,2)
        input_p3d_pre = interp_net_input[:,:,:len(dim_used)].reshape(batch_size, seq_len ,-1,3).permute(0,1,3,2)
    
        for poses_in_batch in range(min(batch_size,1)):
            ind_list = list(range(0,seq_len,1))
            for ind in ind_list:
                visualize_pose_gt=gt_p3d[poses_in_batch,ind,:,:].detach().cpu().numpy()
                visualize_pose_decode=output_p3d[poses_in_batch,ind,:,:].detach().cpu().numpy()
                visualize_pose_pre=input_p3d_pre[poses_in_batch,ind,:,:].detach().cpu().numpy()
                vis_client.display_poses([visualize_pose_gt, visualize_pose_pre, visualize_pose_decode], data_utils.all_bone_connections_after_subsampling("all", dataset=opt.dataset), save_loc=fig_location+"/", custom_name=str(poses_in_batch)+"_", time=ind)

    final_dict = {}
    for key, _ in loss_dict_batch.items():
        final_dict[key] =  loss_dict_batch[key].avg

    return fig_location, final_dict
