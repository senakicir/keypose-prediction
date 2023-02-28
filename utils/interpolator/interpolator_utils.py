import numpy as np 
import torch
import torch.nn as nn

from .. import data_utils as data_utils


ce_loss = nn.CrossEntropyLoss()
torch.cuda.set_device(0)


def gauss(n=32):
    inv = torch.cat([torch.arange(5)/5.0, torch.ones([n-10]), torch.arange(4,-1,-1)/5.0])
    return 1.0-inv

def linearly_interpolate_values(pred_vals, pred_durations, predict_frame_goal, offset_n=None):
    batch_size, keypose_num, joint_dim = pred_vals.shape
    output_seq = torch.zeros((batch_size, predict_frame_goal, joint_dim)).to(pred_vals.device)
    if offset_n is None:
        offset_n = torch.zeros([batch_size]).long().to(pred_vals.device)
    for batch in range(batch_size):
        output_seq_single = torch.zeros((0, joint_dim)).to(pred_vals.device)
        
        #initial poses
        keypose = pred_vals[batch, 0, :]
        output_seq_single= torch.cat((output_seq_single, keypose.unsqueeze(0)), dim=0)

        for count in range(0, keypose_num-1):
            keypose_next = pred_vals[batch, count+1, :]         
            durations = pred_durations[batch, count+1].float() #(1)
            durations[durations<1] = 1.0
            delta_poses = ((keypose_next-keypose)/(durations+1)).unsqueeze(0) #(1,3*22)
            time = torch.arange(1, durations.item()+1).reshape((int(durations.item()), 1)).to(pred_vals.device) #(durations, 1)
            interpolated_poses = keypose.unsqueeze(0) + time*delta_poses
            output_seq_single= torch.cat((output_seq_single, interpolated_poses), dim=0)
            keypose = keypose_next.clone()
        output_seq[batch, :, :] = output_seq_single[offset_n[batch]:offset_n[batch]+predict_frame_goal, :]
    return output_seq


def linearly_interpolate_values2(pred_vals, pred_durations, predict_frame_goal):
    batch_size, keypose_num, joint_dim = pred_vals.shape
    if type(predict_frame_goal) == torch.Tensor:
        predict_frame_goal = int(predict_frame_goal.item())
    output_seq = torch.zeros((batch_size, predict_frame_goal, joint_dim)).to(pred_vals.device)
    
    for batch in range(batch_size):
        output_seq_single = torch.zeros((0, joint_dim)).to(pred_vals.device)
        
        #initial poses
        keypose = pred_vals[batch, 0, :]
        output_seq_single= torch.cat((output_seq_single, keypose.unsqueeze(0)), dim=0)

        for count in range(0, keypose_num-1):
            keypose_next = pred_vals[batch, count+1, :]         
            durations = pred_durations[batch, count+1].float() #(1)
            durations[durations<1] = 1.0
            delta_poses = ((keypose_next-keypose)/(durations+1)).unsqueeze(0) #(1,3*22)
            time = torch.arange(1, durations.item()+1).reshape((int(durations.item()), 1)).to(pred_vals.device) #(durations, 1)
            interpolated_poses = keypose.unsqueeze(0) + time*delta_poses
            output_seq_single= torch.cat((output_seq_single, interpolated_poses), dim=0)
            keypose = keypose_next.clone()
        output_seq[batch, :output_seq_single.shape[0], :] = output_seq_single
        
    return output_seq


def resample_sequence_from_dur(input_seqs, durations, resampled_seq_len=25):
    resampled_output_seq = torch.zeros([input_seqs.shape[0], resampled_seq_len, input_seqs.shape[2]]).cuda()
    input_seq_len = input_seqs.shape[1]
    for batch_ind in range(input_seqs.shape[0]):
        input_seq_batch = input_seqs[batch_ind]
        durations_batch = durations[batch_ind]
        total_seq_duration = torch.sum(durations_batch[1:]).item()+1
        clipped_input = input_seq_batch[:min(int(total_seq_duration), input_seq_len), :]
        resampled_output_seq[batch_ind] = resample_seq(clipped_input, resampled_seq_len)        
    return resampled_output_seq

def resample_seq_batch(input_seq_batch, duration_target):
    resampled_output = torch.zeros([input_seq_batch.shape[0], duration_target, input_seq_batch.shape[2]]).cuda()
    sampling_factor = (input_seq_batch.shape[1]-1)/(duration_target-1)

    indices = (torch.arange(0,duration_target-1)*sampling_factor).cuda()
    time_indices = torch.floor(indices).long()
    weights = (1.-(indices - time_indices)).unsqueeze(1)    
    resampled_output[:, 0:-1, :] = (weights*input_seq_batch[:,time_indices, :] + (1.-weights)*input_seq_batch[:,time_indices+1, :])

    resampled_output[:,0] = input_seq_batch[:,0]
    resampled_output[:,-1]= input_seq_batch[:,-1]
    return resampled_output

def resample_seq(input_seq, duration_target):
    resampled_output = torch.zeros([duration_target, input_seq.shape[1]]).cuda()
    sampling_factor = (input_seq.shape[0]-1)/(duration_target-1)

    indices = (torch.arange(0,duration_target-1)*sampling_factor).cuda()
    time_indices = torch.floor(indices).long()
    weights = (1.-(indices - time_indices)).unsqueeze(1)    
    resampled_output[0:-1, :] = (weights*input_seq[time_indices, :] + (1.-weights)*input_seq[time_indices+1, :])

    resampled_output[0] = input_seq[0]
    resampled_output[-1]= input_seq[-1]
    return resampled_output
        

def interpolator_loss_function(opt, seq_input, recons):
    batch_size, seq_len, dims = seq_input.shape

    """
    :param args:
    :param kwargs:
    :return:
    """

    dataset = opt.dataset
    bone_loss =  opt.interpolator_bone_loss
    velocity_loss = opt.interpolator_velocity_loss

    assert recons.shape == seq_input.shape
    recons_rsh = recons.reshape(batch_size, seq_len, -1, 3)
    input_rsh = seq_input.reshape(batch_size, seq_len, -1, 3)
 
    loss = torch.zeros([1]).cuda()
    loss_bone_loss =  torch.zeros([1])
    loss_velocity = torch.zeros([1])
    loss_mse = torch.zeros([1])


    loss_mse =torch.mean(torch.mean(torch.mean(torch.mean((recons_rsh-input_rsh)**2,dim=3), dim=2), dim=1), dim=0)*1e-1
    loss = loss + loss_mse

    if bone_loss > 0.0:        
        bone_connections = np.array(data_utils.all_bone_connections_after_subsampling("all", dataset=dataset))
        bone_len_gt = torch.mean(torch.mean(torch.mean((input_rsh[:,:,bone_connections[:,0], :]- input_rsh[:,:,bone_connections[:,1], :])**2, dim=3), dim=1),dim=0).unsqueeze(0).unsqueeze(1) #1, num_bone
        bone_len_mine = torch.mean((recons_rsh[:,:,bone_connections[:,0], :]- recons_rsh[:,:,bone_connections[:,1], :])**2, dim=3) #B, N, num_bone
        loss_bone_loss = torch.mean(torch.mean(torch.mean((bone_len_gt-bone_len_mine)**2, dim=2), dim=1), dim=0) *bone_loss
        loss = loss + loss_bone_loss 

    if velocity_loss > 0.0:
        input_vel = input_rsh[:,1:] - input_rsh[:,:-1]# B, N-1, J, 3
        reconst_vel = recons_rsh[:,1:] -  recons_rsh[:,:-1]
        loss_velocity =torch.mean(torch.mean(torch.mean(torch.mean((input_vel-reconst_vel)**2,dim=3), dim=2), dim=1), dim=0)*velocity_loss
        loss = loss + loss_velocity

 
    return {'loss': loss, "bone":loss_bone_loss,  'velocity': loss_velocity, 'mse':loss_mse}


def categorize_durations(durations):
    cat = [24, 48]
    categorized_durations = torch.zeros([durations.shape[0]]).cuda().long()
    categorized_durations[durations<cat[0]] = 0
    categorized_durations[torch.logical_and(cat[0]<durations, durations<=cat[1])] = 1
    categorized_durations[(durations>cat[1])] = 2
    return categorized_durations

def calculate_duration_acc(pred_durations, gt_durations):
    pred_durations_label = torch.argmax(pred_durations, dim=1)
    return torch.mean((pred_durations_label==gt_durations).float())*100
  
