import os
import time
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
import numpy as np
from progress.bar import Bar
import pandas as pd
from math import ceil

# from utils import utils as utils
import utils.data_utils as data_utils
import utils.visualization_client as vis_client
from utils.action_recognition.run_ar import iterate_ar_model
from utils.action_recognition.utils_ar import reshape_seq_as_input
from utils.loss_funcs import add_noise_to_prob
from utils.interpolator.interpolator_utils import resample_seq_batch, linearly_interpolate_values

def model_iter(mode, val_pred, val_pred_prev, label_pred, label_prob_pred, duration_pred, hidden_state, 
                val_gt, label_gt_prob,label_gt_prob_noisy, duration_cat_gt, labels, model, 
                loss_function_manager, rng, input_ind, input_kp_num, all_kp_len, past_one_sec, opt):

    # determine whether we will do teacher forcing
    teacher_forcing_prob = rng.scheduled_sampling()
    teacher_forcing = False if (teacher_forcing_prob > rng.ss_prob or mode=="val" or mode =="test") else True

    #first find the input to the model (depending on where I am in the sequence and whether i am doing teacher forcing)
    if input_ind < all_kp_len-1:
        # record future GT (used in loss function)
        next_val_gt = val_gt[:,input_ind+1, :] 
        next_label_gt = labels[:,input_ind+1] 
        next_duration_gt= duration_cat_gt[:,input_ind+1, :]
        next_label_prob_gt = label_gt_prob[:, input_ind+1, :]
        compute_loss=True
        kp_outputs = (next_val_gt, next_label_prob_gt, next_duration_gt)

        #if teacher forcing, we will be feeding the current GT values to predict future
        if teacher_forcing or input_ind < input_kp_num:
            current_val = val_gt[:,input_ind, :] 
            current_label_gt = labels[:,input_ind] 
            if (mode=="val" or mode =="test"):
                current_label_prob = label_gt_prob[:, input_ind, :]
            else:
                current_label_prob = label_gt_prob_noisy[:, input_ind, :]
            current_duration = duration_cat_gt[:,input_ind,:]
            input_to_model = (current_val, current_label_prob, None, current_duration)

        # if not teacher forcing, we will be feeding our own predicted values to predict future
        else: 
            #label_prob_pred = add_noise_to_prob(label_prob_pred, input_noise)
            input_to_model = (val_pred, label_prob_pred, None, duration_pred)

    else: 
        #means that i do not have future GT anymore
        next_label_prob_gt = None#label_gt_prob[:,-1, :] 
        next_label_gt = None#labels[:,-1] 
        next_duration_gt = duration_cat_gt[:, -1, :]
        kp_outputs = (val_gt[:,-1, :], label_gt_prob[:,-1, :], duration_cat_gt[:, -1, :])

        compute_loss = False
        #convert the output to an input
        input_to_model = (val_pred, label_prob_pred, None, duration_pred)
    
    # feed input to the model with the hidden_state
    val_pred, label_logit_pred, duration_pred, hidden_state = model.forward_kp(input_to_model, hidden_state,  kp_outputs)
    # val_pred, label_logit_pred, duration_pred, hidden_state = kp_outputs[0], kp_outputs[1], kp_outputs[2], hidden_state
    # determine cluster label of prediction as well as the corresponding cluster center

    if opt.use_gt_durations:
        duration_pred = next_duration_gt

    if opt.is_diverse:
        cluster_probabilities, duration_probs = loss_function_manager.process_output_for_sampling(label_logit_pred, duration_pred)
        label_pred = rng.sample_label(cluster_probabilities)
        
        duration_cat_sampled = rng.sample_duration(duration_probs)
    else:       
        duration_cat_sampled = None
        label_pred = torch.argmax(label_logit_pred, dim=1).detach()


    val_pred_from_cc = loss_function_manager.value_from_labels(label_pred, rng, mode)

    # val_pred_from_cc = val_pred

    # calculates loss for this iteration
    if compute_loss and (loss_function_manager.supervise_past or (not loss_function_manager.supervise_past and input_ind >= input_kp_num-1)):
        loss_function_manager.update_loss(input_ind, label_logit_pred, duration_pred, next_label_prob_gt, next_label_gt, next_duration_gt)

    val_pred_return = val_pred_from_cc# if not multitask else val_pred
            
    ## detaches and preps outputs 
    input_noise = None if (mode=="val" or mode =="test") else rng.input_noise(label_logit_pred.shape)
    # input_noise = rng.input_noise(label_logit_pred.shape)

    dist_pred, label_pred = loss_function_manager.values_to_dist(val_pred_from_cc.detach())  
    label_prob_pred = loss_function_manager.dist_to_prob(dist_pred, input_noise=input_noise)
    
    duration_pred = loss_function_manager.decategorize_durations(duration_pred.detach(), duration_cat_sampled)            
    duration_pred = loss_function_manager.categorize_durations(duration_pred)
    return val_pred_return.detach(), duration_pred, label_pred, label_prob_pred.detach(), hidden_state


def run(opt, mode, data_loader, interpolate, model, loss_function_manager, 
        epoch, rng, writer, action, mean_pose=None, update_losses=False, update_figures=False, optimizer=None,
        is_cuda=False, dim_used=[], ar_loss_function_manager=None, omac_ar_loss_function_manager=None):

    # input_kp is the number of keyposes we feed as input
    input_kp_num, output_kp_num = opt.input_kp, opt.output_kp
    all_kp_len =  input_kp_num + output_kp_num
    # input_seq_n is the estimated input sequence the keyposes correspond to
    input_seq_n, output_seq_n = opt.input_seq_n, opt.output_seq_n
    save_loc = opt.ckpt_datetime
    dataset = opt.dataset
    
    fig_location = None
    if update_figures:
        fig_location = save_loc+'/figs/'+str(epoch+1)+"/"+mode+"/"+action
        if not os.path.exists(fig_location):
            os.makedirs(fig_location)
        if not os.path.exists(fig_location+"/GT"):
            os.makedirs(fig_location+"/GT")
        if not os.path.exists(fig_location+"/Ours"):
            os.makedirs(fig_location+"/Ours")

    # switches mode to training, val, test
    model.switch_mode(mode)

    loss_function_manager.reset_all() 

    st = time.time()
    bar = Bar('>>>', fill='>', max=len(data_loader))
    
    # data loader loop
    already_generated=False
    for i, (all_seq, keypose_vals, keypose_durations, offset_n, action_labels) in enumerate(data_loader):
        bt = time.time()            
        
        if is_cuda:

            vals_gt = Variable(keypose_vals.cuda(non_blocking=True)).float()
            keypose_dist, labels = loss_function_manager.values_to_dist(vals_gt)

            input_noise = rng.input_noise(keypose_dist.shape)            

            label_gt_prob_noisy = Variable(loss_function_manager.dist_to_prob(keypose_dist, input_noise).cuda(non_blocking=True)).float()
            label_gt_prob = Variable(loss_function_manager.dist_to_prob(keypose_dist, None).cuda(non_blocking=True)).float()

            labels = Variable(labels.cuda(non_blocking=True)).long()
            durations_gt = Variable(keypose_durations.cuda(non_blocking=True)).float()
            duration_cat_gt = Variable(loss_function_manager.categorize_durations(keypose_durations).cuda(non_blocking=True)).float()

            action_labels = Variable(action_labels.cuda(non_blocking=True)).long()
            all_seq_cuda = Variable(all_seq.cuda(non_blocking=True)).float()

        bt0_5 = time.time()

        batch_size, seq_len, num_of_joints = all_seq.shape[0],125,int(all_seq.shape[2]//3)

        #if we are generating multiple futures, set to 5, otherwise predict 1 future
        generate_seq_num = 1 if not opt.is_diverse else opt.diverse_seq_num 

        if model.is_oracle():
            ##cumulative sum of durations.
            future_durations = durations_gt[:, input_kp_num:]
            future_durations_cumulative = torch.cumsum(future_durations, dim=1)
            max_pred_ind = torch.max(torch.sum(future_durations_cumulative<output_seq_n, dim=1))+1
            num_run_interp = ceil(max_pred_ind/ 3)
        else:
            num_run_interp = ceil(output_seq_n/opt.interpolator_seq_len)

        pred_values_diverse = torch.zeros([generate_seq_num , batch_size, seq_len, num_of_joints, 3])
        for generated_ind in range(generate_seq_num):
            bt1 = time.time()
            
            #initialize hidden state
            hidden_state = model.initHidden(is_cuda, batch_size)

            ## record predictions
            pred_label_prob_all = label_gt_prob[:,0,:].unsqueeze(1).cpu()
            pred_durations_all = durations_gt[:,0].unsqueeze(1).cpu()
            pred_values_all = vals_gt[:, 0, :].unsqueeze(1)
            assert durations_gt.shape[1] == all_kp_len
            
            #set all losses to zero
            loss_function_manager.reset_for_batch()
            label_pred = duration_pred = val_pred= val_pred_prev=label_prob_pred=None
            predicted_kp_num = 0
            keep_looping=True
            past_one_sec=False

            while(keep_looping):
                # each iteration of the RNN network
                val_pred, duration_pred, label_pred, label_prob_pred, hidden_state = model_iter(mode,
                                        val_pred, val_pred_prev, label_pred, label_prob_pred, duration_pred, 
                                        hidden_state, vals_gt, label_gt_prob, label_gt_prob_noisy,
                                        duration_cat_gt, labels, model, loss_function_manager, rng, 
                                        predicted_kp_num, input_kp_num, all_kp_len, past_one_sec, opt)
                
                #record the predictions (labels and durations)
                if predicted_kp_num < input_kp_num-1:
                    pred_label_prob_all = torch.cat((pred_label_prob_all, label_gt_prob[:,predicted_kp_num+1,:].unsqueeze(1).cpu()), dim =1)
                    pred_durations_all = torch.cat((pred_durations_all, durations_gt[:, predicted_kp_num+1].unsqueeze(1).cpu()), dim=1)            
                    pred_values_all = torch.cat((pred_values_all, vals_gt[:,predicted_kp_num+1,:].unsqueeze(1)), dim =1)
                else:
                    pred_label_prob_all = torch.cat((pred_label_prob_all, label_prob_pred.unsqueeze(1).cpu()), dim =1)
                    pred_durations_all = torch.cat((pred_durations_all, 
                                        loss_function_manager.decategorize_durations(duration_pred).cpu()), dim=1)       

                    pred_values_all = torch.cat((pred_values_all, val_pred.unsqueeze(1)), dim =1)
                    #check if we should keep looping or not
                    if torch.mean(torch.sum(pred_durations_all[:, input_kp_num:] , dim=1)-offset_n) > 25:
                        past_one_sec=True
                    if not mode=="val" and not mode=="test" and predicted_kp_num+1>=all_kp_len-1 and not interpolate:
                        keep_looping=False
                    if (torch.min(torch.sum(pred_durations_all[:, input_kp_num:] , dim=1)-offset_n) > output_seq_n and predicted_kp_num+1>=all_kp_len-1): 
                        keep_looping=False
                   
                val_pred_prev = pred_values_all[:, min(predicted_kp_num-1, 0),:]
                predicted_kp_num += 1

            bt2= time.time()

            #interpolation
            future_values = pred_values_all[:, input_kp_num-1:, :]
            if interpolate:
                all_seq_reshaped = all_seq_cuda.clone().reshape(batch_size, all_seq.shape[1], num_of_joints, 3)
                gt_p3d_reshaped = all_seq_reshaped[:, input_seq_n:output_seq_n+input_seq_n, :, :]
                kp_pred_p3d = torch.zeros(all_seq_cuda.shape[0], output_seq_n, all_seq_cuda.shape[2]).cuda()
                kp_pred_p3d[:,:,:] = all_seq_cuda[:,  0, :].unsqueeze(1)

                if opt.run_static:
                    final_pred_p3d = gt_p3d_reshaped.clone()
                elif not opt.use_interpolator:
                    linearly_interpolated_seq = linearly_interpolate_values(future_values.cpu(), 
                                                                    pred_durations_all[:,input_kp_num-1:], 
                                                                    output_seq_n, offset_n).cuda()
                    kp_pred_p3d[:,:,dim_used] = linearly_interpolated_seq
                elif opt.use_interpolator: 
                    linearly_interpolated_seq = linearly_interpolate_values(future_values.cpu(), 
                                                                pred_durations_all[:,input_kp_num-1:], 
                                                                output_seq_n, offset_n).cuda()
                    num_run_interp = ceil(output_seq_n/opt.interpolator_seq_len)
                    for interp_ind in range(num_run_interp):
                        tmp_begin = interp_ind*opt.interpolator_seq_len
                        tmp_end = min((interp_ind+1)*opt.interpolator_seq_len,output_seq_n)
                        tmp_dur = tmp_end- tmp_begin
                        to_intrp = torch.zeros([batch_size, opt.interpolator_seq_len, linearly_interpolated_seq.shape[2]]).cuda()
                        to_intrp[:,:tmp_dur] = linearly_interpolated_seq[:,tmp_begin:tmp_end]
                        if tmp_dur < opt.interpolator_seq_len:
                            to_intrp[:,tmp_dur:] =  linearly_interpolated_seq[:,-1].unsqueeze(1)
                        reconst, _ = model.interpolator_model(to_intrp.permute(0,2,1))
                        kp_pred_p3d[:,tmp_begin:tmp_end,dim_used] = reconst.permute(0,2,1)[:,:tmp_dur,:] 
                        
                    
                kp_pred_p3d = data_utils.set_equal_joints(kp_pred_p3d, dataset)
                kp_pred_p3d_reshaped = kp_pred_p3d.reshape(batch_size*output_seq_n, num_of_joints, 3).permute(0,2,1)
                final_pred_p3d = (kp_pred_p3d_reshaped.permute(0,2,1)).reshape(batch_size,output_seq_n, num_of_joints, 3)
            
                loss_function_manager.update_mse_loss(final_pred_p3d, gt_p3d_reshaped, generated_ind)

            bt3= time.time()

            # update the losses used for training
            loss_model = loss_function_manager.sum_losses(batch_size)
            loss = loss_model
            if mode == "train" and not model.is_oracle():
                optimizer.zero_grad()
                optimizer.backprop(loss_model)
                optimizer.step()
            loss_function_manager.update_loss_batch(batch_size*output_kp_num)
                
            #action recognition on predicted motions
            if mode=="test" or mode=="val":
                #(batch_size, seq_len, 22*3)
                tmp = final_pred_p3d.reshape(batch_size, output_seq_n, num_of_joints*3)[:, :, dim_used]
                ar_input = reshape_seq_as_input(tmp)
                iterate_ar_model(model.ar_model_omac, ar_input, action_labels[:,0], omac_ar_loss_function_manager, epoch)
                pred_values_diverse[generated_ind] = final_pred_p3d
                omac_ar_loss_function_manager.compute_ent_metrics(gt_p3d_reshaped, final_pred_p3d)
                loss_function_manager.compute_ent_metrics(gt_p3d_reshaped, final_pred_p3d, diverse_ind=generated_ind)

            bt4= time.time()
        bt5= time.time()
    

        if opt.is_diverse and not model.is_oracle():
            diverse_ind_list = loss_function_manager.save_best_ave_loss()
            omac_ar_loss_function_manager.compute_diversity(pred_values_diverse.detach().cpu().numpy())
        else:
            diverse_ind_list = [[0]*batch_size]

        # POSE VISUALIZATIONS
        if interpolate and update_figures and opt.save_figs and not already_generated:
            already_generated = True
            if i == 0:
                # # ground truth 
                num_vids = 4 if mode=="test" else 0
                # for poses_in_batch in range(min(batch_size,num_vids)):
                #     ind_list = list(range(0,125,1))
                #     for ind in ind_list:
                #         tmp=all_seq_reshaped[:, input_seq_n:input_seq_n+125, :, :]
                #         visualize_pose_gt=tmp[poses_in_batch,ind,:,:].transpose(1,0).detach().cpu().numpy()
                #         vis_client.display_pose(visualize_pose_gt, data_utils.all_bone_connections(dataset), color="xkcd:black", save_loc=fig_location+"/GT", custom_name=str(poses_in_batch)+"_", time=ind)

                # predictions
                if model.is_oracle():
                    for poses_in_batch in range(min(batch_size,num_vids)):
                        for diverse_seq_num_ind, diverse_seq_num in enumerate(diverse_ind_list):
                            ind_list = list(range(0,125,1))
                            for ind in ind_list:
                                visualize_pose=pred_values_diverse[diverse_seq_num[poses_in_batch], poses_in_batch, ind, :, :].transpose(1,0).detach().cpu().numpy()
                                tmp=all_seq_reshaped[:, input_seq_n:input_seq_n+125, :, :]
                                visualize_pose_gt = tmp[poses_in_batch,ind,:,:].transpose(1,0).detach().cpu().numpy()
                                vis_client.display_poses([visualize_pose, visualize_pose_gt], data_utils.all_bone_connections(dataset), save_loc=fig_location+"/Ours", custom_name=str(poses_in_batch)+"_", time=ind)
                    
                else:
                    for poses_in_batch in range(min(batch_size,num_vids)):
                        for diverse_seq_num_ind, diverse_seq_num in enumerate(diverse_ind_list):
                            ind_list = list(range(0,125,1))
                            for ind in ind_list:
                                visualize_pose=pred_values_diverse[diverse_seq_num[poses_in_batch], poses_in_batch, ind, :, :].transpose(1,0).detach().cpu().numpy()
                                vis_client.display_pose(visualize_pose, data_utils.all_bone_connections(dataset), color="xkcd:black", save_loc=fig_location+"/Ours", custom_name=str(poses_in_batch)+"_"+str(diverse_seq_num_ind)+"_", time=ind)
                    
                
        bar.suffix = '{}/{}|batch time {:.3f}s|loss{:.5f}|model:{:.3f}|interp{:.3f}|ar{:.3f}|vis{:.3f}|total time{:.2f}s. Predicted {} kp, mean dur: {:.2f}'.format(
                                                                         i + 1, len(data_loader), time.time() - bt, loss, 
                                                                         bt2-bt0_5, bt3-bt2, bt4-bt3, bt5-bt4,
                                                                         time.time() - st, predicted_kp_num,
                                                                         torch.mean(torch.mean(pred_durations_all, dim=1)).item())
        bar.next()
            

    
    bar.finish() 
    return fig_location


