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
from .utils_ar import reshape_seq_as_input


def iterate_ar_model(model, sequence_input, labels, loss_function_manager, epoch):
    batch_size = sequence_input.shape[0]

    #pass input to model 
    predictions = model(sequence_input)

    #compute loss
    loss = loss_function_manager.update_loss(predictions, labels, batch_size)

    # update the confusion matrix and accuracy of this batch
    loss_function_manager.update_conf_mat_and_acc(predictions, labels, batch_size)

    return loss



def run_action_recognition(mode, data_loader, model, loss_function_manager, epoch, save_loc, writer, rng, optimizer=False, is_cuda=True, dim_used=[]):
    update_figures_epoch = 1
    
    save_figs_flag_poses = ((epoch+1)%update_figures_epoch==0) 

    if save_figs_flag_poses:
        fig_location = save_loc+'/figs/'+str(epoch+1)+"/"+mode
        if not os.path.exists(fig_location):
            os.makedirs(fig_location)

    if mode=="train":
        model.train()
    else: #val or test
        model.eval()
    loss_function_manager.reset_all() 
    
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(data_loader))

    for i, (sequences, labels) in enumerate(data_loader):
        bt = time.time()
        batch_size = sequences.shape[0]

        if is_cuda:
            assert sequences.ndim==3 # shape(batch_size, seq_len, joints*3)
            sequences = Variable(sequences.cuda(non_blocking=True)).float()
            labels = Variable(labels.cuda(non_blocking=True)).long()

        ## convert to format N,C,T,V,M (batch_size, channels, seq_len, joints, num_people)
        sequence_input = reshape_seq_as_input(sequences[:, :, dim_used])
        if mode == "train":
            sequence_input = sequence_input + rng.generate_body_noise(sequence_input.shape)
        assert list(sequence_input.shape) == [sequences.shape[0], 3, sequences.shape[1], len(dim_used)//3, 1]

        loss = iterate_ar_model(model, sequence_input, labels, loss_function_manager, epoch)

        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bar.suffix = '{}/{}|batch time {:.3f}s|total time{:.2f}s.'.format(i + 1, len(data_loader), 
                                                                            time.time() - bt,time.time() - st)
        bar.next()

    bar.finish() 
    return fig_location


