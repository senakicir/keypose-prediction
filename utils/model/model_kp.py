#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
from torch.nn import GRUCell, GRU, LSTMCell
import torch
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import math

from .utils import weights_init, LayerNormGRUCell

class KP_Oracle(nn.Module):
    
    def __init__(self, fixed_duration=False):
        super(KP_Oracle, self).__init__()
        self.dumm_cell = LayerNormGRUCell(input_size=1, hidden_size=1)

    def forward(self, kp_inputs, hidden_states, kp_outputs):
        #returns distance labels and durations
        kp_val, prob, duration = kp_outputs
        return kp_val, prob, duration, None

    def is_oracle(self):
        return True

    def initHidden(self, is_cuda, batch_size):
        return None


class KP_GRU_Prob(nn.Module):
    
    def __init__(self, cluster_n, hidden_size, dur_cat_dim):
        ##input size will be n_clusters
        super(KP_GRU_Prob, self).__init__()

        self.hidden_size = hidden_size
        self.cluster_n = cluster_n
        dur_dim = dur_cat_dim 

        self.cell_cc = LayerNormGRUCell(input_size=self.cluster_n, hidden_size=hidden_size)
        self.cell_concat = LayerNormGRUCell(input_size=hidden_size+dur_dim, hidden_size=hidden_size)
        self.cell_concat2 = LayerNormGRUCell(input_size=hidden_size, hidden_size=hidden_size)
        self.linear1 = nn.Linear(hidden_size, self.cluster_n)
        self.linear2 = nn.Linear(hidden_size, dur_dim)
        
        self.relu = nn.ReLU()
       # weights_init(self)

    def forward(self, kp_input, hidden_states, kp_outputs=None):
        _, kp_label, _, duration = kp_input
        new_hidden_states = []

        output_cc = self.cell_cc(kp_label, hidden_states[0])        
        # output_cc = self.cell_cc(val, hidden_states[0])
        new_hidden_states.append(output_cc)

        input_concat = torch.cat([output_cc, duration], dim=1)
        concat_gru_output = self.cell_concat(input_concat, hidden_states[1])
        new_hidden_states.append(concat_gru_output)

        concat_gru_output2 = self.cell_concat2(concat_gru_output, hidden_states[2])
        new_hidden_states.append(concat_gru_output2)

        output_sum = concat_gru_output + concat_gru_output2
        prob = self.linear1(output_sum)
        duration = self.linear2(output_sum)

        return None, prob, duration, new_hidden_states

    def is_oracle(self):
        return False

    def initHidden(self, is_cuda, batch_size):
        self.hidden_states = []
        self.hidden_states.append(Variable(torch.zeros((batch_size, self.hidden_size)).cuda()).float())
        self.hidden_states.append(Variable(torch.zeros((batch_size, self.hidden_size)).cuda()).float())
        self.hidden_states.append(Variable(torch.zeros((batch_size,  self.hidden_size)).cuda()).float())
        return self.hidden_states


class KP_LSTM_Prob(nn.Module):
    
    def __init__(self, cluster_n, hidden_size, dur_cat_dim):
        ##input size will be n_clusters
        super(KP_LSTM_Prob, self).__init__()

        self.hidden_size = hidden_size
        self.cluster_n = cluster_n
        dur_dim = dur_cat_dim

        self.cell_cc = LayerNormGRUCell(input_size=cluster_n, hidden_size=hidden_size)
        self.cell_concat = LayerNormGRUCell(input_size=hidden_size+dur_dim, hidden_size=hidden_size)
        self.cell_concat2 = LayerNormGRUCell(input_size=hidden_size, hidden_size=hidden_size)
        self.dropout = nn.Dropout(p=0.5)

        self.linear1 = nn.Linear(hidden_size, cluster_n)
        self.linear2 = nn.Linear(hidden_size, dur_dim)
        self.relu = nn.ReLU()

    def forward(self, kp_input, state_tuple, kp_outputs=None):
        _, kp_label, _, duration = kp_input
        hidden_states, cell_states = state_tuple
        new_hidden_states = []
        new_cell_states = []

        h_1, c_1 = self.cell_cc(kp_label, (hidden_states[0], cell_states[0]))
        new_hidden_states.append(h_1)
        new_cell_states.append(c_1)

        input_concat = torch.cat([h_1, duration], dim=1)
        h_2, c_2 = self.cell_concat(input_concat, (hidden_states[1], cell_states[1]))
        new_hidden_states.append(h_2)
        new_cell_states.append(c_2)

        h_3, c_3 = self.cell_concat2(h_2, (hidden_states[2], cell_states[2]))
        new_hidden_states.append(h_3)
        new_cell_states.append(c_3)

        output_sum = h_3
        prob = self.linear1(output_sum)
        duration = self.linear2(output_sum)

        if not self.dur_categorical:
            duration = self.relu(duration)

        return None, prob, duration, (new_hidden_states, new_cell_states)

    def is_oracle(self):
        return False

    def initHidden(self, is_cuda, batch_size):
        self.hidden_states = []
        self.cell_states = []
        for i in range(3):
            hidden_state = torch.zeros([batch_size, self.hidden_size]).cuda().float()
            cell_state = torch.zeros([batch_size, self.hidden_size]).cuda().float()
            # torch.nn.init.xavier_normal_(hidden_state)
            # torch.nn.init.xavier_normal_(cell_state)
            self.hidden_states.append(hidden_state)
            self.cell_states.append(cell_state)
        return (self.hidden_states, self.cell_states)



class KP_GRU_Prob_v2(nn.Module):
    
    def __init__(self, cluster_n, hidden_size, dur_cat_dim):
        ##input size will be n_clusters
        super(KP_GRU_Prob_v2, self).__init__()

        self.hidden_size = hidden_size
        self.cluster_n = cluster_n
        dur_dim = dur_cat_dim

        self.cell_cc = LayerNormGRUCell(input_size=cluster_n, hidden_size=hidden_size)
        self.cell_concat = LayerNormGRUCell(input_size=hidden_size+dur_dim, hidden_size=hidden_size)
        self.cell_concat2 = LayerNormGRUCell(input_size=hidden_size, hidden_size=hidden_size)
        self.cell_concat3 = LayerNormGRUCell(input_size=hidden_size, hidden_size=hidden_size)

        self.linear1 = nn.Linear(hidden_size, cluster_n)
        self.linear2 = nn.Linear(hidden_size, dur_dim)
        
        self.relu = nn.ReLU()
       # weights_init(self)


    def forward(self, kp_input, hidden_states, kp_outputs=None):
        _, kp_label, _, duration = kp_input
        new_hidden_states = []

        output_cc = self.cell_cc(kp_label, hidden_states[0])
        new_hidden_states.append(output_cc)

        input_concat = torch.cat([output_cc, duration], dim=1)
        concat_gru_output = self.cell_concat(input_concat, hidden_states[1])
        new_hidden_states.append(concat_gru_output)

        concat_gru_output2 = self.cell_concat2(concat_gru_output, hidden_states[2])
        new_hidden_states.append(concat_gru_output2)

        concat_gru_output3 = self.cell_concat3(concat_gru_output2, hidden_states[3])
        new_hidden_states.append(concat_gru_output3)

        output_sum = concat_gru_output + concat_gru_output2 + concat_gru_output3
        prob = self.linear1(output_sum)
        duration = self.linear2(output_sum)

        return None, prob, duration, new_hidden_states

    def is_oracle(self):
        return False

    def initHidden(self, is_cuda, batch_size):
        self.hidden_states = []
        self.hidden_states.append(Variable(torch.zeros((batch_size, self.hidden_size)).cuda()).float())
        self.hidden_states.append(Variable(torch.zeros((batch_size, self.hidden_size)).cuda()).float())
        self.hidden_states.append(Variable(torch.zeros((batch_size,  self.hidden_size)).cuda()).float())
        self.hidden_states.append(Variable(torch.zeros((batch_size,  self.hidden_size)).cuda()).float())
        return self.hidden_states




class KP_GRU_Prob_v3(nn.Module):
    
    def __init__(self, cluster_n, hidden_size, dur_cat_dim):
        ##input size will be n_clusters
        super(KP_GRU_Prob_v3, self).__init__()

        self.hidden_size = hidden_size
        self.cluster_n = cluster_n
        dur_dim = dur_cat_dim 

        self.cell_cc = LayerNormGRUCell(input_size=cluster_n, hidden_size=hidden_size)
        self.cell_concat = LayerNormGRUCell(input_size=hidden_size+dur_dim, hidden_size=hidden_size)

        self.linear1 = nn.Linear(hidden_size, cluster_n)
        self.linear2 = nn.Linear(hidden_size, dur_dim)
        
        self.relu = nn.ReLU()
       # weights_init(self)


    def forward(self, kp_input, hidden_states, kp_outputs=None):
        _, kp_label, _, duration = kp_input
        new_hidden_states = []

        output_cc = self.cell_cc(kp_label, hidden_states[0])
        new_hidden_states.append(output_cc)

        input_concat = torch.cat([output_cc, duration], dim=1)
        concat_gru_output = self.cell_concat(input_concat, hidden_states[1])
        new_hidden_states.append(concat_gru_output)

        output_sum = concat_gru_output
        prob = self.linear1(output_sum)
        duration = self.linear2(output_sum)

        return None, prob, duration, new_hidden_states

    def is_oracle(self):
        return False

    def initHidden(self, is_cuda, batch_size):
        self.hidden_states = []
        self.hidden_states.append(Variable(torch.zeros((batch_size, self.hidden_size)).cuda()).float())
        self.hidden_states.append(Variable(torch.zeros((batch_size, self.hidden_size)).cuda()).float())
        return self.hidden_states

