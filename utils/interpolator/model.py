import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import math
from typing import List, Callable, Union, Any, TypeVar, Tuple
# from torch import tensor as Tensor



class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48, node_n_2=None):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if node_n_2 is None:
            node_n_2 = node_n
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n_2, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class ConvBlock(nn.Module):
    def __init__(self, node_n, in_features, p_dropout, bias=True):
        """
        Define a residual block of GCN
        """
        super(ConvBlock, self).__init__()
        # self.maxpool = nn.AdaptiveMaxPool2d((1, 20))
        self.cnn1 = nn.Conv1d(node_n, in_features, kernel_size=(5), padding=(2), bias=bias)
        self.bn1 = nn.BatchNorm1d(in_features)
        self.relu = nn.ReLU()
        self.cnn2 = nn.Conv1d(in_features, node_n, kernel_size=(5), padding=(2), bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        ## B, 66, feat
        B, n, feat = x.shape
        x = self.cnn1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class KP_Interpolator(nn.Module):
    def __init__(self, input_feature, hidden_feature, output_feature, p_dropout, num_stage=1, node_n=48, pred_duration=False, use_cnn=False):
        """
        Adapted from Wei Mao's LTD code: https://github.com/wei-mao-2019/LearnTrajDep/blob/master/utils/model.py
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(KP_Interpolator, self).__init__()
        self.num_stage = num_stage

        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)
        self.gc7 = GraphConvolution(hidden_feature, output_feature, node_n=node_n, node_n_2=node_n)

        self.use_cnn = use_cnn
        if use_cnn:
            self.convbs = []
            for i in range(num_stage):
                self.convbs.append(ConvBlock(node_n, hidden_feature, p_dropout=p_dropout))
            self.convbs = nn.ModuleList(self.convbs)


        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()
        self.pred_duration = pred_duration
        if self.pred_duration:
            self.fc_label = nn.Linear(hidden_feature*node_n, 3)



    def forward(self, x):
        #x is         #B, 66, 125
        #positional is B, 1 ,125
        b, n, t = x.shape

        y = self.gc1(x)
        b, j, t = y.shape

        y = self.bn1(y.view(b, -1)).view(b, j, t)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)
            if self.use_cnn:
                y = self.convbs[i](y)
            if self.pred_duration:
                if i==2:
                    y_label = self.fc_label(y.view(b, -1))
        y = self.gc7(y)
        # y = y[:,:n-1,:]
        if not self.pred_duration:
            y_label = None

        return y, y_label

    def load_weights(self, run_info, opt):
        model_pth = opt.interpolator_model_path + "/model_interpolate_"+opt.dataset+"_last.pth.tar"
        print(">>> loading ckpt len from '{}'".format(model_pth))
        dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        ckpt = torch.load(model_pth, map_location=dev)
        start_epoch = ckpt['epoch']
        err_best = ckpt['err']
        self.load_state_dict(ckpt['state_dict'])
        print(">>> ckpt len loaded (epoch: {} | err: {})".format(start_epoch, err_best))


