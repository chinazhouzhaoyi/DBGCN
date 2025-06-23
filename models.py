"""
paper: DBGCN: Dual-branch Graph Convolutional Network for organ instance inference on sparsely labeled 3D plant data
file:  models_2layers.py
about: Two layers deep DBGCN network
author: Zhaoyi Zhou
date: 2025-6-19
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import math

class GraphConvolution(Module):

    def __init__(self, in_features, out_features, residual=True, variant=True):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features #variant=False

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        """init parameters"""
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1-alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        output = theta * torch.mm(support, self.weight) + (1-theta)*r
        if self.residual:
            output = output + input
        return output



class edge_convAgg(nn.Module):
    '''
       dynamic Conv Module
    '''
    def __init__(self, out_features,K=20):

        super(edge_convAgg, self).__init__()
        self.K = K
        self.out_features = out_features
        self.lin = nn.Linear(2 * self.out_features, self.out_features)
        self.lin.reset_parameters()

    def createSingleKNNGraph(self, X):
        '''
        generate a KNN graph for a single point cloud
        :param X:  X is a Tensor, shape: [N, F]
        :return: KNN graph, shape: [N, K, F]
        '''
        N, Fea = X.shape

        dist_mat = torch.pow(X, 2).sum(dim=1, keepdim=True).expand(N, N) + \
                   torch.pow(X, 2).sum(dim=1, keepdim=True).expand(N, N).t()
        dist_mat.addmm_(1, -2, X, X.t())

        dist_mat_sorted, sorted_indices = torch.sort(dist_mat, dim=1)
        knn_indexes = sorted_indices[:, 1:self.K+1]
        knn_graph = X[knn_indexes]

        return knn_graph,knn_indexes

    def forward(self, X):
        '''
        :param X:  shape: [B, N, F]
        :return:  shape: [B, N, F]
        '''
        B, N, Fea = X.shape
        device = torch.device('cuda')
        KNN_Graph = torch.zeros(B, N, self.K, self.out_features).to(device)

        # creating knn graph
        # X: [B, N, F]
        # knn_idx=[]
        # for idx, x in enumerate(X):
        #     KNN_Graph[idx],knn_indexes = self.createSingleKNNGraph(x)
        #     knn_idx.append(knn_indexes)
        # X: [B, N, F]
        x1 = X.reshape([B, N, 1, Fea])
        x1 = x1.expand(B, N, self.K, Fea)  # x1: [B, N, K, F]
        x2 = KNN_Graph - x1 # x2: [B, N, K, F]
        x2 = torch.mean(x2, dim=(-2,), keepdim=False) # x2: [B, N, 1, F]
        x2=x2.squeeze() # [B, N, F]
        X=X.squeeze()
        x_in = torch.cat([X, x2], dim=1)
        x_out = self.lin(x_in)
        return x_out


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True) #mean.shape:(N, 1)
        var = x.var(-1, unbiased=False, keepdim=True)# var.shape:(N, 1)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

class DBGCN(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, knum, lamda, alpha, variant):# variant=false
        super(DBGCN, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant))
        self.params1 = list(self.convs.parameters())
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.knum = knum
        self.layerNorm1 = LayerNorm(nhidden)
        self.layerNorm2 = LayerNorm(nhidden)

        self.dynamic_conv = nn.ModuleList()
        for _ in range(nlayers):
            self.dynamic_conv.append(edge_convAgg(nhidden, K=knum))
        self.params_dynamic = list(self.dynamic_conv.parameters())

        self.layernorm = nn.ModuleList()
        self.layernorm.append(self.layerNorm1)
        self.layernorm.append(self.layerNorm2)
        self.params_norm = list(self.layernorm.parameters())

        self.lin = nn.Linear(nhidden, nclass)
        self.last_fc = nn.ModuleList()
        self.last_fc.append(self.lin)
        self.params_last_fc = self.last_fc.parameters()

    def reset_parameters(self):
        self.dynamic_conv.reset_parameters()
        self.fcs.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, adj):
        _layers = []
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
            _layers.append((layer_inner))

        x = self.act_fn(self.fcs[0](x))
        dylayer = x.unsqueeze(0)
        for i, dycon in enumerate(self.dynamic_conv):
            if i == 0:
                dylayer = F.dropout(F.relu(dycon(dylayer)), self.dropout, training=self.training)
            else:
                dylayer = dylayer.unsqueeze(0)
                dylayer = F.dropout(F.relu(dycon(dylayer)), self.dropout, training=self.training) + 0.1 * x
        x_dg = dylayer
        cross_out=self.layerNorm1(layer_inner) + self.layerNorm2(x_dg)
        out = self.lin(cross_out)
        return F.log_softmax(out, dim=1), cross_out



