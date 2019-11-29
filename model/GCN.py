"""Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/MichSchli/RelationPrediction

Difference compared to MichSchli/RelationPrediction
* report raw metrics instead of filtered metrics

Adopted from DGL, https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn
"""

import torch
import torch.nn as nn

import math

import dgl.function as fn
import torch.nn.functional as F
import torch.nn.init as init

from utils.module import GateNetwork,cummax,ONLSTMCell
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EmbeddingLayer(nn.Module):
    def __init__(self, g,num_nodes, h_dim,pretrained=None):
        super(EmbeddingLayer, self).__init__()
        if pretrained is None:
            self.embedding = torch.nn.Embedding(num_nodes, h_dim,padding_idx=0)
        else:
            self.embedding = torch.nn.Embedding.from_pretrained(pretrained,freeze=True)
        self.g = g

    def forward(self,node_id):
        self.g.ndata['h'] = self.embedding(node_id)
        return self.g.ndata['h']


class OnLSTMEmbeddingLayer(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(OnLSTMEmbeddingLayer, self).__init__()


class GCNLayer(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 out_feats,
                 dropout,
                 bias=True,
                 activation=True):
        super(GCNLayer, self).__init__()
        self.g = g
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = nn.LeakyReLU(0.2)
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        init.xavier_uniform_(self.weight.data,gain=nn.init.calculate_gain('relu'))
        stdv = 1. / math.sqrt(self.weight.size(1))
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # def reset_parameters(self):
    #     stdv = 1. / math.sqrt(self.weight.size(1))
    #     self.weight.data.uniform_(-stdv, stdv)
    #     if self.bias is not None:
    #         self.bias.data.uniform_(-stdv, stdv)

    def forward(self,h):
        if self.dropout:
            h = self.dropout(h)
        m = torch.mm(h,self.weight) * self.g.ndata['norm']
        h = torch.mm(h,self.weight)
        self.g.ndata['m'] = m
        self.g.update_all(fn.copy_src(src='m', out='m'),
                          fn.sum(msg='m', out='n'))
        n = self.g.ndata.pop('n')
        # normalization by square root of dst degree
        n = n * self.g.ndata['norm']
        # bias
        if self.bias is not None:
            n = n + self.bias
        if self.activation:
            n = self.activation(n)
        return n


class GCNGateLayer(GCNLayer):

    def __init__(self, g,in_feats,out_feats,dropout,bias=True,activation=True):
        "docstring"
        super(GCNGateLayer, self).__init__(g,in_feats,out_feats,dropout,bias,activation)

        self.gate = GateNetwork(out_feats)

    def forward(self,h):
        if self.dropout:
            h = self.dropout(h)
        h = torch.mm(h,self.weight)
        m = torch.mm(h,self.weight) * self.g.ndata['norm']
        self.g.ndata['m'] = m
        self.g.update_all(fn.copy_src(src='m',out='m'),fn.sum(msg='m',out='n'))
        n = self.g.ndata.pop('n')
        # x = self.g.ndata.pop('x')
        n = n * self.g.ndata['norm']
        h = self.gate(n,h)

        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)

        return h


class GCNCummaxGateLayer(GCNLayer):

    def __init__(self, g,in_feats,out_feats,dropout,bias=True,activation=True):
        "docstring"
        super(GCNCummaxGateLayer, self).__init__(g,in_feats,out_feats,dropout,bias,activation)

        self.ix = nn.Linear(in_feats,out_feats)
        self.ih = nn.Linear(in_feats,out_feats)
        self.fx = nn.Linear(in_feats,out_feats)
        self.fh = nn.Linear(in_feats,out_feats)

    def forward(self,h):
        if self.dropout:
            h = self.dropout(h)
        x = torch.mm(h,self.weight)
        m = x * self.g.ndata['norm']
        self.g.ndata['m'] = m
        self.g.update_all(fn.copy_src(src='m',out='m'),fn.sum(msg='m',out='n'))
        n = self.g.ndata.pop('n')
        n = n * self.g.ndata['norm']

        forget_gate = cummax(self.fx(n)+self.fh(x),dim=1)
        input_gate = 1 - forget_gate
        # omega = forget_gate * input_gate
        h = forget_gate * x + input_gate * n

        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)

        return h


class GCNCummaxLSTMLayer(GCNLayer):
    def __init__(self, g,in_feats,out_feats,dropout,chunk_size,bias=True,activation=True):
        "docstring"
        super(GCNCummaxLSTMLayer, self).__init__(g,in_feats,out_feats,dropout,bias,activation)

        self.cell = ONLSTMCell(in_feats,out_feats,chunk_size,)

    def forward(self,h):
        if self.dropout:
            h = self.dropout(h)
        h = torch.mm(h,self.weight)
        m = torch.mm(h,self.weight) * self.g.ndata['norm']
        self.g.ndata['m'] = m
        self.g.update_all(fn.copy_src(src='m',out='m'),fn.sum(msg='m',out='n'))
        n = self.g.ndata.pop('n')
        n = n * self.g.ndata['norm']

        # note now, the previous hidden state is x, the input is n
        c = self.g.ndata['c']
        hx = h
        cx = c
        x = n
        (hx,cx) = self.cell(x,(hx,cx))
        self.g.ndata['c'] = cx

        return hx


class GCNCummaxGRULayer(GCNLayer):
    def __init__(self, g,in_feats,out_feats,dropout,bias=True,activation=True):
        "docstring"
        super(GCNCummaxGRULayer, self).__init__(g,in_feats,out_feats,dropout,bias,activation)

        self.input_size = in_feats
        self.hidden_size = out_feats

        self.ix = nn.Linear(in_feats,3*out_feats)
        self.ih = nn.Linear(out_feats,3*out_feats)

        init.xavier_uniform_(self.ix.weight.data)
        init.xavier_uniform_(self.ih.weight.data)

    def forward(self,h):
        if self.dropout:
            h = self.dropout(h)
        h = torch.mm(h,self.weight)
        m = h * self.g.ndata['norm']
        self.g.ndata['m'] = m
        self.g.update_all(fn.copy_src(src='m',out='m'),fn.sum(msg='m',out='n'))
        n = self.g.ndata.pop('n')
        n = n * self.g.ndata['norm']

        # note now, the previous hidden state is x, the input is n
        ri,zi,ci = self.ix(n).chunk(3,1)
        rh,zh,ch = self.ih(h).chunk(3,1)
        r = cummax(ri + rh)
        z = 1 - cummax(zi + zh)
        c = torch.tanh(ci + r * ch)
        h = (1-z) * c + z * h

        return h


class GCNGRULayer(GCNLayer):
    def __init__(self,g,in_feats,out_feats,dropout,bias=True,activation=True):
        super(GCNGRULayer, self).__init__()
        self.gru = nn.GRUCell(in_feats,out_feats)

    def forward(self,h):

        if self.dropout:
            h = self.dropout(h)
        h = torch.mm(h,self.weight)
        m = h * self.g.ndata['norm']
        self.g.ndata['m'] = m
        self.g.update_all(fn.copy_src(src='m',out='m'),fn.sum(msg='m',out='n'))
        n = self.g.ndata.pop('n')
        n = n * self.g.ndata['norm']
        h = self.gru(n,h)

        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)

        return h


class GCNLSTMLayer(GCNLayer):
    def __init__(self, g,in_feats,out_feats,dropout,bias=True,activation=True):
        "docstring"
        super(GCNLSTMLayer, self).__init__(g,in_feats,out_feats,dropout,bias,activation)

        self.ioux = nn.Linear(in_feats,3*out_feats)
        self.iouh = nn.Linear(in_feats,3*out_feats)
        self.fx = nn.Linear(in_feats,out_feats)
        self.fh = nn.Linear(in_feats,out_feats)
        self.reset_parameters()

    def reset_parameters(self):
        init.orthogonal_(self.ioux.weight.data)
        init.constant_(self.ioux.bias.data,val=0)

        init.orthogonal_(self.iouh.weight.data)
        init.constant_(self.iouh.bias.data,val=0)

        init.orthogonal_(self.fx.weight.data)
        init.constant_(self.fx.bias.data,val=1)

        init.orthogonal_(self.fh.weight.data)
        init.constant_(self.fh.bias.data,val=1)

    def forward(self,h):
        if self.dropout:
            h = self.dropout(h)
        h = torch.mm(h,self.weight)
        m = h * self.g.ndata['norm']
        self.g.ndata['m'] = m
        self.g.update_all(fn.copy_src(src='m',out='m'),fn.sum(msg='m',out='n'))
        n = self.g.ndata.pop('n')
        c = self.g.ndata.pop('c')
        n = n * self.g.ndata['norm']

        # note now, the previous hidden state is x, the input is n
        iou = self.ioux(n) + self.iouh(h)
        i,o,u = torch.split(iou,iou.size(1)//3,dim=1)
        i,o,u = torch.sigmoid(i),torch.sigmoid(o),torch.tanh(u)

        f = torch.sigmoid(self.fh(h) + self.fx(n))

        c = i * u + f * c
        self.g.ndata['c'] = c
        h = torch.mul(o,torch.tanh(c))

        return h


class GCN(nn.Module):
    def __init__(self,
                 g,
                 num_nodes,
                 n_hidden,
                 n_classes,
                 n_layers,
                 dropout,
                 combine_fn,
                 chunk_size=None,
                 pretrained=None,):
        super(GCN, self).__init__()
        self.num_nodes = num_nodes
        self.n_hidden = n_hidden
        self.layers = nn.ModuleList()
        self.chunk_size = chunk_size
        self.combine_fn = combine_fn
        # input layer
        # self.layers.append(EmbeddingLayer(g,num_nodes,n_hidden,pretrained))
        self.input_layer = EmbeddingLayer(g,num_nodes,n_hidden,pretrained)
        # hidden layers
        for i in range(n_layers):
            if combine_fn is None:
                self.layers.append(GCNLayer(g, n_hidden, n_hidden, dropout))
            elif combine_fn == 'lstm':
                self.layers.append(GCNLSTMLayer(g,n_hidden,n_hidden,dropout))
            elif combine_fn == 'gate':
                self.layers.append(GCNGateLayer(g,n_hidden,n_hidden,dropout))
            elif combine_fn == 'cummaxgate':
                self.layers.append(GCNCummaxGateLayer(g,n_hidden,n_hidden,dropout))
            elif combine_fn == 'cummax_lstm':
                self.layers.append(GCNCummaxLSTMLayer(g,n_hidden,n_hidden,dropout,chunk_size))
            elif combine_fn == 'cummax_gru':
                self.layers.append(GCNCummaxGRULayer(g,n_hidden,n_hidden,dropout,chunk_size))
        self.g = g

    def forward(self,h=None):
        if h is None:
            id = self.g.ndata['id'].squeeze()
            h = self.input_layer(id)
        if self.combine_fn == 'cummax_lstm':
            self.n_chunk = int(self.n_hidden // self.chunk_size)
            self.g.ndata['c'] = torch.zeros((self.num_nodes,self.n_chunk,self.chunk_size)).to(device)
        for layer in self.layers:
            h = layer.forward(h)
        return h
