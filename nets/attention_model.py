import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.neural_module import MultiHeadAttention, PositionWiseFeedForward


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Linear(in_features, out_features, bias=False)
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)

        self.normlizer = nn.BatchNorm1d(out_features, affine=True)

    def forward(self, h):
        """
        in_f = out_f = d_model
        :param h: [bs, gs, in_f]
        :return:
        """
        #[bs, gs, out_feature]
        h_prime = self.W(h)

        if self.concat:
            h_prime = F.elu(h_prime)
            h_prime = self.normlizer(h_prime.view(-1, h_prime.size(-1))).view(*h_prime.size())

        return h_prime


class GraphAttention(nn.Module):

    def __init__(self, in_features, n_hid, dropout, alpha, nheads):
        super(GraphAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        n_hid = n_hid // nheads
        self.attentions = [GraphAttentionLayer(in_features, n_hid, dropout, alpha, concat=True)
                          for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(n_hid * nheads, n_hid * nheads, dropout, alpha, concat=False)
        self.ffd = PositionWiseFeedForward(n_hid * nheads, n_hid * nheads, dropout)
        self.normalizer1 = nn.BatchNorm1d(n_hid * nheads, affine=True)
        self.normalizer2 = nn.BatchNorm1d(n_hid * nheads, affine=True)

    def forward(self, x):
        """

        :param x: [bs, gs, d_model]
        :return: [bs, gs, d_model]
        """
        x = self.dropout(x)
        residual1= x
        x = torch.cat([att(x) for att in self.attentions], dim=-1)
        x = self.dropout(x)
        x = self.out_att(x)
        x = self.normalizer1((F.elu(x) + residual1).view(-1, x.size(-1))).view(*x.size())
        residual2 = x
        x = self.ffd(x) + residual2
        x = self.normalizer2(x.view(-1, x.size(-1))).view(*x.size())
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, d_k, d_v, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout)
        self.pos_ffd = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, enc_input):
        enc_out = self.self_attn(enc_input, enc_input, enc_input)
        enc_out = self.pos_ffd(enc_out)
        return enc_out


class TransformerEncoder(nn.Module):

    def __init__(self, n_layers, n_heads, d_k, d_v, d_model,
                 d_ffd, dropout):
        super(TransformerEncoder, self).__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_ffd = d_ffd
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ffd, d_k, d_v, dropout)
            for _ in range(n_layers)
        ])
        # self.normalizer = nn.BatchNorm1d(d_model, affine=True)

    def forward(self, enc_input):
        '''

        :param enc_input: [bs, gs, d_model]
        :return:
        '''
        enc_out = self.dropout(enc_input)

        for enc_layer in self.layers:
            enc_out = enc_layer(enc_out)

        # [bs, gs, d_model]
        # return self.normalizer(enc_out.view(-1, enc_out.size(-1))).view(*enc_out.size())
        return enc_out


