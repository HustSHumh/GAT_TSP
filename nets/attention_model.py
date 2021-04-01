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
        self.a = nn.Linear(2 * out_features, 1)
        nn.init.xavier_uniform_(self.a.weight, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.normlizer = nn.BatchNorm1d(out_features, affine=True)

    def forward(self, h):
        """
        in_f = out_f = d_model
        :param h: [bs, gs, in_f]
        :return:
        """
        batch_size, graph_size = h.size(0), h.size(1)
        diag_ = torch.diag(torch.ones(graph_size-1), 1)
        # 对角线是0  其他非零
        # adj = torch.ones(size=(graph_size, graph_size)) - torch.eye(graph_size)[None, :, :].expand(batch_size, -1, -1)
        adj = (torch.eye(graph_size) + diag_ + diag_.T)[None, :, :].expand(batch_size, -1, -1)
        #[bs, gs, out_feature]
        Wh = self.W(h)
        # [bs, gs, gs, 2*out_f]
        a_input = self._prepare_attentional_mechanism_input(Wh)
        # [bs, gs, gs]
        e = self.leakyrelu(self.a(a_input).squeeze(-1))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where((adj > 0).to(e.device), e, zero_vec)

        # [bs, gs, gs]
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        # [bs, gs, out_f]
        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            h_prime = F.elu(h_prime)
            h_prime = self.normlizer(h_prime.view(-1, h_prime.size(-1))).view(*h_prime.size())

        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        """

        :param Wh: (bs, gs, out_f)
        :return:
        """

        batch_size, N = Wh.size(0), Wh.size(1)
        # [bs, N * N, out_f]
        Wh_repearted_in_chunks = Wh.repeat_interleave(N, dim=1)
        # [bs, N * N, out_f]
        Wh_repearted_alternating = Wh.repeat(1, N, 1)

        # [bs, N * N, 2 * out_f]
        all_combinations_matrix = torch.cat([Wh_repearted_in_chunks, Wh_repearted_alternating], dim=-1)

        # [bs, N, N, 2 * out_f]
        return all_combinations_matrix.view(batch_size, N, N, 2 * self.out_features)


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


