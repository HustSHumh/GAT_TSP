import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from nets.neural_module import MultiHeadAttention, PositionWiseFeedForward

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        """
        :param q: (bs, gs, lq, dk)
        :param k:
        :param v:
        :param mask:
        :return:
        """
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask==0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v)

        return out, attn


class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, d_model, d_k, d_v, dropout):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = nn.Dropout(dropout)

        self.w_qs = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.attn = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.normalizer = nn.BatchNorm1d(d_model, affine=True)

    def forward(self, q, k, v, mask=None):
        """

        :param q: (bs, gs, dmodel)
        :param k:
        :param v:
        :param mask:
        :return:
        """
        k = q if k is None else k
        v = k if v is None else v

        d_k, d_v, n_heads = self.d_k, self.d_v, self.n_heads
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        Q = self.w_qs(q).view(batch_size, len_q, n_heads, d_v)
        K = self.w_ks(k).view(batch_size, len_k, n_heads, d_v)
        V = self.w_vs(v).view(batch_size, len_v, n_heads, d_v)

        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        Q, attn = self.attn(Q, K, V, mask=None)
        Q = Q.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        Q = self.dropout(self.fc(Q))

        Q += residual

        out = self.normalizer(Q.view(-1, Q.size(-1))).view(*Q.size())

        return out


class PositionWiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ffd, dropout):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ffd)
        self.fc2 = nn.Linear(d_ffd, d_model)

        self.normalizer = nn.BatchNorm1d(d_model, affine=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.fc2(F.relu(self.fc1(x)))
        x = self.dropout(x)
        x += residual
        out = self.normalizer(x.view(-1, x.size(-1))).view(*x.size())

        return out


class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, d_ffd, dropout):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout)
        self.pos_ffd = PositionWiseFeedForward(d_model, d_ffd, dropout)

    def forward(self, enc_input):
        enc_out = self.attn(enc_input, enc_input, enc_input)
        enc_out = self.pos_ffd(enc_out)

        return enc_out


class Encoder(nn.Module):

    def __init__(self, n_layers, n_heads, d_model, d_ffd, dropout):
        super(Encoder, self).__init__()
        d_k = d_v = d_model // n_heads
        self.n_layers = n_layers
        self.layers = nn.ModuleList([
            EncoderLayer(n_heads, d_model, d_k, d_v, d_ffd, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, enc_input):
        for i in range(self.n_layers):
            enc_out = self.layers[i](enc_input)
            enc_input = enc_out
        return enc_input
