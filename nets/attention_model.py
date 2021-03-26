import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.neural_module import MultiHeadAttention, PositionWiseFeedForward


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, d_k, d_v, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout)
        self.pos_ffd = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, enc_input):
        enc_out = self.self_attn(enc_input, enc_input, enc_input)
        enc_out = self.pos_ffd(enc_out)
        return enc_out


class LightConvLayer(nn.Module):

    def __init__(
            self,
            d_model,
            kenel_size=3,
            padding=1,
            weight_softmax=False,
            n_heads=1,
            weight_dropout=0.0
    ):
        super(LightConvLayer, self).__init__()

        self.kenel_size = kenel_size
        self.padding = padding
        self.n_heads = n_heads
        self.weight_softmax = weight_softmax
        self.weight_dropout_module = nn.Dropout(weight_dropout)

        self.weight = nn.Parameter(torch.Tensor(n_heads, 1, kenel_size))

        self.fc = nn.Linear(d_model, d_model, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        '''

        :param x: [bs, gs, d_model]
        :return:
        '''
        n_heads = self.n_heads

        batch_size, graph_size, d_model = x.size()


        weight = self.weight
        if self.weight_softmax:
            weight = F.softmax(weight, dim=-1)
        x = x.view(-1, n_heads, d_model)

        out = F.conv1d(x, weight, padding=self.padding, groups=self.n_heads)
        out = out.view(batch_size, graph_size, d_model)
        out = self.fc(out)

        return out

class EncodeLayer(nn.Module):
    '''
    input: [bs, gs, d_model]
    out: [bs, gs, d_model]
    '''
    def __init__(self, n_heads, d_model, d_ffd, dropout=0.0):
        super(EncodeLayer, self).__init__()

        n_heads = n_heads // 2

        d_k = d_v = n_heads // 2

        self.attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout)
        self.conv = LightConvLayer(d_model, n_heads=n_heads)
        self.ffd = PositionWiseFeedForward(d_model, d_ffd, dropout)
        self.fc1 = nn.Linear(d_model, d_model, bias=False)
        self.fc2 = nn.Linear(d_model, d_model, bias=False)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        """

        :param x: [bs, gs, d_model]
        :return:
        """
        residual = x
        attn_out = self.attn(x, x, x)
        conv_out = self.conv(x)
        out = self.fc1(conv_out) + self.fc2(attn_out)

        return self.ffd(out + residual)



if __name__ == '__main__':
    a = EncodeLayer(8, 128, 128)
    b = torch.randn(1024, 20, 128)
    print(a(b).size())

