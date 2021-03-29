import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.attention_model import TransformerEncoder, GraphAttention


class Encoder(nn.Module):

    def __init__(self, trans_layers, n_heads, d_model, d_ffd, alpha, dropout):
        super(Encoder, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        # self.embedder = nn.Linear(2, self.d_model)
        self.transformer_encoder = TransformerEncoder(trans_layers, n_heads, self.d_k, self.d_v,
                                                      d_model, d_ffd, dropout)
        self.graph_attn = GraphAttention(2, d_model, dropout, alpha, n_heads)

    def forward(self, input_node):
        '''

        :param input_node: [bs, gs, 2]
        :return: [bs, gs, d_model]
        '''

        # [bs, gs, 2]
        enc_input = self.graph_attn(input_node)
        # [bs, gs, d_mdoel]
        trans_out = self.transformer_encoder(enc_input)
        # return trans_out

        return trans_out






if __name__ == '__main__':
    x = torch.Tensor(np.arange(12).reshape(1, 6, 2))
    embedder = nn.Linear(2, 128)
    x = embedder(x)
    model = GraphAttention(128, 128, 0.6, 0.2, 8)
    print(model(x))
