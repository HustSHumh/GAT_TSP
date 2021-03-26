import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.attention_model import EncodeLayer


class Encoder(nn.Module):
    def __init__(self, node_dim, n_layers, n_heads, d_model, d_ffd, dropout=0.0):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList([
            EncodeLayer(n_heads, d_model, d_ffd, dropout)
            for _ in range(n_layers)
        ])
        self.init_embed = nn.Linear(node_dim, d_model, bias=False)
        self.normalizer = nn.BatchNorm1d(d_model, affine=True)

        nn.init.xavier_normal_(self.init_embed.weight)


    def forward(self, x):
        x = self.init_embed(x)
        for i in range(self.n_layers):
            x = self.layers[i](x)
            x = self.normalizer(x.view(-1, x.size(-1))).view(*x.size())

        return x


if __name__ == '__main__':
    x = torch.randn(1024, 20, 2)
    model = Encoder(2, 3, 8, 128, 512)
    print(model(x).size())
