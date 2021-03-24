import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import NamedTuple


class NT(NamedTuple):
    a : torch.Tensor
    b : torch.Tensor

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return NT(
                a = self.a[key],
                b = self.b[key]
            )
        return tuple.__getitem__(self, key)


if __name__ == '__main__':
    batch_size = 20
    graph_size = 20
    adj = torch.ones(size=(graph_size, graph_size)) - torch.eye(graph_size)[None, :, :].expand(batch_size, -1, -1)
