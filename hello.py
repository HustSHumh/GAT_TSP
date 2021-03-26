import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


if __name__ == '__main__':
    a = torch.randn(10, 20, 2)
    x = a[:, :, 0]
    y = a[:, :, 1]
    print(F.pairwise_distance(x, y))



