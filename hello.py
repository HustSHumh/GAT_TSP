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
    a = torch.Tensor(np.arange(24).reshape(2, 3, 4))[:, 0, :]
    print(torch.cuda.is_available())
