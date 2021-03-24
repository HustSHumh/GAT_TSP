import warnings

import torch
import numpy as np
import os
import json
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
import torch.nn.functional as F

def torch_load_cpu(load_path):
    return torch.load(load_path, map_location=lambda storage, loc: storage)

def move_to(var, device):
    if isinstance(var, dict):
        return {k:move_to(v, device) for k, v in var.items()}
    return var.to(device)


