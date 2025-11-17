import numpy as np
import torch

def rmse(a, b):
    return float(torch.sqrt(torch.mean((a-b)**2)).item())

def nrmse(a, b):
    r = a.max() - a.min()
    return 100*rmse(a, b)/float(r.item()+1e-12)

def align_mae():
    pass