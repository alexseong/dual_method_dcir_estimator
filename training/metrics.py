import numpy as np
import torch

def rmse(a, b):
    return float(torch.sqrt(torch.mean((a-b)**2)).item())

def nrmse(a, b):               # Normalized RMSE
    r = a.max() - a.min()
    return 100*rmse(a, b)/float(r.item()+1e-12)

def align_mae(R0_list, Rdrop_list):
    # match by time index if provided as dicts {k: value}
    common = R0_list.keys()