import torch

def rk4_step(x, I, dt, f, *f_args):
    k1 = f(x, I, *f_args)
    