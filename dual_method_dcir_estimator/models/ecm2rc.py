import torch

def f_state_2rc(x, I, R1, C1, R2, C2, Q=3600.0, eta=1.0):
    """
    x = [v_c1, v_c2, soc]
    dx/dt for 2RC ECM
    """
    vc1, vc2, soc = x[..., 0], x[..., 1], x[..., 2]
    dvc1 = -vc1/(R1*C1) + I/C1
    dvc2 = -vc2/(R2*C2) + I/C2
    dsoc = -(eta/Q)*I
    return torch.stack([dvc1, dvc2, dsoc])