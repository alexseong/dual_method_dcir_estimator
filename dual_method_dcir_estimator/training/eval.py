import torch
import numpy as np
from .metrics import rmse, nrmse, align_mae
from .baselines import detect_steps, vdrop_dcirs

def evaluate_window(model, batch, cfg_eval):
    model.eval()
    V =  batch["V" ].to(model.device)
    I =  batch["I" ].to(model.device)
    Tz = batch["Tz"].to(model.device)
    t =  batch["t" ].to(model.device)

    with torch.no_grad():
        out = model(V, I, Tz)

    rmse_v = rmse(out["V_pred"], V)
    nrmse_v = nrmse(out["V_pred"], V) # Normalized RMSE

    # baseline DCIR vs R0 at pulse onsets
    I_np = I[0].cpu().numpy()
    V_np = V[0].cpu().numpy()
    k_list = detect_steps(I_np, deltaI=cfg_eval["eval"]["vdrop_deltaI"])  ## can't understand
    Rdrop_fast = vdrop_dcirs(V_np, I_np, k_list, tau=cfg_eval["vdrop_fast_tau"])
    Rdrop_slow = vdrop_dcirs(V_np, I_np, k_list, tau=cfg_eval["vdrop_slow_tau"])

    # Map to dict for alignment
    R0_map = {k: float(out["params"]["R0"][0, k].cpu().item()) for k, _ in Rdrop_fast}
    Rdrop_fast_map = {k: float(R) for k, R in Rdrop_fast}
    Rdrop_slow_map = {k: float(R) for k, R in Rdrop_slow}

    align_fast = align_mae(R0_map, Rdrop_fast_map)
    align_slow = align_mae(R0_map, Rdrop_slow_map)

    return dict(rmse=rmse_v, nrmse=nrmse_v, align_fast=align_fast, align_slow=align_slow, out=out)