import torch
import numpy as np
from .metrics import rmse, nrmse, align_mae
from .baselines import detect_steps

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
    k_list = detect_steps(I_np, deltaI=cfg_eval["eval"]["vdrop_deltaI"])