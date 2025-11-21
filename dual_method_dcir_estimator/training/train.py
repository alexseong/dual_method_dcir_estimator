import os
import yaml
import math
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from ..data.loaders import dataframe_from_input
from ..data.preprocess import preprocess, PreprocConfig, Standardizer
from ..data.segmenters import sliding_windows
from ..models.full_model import DCIRNeuralODE
from ..loss.losses import VoltageLoss
from .utils import set_seed

class WindowsDataset(Dataset):
    def __init__(self, df, H, stride, min_dyn_frac, dt, temp_std):
        self.df = df
        self.H = H
        self.dt = dt
        self.temp_std = temp_std
        self.windows = sliding_windows(df, H, stride, min_dyn_frac)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        s, e = self.windows[idx]
        seg = self.df.iloc[s:e]
        V = torch.from_numpy(seg["V"].values).float()
        I = torch.from_numpy(seg["I"].values).float()
        Tz = torch.from_numpy(seg["Tz"].values).float()
        t = torch.from_numpy(seg["t"].values).float()
        return dict(V=V, I=I, Tz=Tz, t=t)

def collate_pad(batch):
    # fixed windows, stack
    V = torch.stack([b["V"] for b in batch], 0)
    I = torch.stack([b["I"] for b in batch], 0)
    Tz = torch.stack([b["Tz"] for b in batch], 0)
    t = torch.stack([b["t"] for b in batch], 0)
    return dict(V=V, I=I, Tz=Tz, t=t)

def cosine_lr(optimizer, base_lr, min_lr, step, total_steps):
    lr = min_lr + 0.5*(base_lr-min_lr)*(1 + math.cos(math.pi*step/total_steps))
    for pg in optimizer.param_groups:
        pg["lr"] = lr

def train_loop(config_path, data_path, run_dir):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    os.makedirs(run_dir, exist_ok=True)

    # Load data
    raw = dataframe_from_input(data_path)
    pre_cfg = PreprocConfig(
        dt=cfg["data"]["dt"], #1.0
        current_sign=cfg["data"]["current_sign"], # "discharge_positive"  # enforce discharge-positive rule
        temp_standardize=cfg["data"]["temp_standardize"] # true
    )

    df, temp_std = preprocess( raw, pre_cfg )

    # Split naive (example): 60/20/20 by contiguous thirds
    n = len(df)
    df_train = df.iloc[:int(0.6*n)].reset_index(drop=True)
    df_val = df.iloc[int(0.6*n):int(0.8*n)].reset_index(drop=True)
    df_test = df.iloc[int(0.8*n):].reset_index(drop=True)

    H = int(cfg["data"]["window"]) #512
    stride = int(cfg["data"]["stride"]) #256

    ds_train = WindowsDataset(df_train, H, stride, cfg["data"]["min_dynamic_frac"], cfg["data"]["dt"], temp_std)
    ds_val   = WindowsDataset(df_val, H, stride, cfg["data"]["min_dynamic_frac"], cfg["data"]["dt"], temp_std)
    dl_train = DataLoader(ds_train, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collate_pad)
    dl_val   = DataLoader(ds_val, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collate_pad)

    device = torch.device(cfg["device"] if torch.cuda.is_available() and cfg["device"]=="cuda" else "cpu")
    model = DCIRNeuralODE(
        ocv="analytic",
        hidden=cfg["model"]["hidden"],
        residual_hidden=cfg["model"]["residual_hidden"],
        param_scales=dict(
            r0=float(cfg["model"]["r0_scale"]),
            r1=float(cfg["model"]["r1_scale"]),
            r2=float(cfg["model"]["r2_scale"]),
            c1=float(cfg["model"]["c1_scale"]),
            c2=float(cfg["model"]["c2_scale"])
        ),
        param_eps=float(cfg["model"]["param_softplus_eps"]),
        dt=cfg["data"]["dt"]
    ).to(device)
    model.device = device

    criterion = VoltageLoss(cfg["loss"]["lambda_sm"], cfg["loss"]["lambda_res"])
    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["loss"]["weight_decay"]
    )    

    best_val = 1e9
    patience = 0
    total_steps = cfg["train"]["epochs"] * max(1, len(dl_train))

    step = 0
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for batch in dl_train:
            V = batch["V"].to(device)
            I = batch["I"].to(device)
            Tz = batch["Tz"].to(device)
            out = model(V, I, Tz)
            dV = out["V_pred"] - (model.ocv(out["states"]["soc"]) - out["params"]["R0"]*I - out["states"]["vc1"] - out["states"]["vc2"])
            loss, parts = criterion(out["V_pred"], V, out["params"], out["states"], residual_mag=dV, weight_decay=0.0, model=None)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optim.step()
            cosine_lr(optim, cfg["train"]["lr"], cfg["train"]["lr_min"], step, total_steps)
            step += 1

        # Validation
        model.eval()
        with torch.no_grad():
            vloss = 0.0
            count = 0
            for batch in dl_val:
                V = batch["V"].to(device)
                I = batch["I"].to(device)
                Tz = batch["Tz"].to(device)
                out = model(V, I, Tz)
                dV = out["V_pred"] - (model.ocv(out["states"]["soc"]) - out["params"]["R0"]*I - out["states"]["vc1"] - out["states"]["vc2"])
                loss, _ = criterion(out["V_pred"], V, out["params"], out["states"], residual_mag=dV)
                vloss += loss.item()
                count += 1
            vloss /= max(1, count)

        print(f"Epoch {epoch:03d}  val_loss={vloss:.6f}")
        if vloss < best_val:
            best_val = vloss
            patience = 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))
        else:
            patience += 1
            if patience >= cfg["train"]["early_stop_patience"]:
                print("Early stopping.")
                break
    print(f"Best val loss: {best_val:.6f}")