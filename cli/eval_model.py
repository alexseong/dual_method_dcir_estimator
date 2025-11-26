import argparse
import os
import yaml
import torch 
import pandas as pd
from dual_method_dcir_estimator.data.loaders import dataframe_from_input
from dual_method_dcir_estimator.data.preprocess import preprocess, PreprocConfig, Standardizer
from dual_method_dcir_estimator.training.plots import overlay_voltage, soc_plot, r0_scatter
from dual_method_dcir_estimator.training.eval import evaluate_window
from dual_method_dcir_estimator.models.full_model import DCIRNeuralODE

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    raw = dataframe_from_input(args.data)
    pre_cfg = PreprocConfig(
        dt=cfg["data"]["dt"],
        current_sign=cfg["data"]["current_sign"],
        temp_standardize=cfg["data"]["temp_standardize"]
    )

    df, temp_std = preprocess(raw, pre_cfg)
    # Take one big window for demo; or reuse dataset/segmenter as needed

    H = cfg["data"]["window"]
    if len(df) < H:
        raise ValueError("Not enough data for one window; lower H.")

    seg = df.iloc[:H]
    batch = {
        "V":  torch.from_numpy(seg["V" ].values).float().unsqueeze(0),
        "I":  torch.from_numpy(seg["I" ].values).float().unsqueeze(0),
        "Tz": torch.from_numpy(seg["Tz"].values).float().unsqueeze(0),
        "t":  torch.from_numpy(seg["t" ].values).float().unsqueeze(0),
    }

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
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval

    os.makedirs(args.out_dir, exist_ok=True)
    res = evaluate_window(model, batch, float(cfg["eval"]))

if __name__ == "__main__":
    main()
