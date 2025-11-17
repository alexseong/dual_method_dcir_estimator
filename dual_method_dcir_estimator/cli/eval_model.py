import argparse
import os
import yaml
import torch 
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
    ap.add_argument("--run_dir", required=True)
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

    os.makedirs(args.run_dir, exist_ok=True)
    train_loop(args.config, args.data, args.run_dir)

if __name__ == "__main__":
    main()