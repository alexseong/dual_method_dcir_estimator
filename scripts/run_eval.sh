#!/usr/bin/env bash
# set -e
python -m dual_method_dcir_estimator.cli.eval_model \
    --config config/default.yaml \
    --data data/simulated/sim1.parquet \
    --checkpoint runs/sim1/best.pt \
    --out_dir runs/sim1/eval