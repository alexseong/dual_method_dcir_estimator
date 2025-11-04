#!/usr/bin/env bash
# set -e
python -m dual_method_dcir_estimator.cli.train_model \
    --config config/default.yaml \
    --data data/simulated/sim1.parquet \
    --run_dir runs/sim1