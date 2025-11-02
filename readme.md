# Battery DCIR — Temperature-aware 2RC Neural-ODE (RK4 + Residual NN)

This repo provides a full, reproducible pipeline to **simulate, train, and evaluate** a
temperature-aware **2RC + RK4 Neural-ODE** for **DCIR estimation**. It also includes
a **Voltage-Drop** baseline (ΔV/ΔI), stratified splits by SOC/Temperature, and publication-quality plots.

**Key ideas**
- Physics backbone: 2RC ECM with differentiable **RK4** integrator
- **Temperature-aware** parameter head → R0, R1, C1, R2, C2 (positivity via Softplus)
- **Residual NN** to capture non-modeled effects (hysteresis, bias)
- Classical **Voltage-Drop DCIR** baseline for fair alignment checks
- End-to-end **PyTorch** (autograd through RK4)

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Quickstart
### 1) Simulate data

```bash
python -m dual_method_dcir_estimator.cli.simulate_data \
  --out data/simulated/sim1.parquet --seed 42
```

### 2) Train

```bash
python -m dual_method_dcir_estimator.cli.train_model \
  --config config/default.yaml \
  --data data/simulated/sim1.parquet \
  --run_dir runs/sim1
```

### 3) Evaluate (plots + metrics + voltage-drop alignment)

```bash
python -m battery_dcir.cli.eval_model \
  --config config/default.yaml \
  --data data/simulated/sim1.parquet \
  --checkpoint runs/sim1/best.pt \
  --out_dir runs/sim1/eval
```

### 4) Real data (AVL CSV)

Place your CSV under data/real/ and call --data data/real/your.csv. <br>
We’ll parse the [HEADER]/[DATA] blocks and enforce our **sign convention: discharge-positive, charge-negative**. <br>
(If your CSV is the opposite, it is automatically transformed.)

### Config

Edit config/default.yaml for horizon, loss weights, OCV choice, etc.

### Papers

This repo supports the manuscript chapters: modeling, methodology, experiment protocol, and plots (Fig 6A–D).