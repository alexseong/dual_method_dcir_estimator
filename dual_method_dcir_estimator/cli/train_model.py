import argparse, os
from dual_method_dcir_estimator.training.train import train_loop

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)
    train_loop(args.config, args.data, args.run_dir)

if __name__ == "__main__":
    main()