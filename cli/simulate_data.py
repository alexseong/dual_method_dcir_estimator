import argparse
from dual_method_dcir_estimator.data.simulate import simulate_2rc_sequence, save_parquet

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--T", type=int, default=8000)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--Imax", type=float, default=25.0)
    args = ap.parse_args()

    df = simulate_2rc_sequence(T=args.T, dt=args.dt, Imax=args.Imax, seed=args.seed)
    save_parquet(df, args.out)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()