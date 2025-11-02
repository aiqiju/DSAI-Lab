#!/usr/bin/env python3
"""
Compute a simple intrinsic-dimension curve D(t) from a CSV.
This is a stub implementation using PCA-rank and a naive TwoNN proxy if available.

Usage:
  python code/compute_id_curve.py --csv data/your.csv --timecol time --out results/
"""
import argparse, os, sys
import numpy as np
import pandas as pd

def pca_rank(X, var_ratio=0.95):
    # simple PCA via SVD (no sklearn dependency)
    X = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = (S**2)
    cum = np.cumsum(var) / np.sum(var)
    k = int(np.searchsorted(cum, var_ratio) + 1)
    return max(1, min(k, X.shape[1]))

def estimate_id_per_time(df, timecol, var_ratio=0.95):
    times = []
    ids = []
    for t, sub in df.groupby(timecol):
        X = sub.drop(columns=[timecol]).values
        if len(sub) < 3:
            continue
        k = pca_rank(X, var_ratio=var_ratio)
        times.append(t)
        ids.append(k)
    return pd.DataFrame({"time": times, "D_pca": ids}).sort_values("time")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--timecol", default="time")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv)
    if args.timecol not in df.columns:
        print(f"ERROR: time column '{args.timecol}' not in CSV.", file=sys.stderr)
        sys.exit(1)

    curve = estimate_id_per_time(df, args.timecol)
    out_csv = os.path.join(args.out, "id_curve.csv")
    curve.to_csv(out_csv, index=False)

    # Simple plot (matplotlib if available)
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(curve["time"], curve["D_pca"], marker="o")
        plt.xlabel("time")
        plt.ylabel("intrinsic dimension (PCA 95%)")
        plt.title("D(t) — PCA-based estimate")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "id_curve.png"), dpi=150)
    except Exception as e:
        print("Plotting skipped:", e)

    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    main()
