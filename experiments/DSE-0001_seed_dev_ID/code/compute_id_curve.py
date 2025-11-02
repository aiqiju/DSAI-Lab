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
# --- simple changepoint detection: two-piece linear fit vs single line ---
import math, json

def _sse_line(x, y):
    # y ≈ a*x + b
    if len(x) < 2:
        return float("inf")
    a, b = np.polyfit(x, y, 1)
    yhat = a*np.array(x) + b
    return float(np.sum((np.array(y) - yhat)**2))

def detect_changepoint(times, values):
    """
    times: list/array of x (e.g., [0,1,2,...])
    values: list/array of y (D_pca)
    Returns: dict with best_t, delta_bic, sse_single, sse_two
    BIC_single = n*ln(RSS/n) + k*ln(n) with k=2 (slope,intercept)
    BIC_two    = n*ln(RSS/n) + k*ln(n) with k=4 (two lines)
    """
    x = list(times)
    y = list(values)
    n = len(x)
    if n < 4:  # need at least 2+2 points to split
        return {"best_t": None, "delta_bic": 0.0, "sse_single": None, "sse_two": None}

    sse_single = _sse_line(x, y)
    # scan internal breakpoints (ensure >=2 points on each side)
    best = {"best_t": None, "sse_two": float("inf")}
    for i in range(2, n-1):  # candidate index i splits at x[i-1]/x[i]
        left_x,  left_y  = x[:i], y[:i]
        right_x, right_y = x[i:], y[i:]
        sse2 = _sse_line(left_x, left_y) + _sse_line(right_x, right_y)
        if sse2 < best["sse_two"]:
            best = {"best_t": x[i], "sse_two": sse2}

    # BIC
    # k_single=2; k_two=4
    if sse_single <= 0 or best["sse_two"] <= 0:
        delta_bic = 0.0
    else:
        bic_single = n*math.log(sse_single/n) + 2*math.log(n)
        bic_two    = n*math.log(best["sse_two"]/n) + 4*math.log(n)
        delta_bic  = bic_single - bic_two  # >0 means two-piece is better

    return {
        "best_t": best["best_t"],
        "delta_bic": float(delta_bic),
        "sse_single": float(sse_single),
        "sse_two": float(best["sse_two"]),
    }


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

    # changepoint detection on (time, D_pca)
    cp = detect_changepoint(curve["time"].tolist(), curve["D_pca"].tolist())
    with open(os.path.join(args.out, "changepoint.json"), "w") as f:
        json.dump(cp, f, indent=2)


    # Simple plot (matplotlib if available)
    try:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(curve["time"], curve["D_pca"], marker="o")
        plt.xlabel("time")
        plt.ylabel("intrinsic dimension (PCA 95%)")
        title = "D(t) — PCA-based estimate"
        try:
            if cp.get("best_t") is not None:
                bt = cp["best_t"]
                plt.axvline(bt, linestyle="--")
                title += f"  |  changepoint≈{bt}, ΔBIC={cp['delta_bic']:.2f}"
        except Exception:
            pass
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "id_curve.png"), dpi=150)


    except Exception as e:
        print("Plotting skipped:", e)

    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    main()
