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
    """
    按 time 分组，用 SVD-PCA 估计“解释到 var_ratio 的主成分个数”作为 D_pca。
    仅使用数值特征列；每组样本少于3条则跳过。
    """
    times, ids = [], []

    # 仅保留数值特征列（去掉time本身）
    feat_cols = [
        c for c in df.columns
        if c != timecol and pd.api.types.is_numeric_dtype(df[c])
    ]

    for t, sub in df.groupby(timecol):
        if len(sub) < 3 or len(feat_cols) == 0:
            continue
        X = sub[feat_cols].to_numpy()

        # 防御：如果全 NaN 或者列数为0，跳过
        if X.size == 0 or np.all(~np.isfinite(X)):
            continue

        k = pca_rank(X, var_ratio=var_ratio)
        # 夹在 [1, 特征数] 范围内
        k = int(max(1, min(k, X.shape[1])))

        times.append(t)
        ids.append(k)

    return pd.DataFrame({"time": times, "D_pca": ids}).sort_values("time")




# --- TwoNN intrinsic dimension estimator (Facco et al., 2017, minimal naive impl) ---
def _twonn_id(X):
    """
    X: (n_samples, n_features)
    returns: scalar ID estimate or None
    """
    n = X.shape[0]
    if n < 10:  # need enough points to be meaningful
        return None
    # compute pairwise distances (naive O(n^2))
    # use Euclidean; center for stability
    Xc = X - X.mean(axis=0, keepdims=True)
    d2 = np.sum((Xc[:, None, :] - Xc[None, :, :])**2, axis=2)
    d = np.sqrt(np.maximum(d2, 0.0))
    # for each point, find 1st and 2nd nearest neighbors (excluding self=0)
    r1 = np.empty(n)
    r2 = np.empty(n)
    for i in range(n):
        di = np.sort(d[i][d[i] > 0])  # exclude self
        if len(di) < 2:
            return None
        r1[i], r2[i] = di[0], di[1]
    mu = r2 / r1
    # discard pathological values
    mu = mu[np.isfinite(mu) & (mu > 1)]
    if len(mu) < 10:
        return None
    # ID estimate: m = -1 / <log(mu - 1?) or log(mu)?>
    # For TwoNN, use E[log(mu)] = 1/m  => m = 1 / mean(log(mu))
    val = np.mean(np.log(mu))
    if val <= 0:
        return None
    return float(1.0 / val)

def estimate_id_per_time_twonn(df, timecol):
    times, ids = [], []
    for t, sub in df.groupby(timecol):
        X = sub.drop(columns=[timecol]).select_dtypes(include=["number"]).values
        if X.shape[0] < 10:
            continue
        m = _twonn_id(X)
        if m is not None and np.isfinite(m):
            times.append(t)
            # cap to feature count (optional)
            ids.append(min(m, X.shape[1]))
    return pd.DataFrame({"time": times, "D_twonn": ids}).sort_values("time")


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

    ap.add_argument(
        "--var_ratio", type=float, default=0.95,
        help="PCA累计方差阈值(0~1)，如 0.90 / 0.95 / 0.99"
    )


    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv)
    if args.timecol not in df.columns:
        print(f"ERROR: time column '{args.timecol}' not in CSV.", file=sys.stderr)
        sys.exit(1)

    curve = estimate_id_per_time(df, args.timecol, var_ratio=args.var_ratio)

    out_csv = os.path.join(args.out, "id_curve.csv")
    curve.to_csv(out_csv, index=False)


    # TwoNN-based curve
    curve2 = estimate_id_per_time_twonn(df, args.timecol)
    out_csv2 = os.path.join(args.out, "id_curve_twonn.csv")
    curve2.to_csv(out_csv2, index=False)


    # changepoint detection on (time, D_pca)
    cp = detect_changepoint(curve["time"].tolist(), curve["D_pca"].tolist())
    with open(os.path.join(args.out, "changepoint.json"), "w") as f:
        json.dump(cp, f, indent=2)


    # Simple plot (matplotlib if available)
    try:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(curve["time"], curve["D_pca"], marker="o", label="PCA(95%)")
        try:
            if not curve2.empty:
                plt.plot(curve2["time"], curve2["D_twonn"], marker="s", linestyle="--", label="TwoNN")
        except Exception:
            pass
        plt.xlabel("time")
        plt.ylabel("intrinsic dimension")
        title = "D(t) — PCA vs TwoNN"
        try:
            if cp.get("best_t") is not None:
                bt = cp["best_t"]
                plt.axvline(bt, linestyle="--")
                title += f"  |  changepoint≈{bt}, ΔBIC={cp['delta_bic']:.2f}"
        except Exception:
            pass
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "id_curve.png"), dpi=150)


    except Exception as e:
        print("Plotting skipped:", e)

    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    main()
