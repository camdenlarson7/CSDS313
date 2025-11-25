import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def load_two_columns(csv_path: str, col_names=None):
    df = pd.read_csv(csv_path, header=None, names=["X","Y"])
    if col_names is None:
        if df.shape[1] != 2:
            raise ValueError(
                f"{csv_path} has {df.shape[1]} columns; specify exactly two with --cols"
            )
        cols = list(df.columns)
    else:
        if len(col_names) != 2:
            raise ValueError("--cols requires exactly two column names.")
        cols = col_names
    x = df[cols[0]].to_numpy(dtype=float)
    y = df[cols[1]].to_numpy(dtype=float)
    return x, y, cols

def scatter_with_fit(x, y, cols, outpath):
    plt.figure(figsize=(6.6, 4.4))
    plt.scatter(x, y, s=12, alpha=0.85)
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(np.min(x), np.max(x), 200)
    plt.plot(xs, m * xs + b, linewidth=2)
    plt.xlabel(cols[0])
    plt.ylabel(cols[1])
    plt.title(f"Scatter with best-fit line: {cols[0]} vs {cols[1]}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def fisher_ci(r, n, alpha=0.05):
    if n <= 3 or abs(r) >= 1:
        return (r, r)
    z = 0.5 * np.log((1 + r) / (1 - r))
    se = 1 / np.sqrt(n - 3)
    from scipy.stats import norm
    zcrit = norm.ppf(1 - alpha / 2)
    lo = z - zcrit * se
    hi = z + zcrit * se
    lo_r = (np.exp(2 * lo) - 1) / (np.exp(2 * lo) + 1)
    hi_r = (np.exp(2 * hi) - 1) / (np.exp(2 * hi) + 1)
    return (float(lo_r), float(hi_r))

def main():
    ap = argparse.ArgumentParser(description="Task 2(a): Pearson correlation for p2a.csv")
    ap.add_argument("--data", default=os.path.join("..", "data", "p2a.csv"),
                    help="Path to p2a.csv (default: ../data/p2a.csv from src/)")
    ap.add_argument("--cols", nargs=2, default=None,
                    help="Two column names to use (default: use the only two in the file).")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="Significance level alpha (default: 0.05).")
    ap.add_argument("--plot", action="store_true",
                    help="If set, save a scatter plot with best-fit line to ../results/")
    args = ap.parse_args()

    # Load data
    x, y, cols = load_two_columns(args.data, args.cols)
    n = len(x)

    # Pearson correlation (two-sided p-value)
    r, p = pearsonr(x, y)

    # CI for effect-size context
    lo, hi = fisher_ci(r, n, alpha=args.alpha)

    # Print report
    print("=" * 70)
    print("Task 2(a): Pearson correlation for two variables")
    print(f"File: {args.data}")
    print(f"Columns: {cols[0]} (X), {cols[1]} (Y)")
    print(f"Samples (N): {n}")
    print("-" * 70)
    print(f"Pearson r: {r:.6f}")
    print(f"Two-sided p-value: {p:.6g}")
    print(f"{int((1-args.alpha)*100)}% CI for r (Fisher): [{lo:.3f}, {hi:.3f}]")
    print("-" * 70)
    print(f"Alpha (α): {args.alpha}")
    decision = "REJECT H0" if p < args.alpha else "fail to reject H0"
    print(f"Decision: {decision}")
    direction = "positive" if r > 0 else ("negative" if r < 0 else "none")
    magnitude = "negligible"
    ar = abs(r)
    if ar >= 0.5:
        magnitude = "large"
    elif ar >= 0.3:
        magnitude = "moderate"
    elif ar >= 0.1:
        magnitude = "small"
    print(f"Interpretation: {direction} association, {magnitude} magnitude.")
    print("=" * 70)

    # Plot
    if args.plot:
        results_dir = os.path.join("..", "results")
        os.makedirs(results_dir, exist_ok=True)
        outpath = os.path.join(results_dir, "p2a_scatter.png")
        scatter_with_fit(x, y, cols, outpath)
        print(f"Saved scatter to: {outpath}")

if __name__ == "__main__":
    main()
