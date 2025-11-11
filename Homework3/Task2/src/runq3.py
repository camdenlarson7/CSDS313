# Task2/src/run_p2c.py
import argparse, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import pearsonr, norm

def load_two_columns(csv_path: str, col_names=None, header="infer"):
    df = pd.read_csv(csv_path, header=None if header is None else "infer")
    if col_names is None:
        if df.shape[1] != 2:
            raise ValueError(f"{csv_path} has {df.shape[1]} columns; specify exactly two with --cols")
        cols = list(df.columns)
    else:
        if len(col_names) != 2:
            raise ValueError("--cols requires exactly two column names.")
        cols = col_names
    x = df[cols[0]].to_numpy(dtype=float)
    y = df[cols[1]].to_numpy(dtype=float)
    return x, y, [str(cols[0]), str(cols[1])]

def fisher_ci(r, n, alpha=0.05):
    if n <= 3 or abs(r) >= 1: return (r, r)
    z = 0.5*np.log((1+r)/(1-r)); se = 1/np.sqrt(n-3)
    zcrit = norm.ppf(1 - alpha/2)
    lo = z - zcrit*se; hi = z + zcrit*se
    to_r = lambda z: (np.exp(2*z)-1)/(np.exp(2*z)+1)
    return float(to_r(lo)), float(to_r(hi))

def scatter_with_fit(x, y, cols, outpath, title):
    plt.figure(figsize=(6.6, 4.4))
    plt.scatter(x, y, s=12, alpha=0.85)
    m, b = np.polyfit(x, y, 1); xs = np.linspace(np.min(x), np.max(x), 200)
    plt.plot(xs, m*xs + b, linewidth=2)
    plt.xlabel(cols[0]); plt.ylabel(cols[1]); plt.title(title)
    plt.tight_layout(); os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=150); plt.close()

def main():
    ap = argparse.ArgumentParser(description="Task 2(c): Pearson correlation for p2c.csv")
    ap.add_argument("--data", default=os.path.join("..", "data", "p2c.csv"),
                    help="Path to p2c.csv (default: ../data/p2c.csv from src/)")
    ap.add_argument("--cols", nargs=2, default=None, help="Two column names to use.")
    ap.add_argument("--alpha", type=float, default=0.05, help="Significance level alpha.")
    ap.add_argument("--plot", action="store_true", help="Save scatter plot to ../results/p2c_scatter.png")
    ap.add_argument("--header", choices=["infer","none"], default="infer",
                    help="CSV header handling (default infer; use 'none' if file has no header).")
    args = ap.parse_args()

    header = None if args.header == "none" else "infer"
    x, y, cols = load_two_columns(args.data, args.cols, header=header)
    n = len(x)

    r, p = pearsonr(x, y)
    lo, hi = fisher_ci(r, n, alpha=args.alpha)

    print("="*70)
    print("Task 2(c): Pearson correlation for two variables")
    print(f"File: {args.data}")
    print(f"Columns: {cols[0]} (X), {cols[1]} (Y)")
    print(f"Samples (N): {n}")
    print("-"*70)
    print(f"Pearson r: {r:.6f}")
    print(f"Two-sided p-value: {p:.6g}")
    print(f"{int((1-args.alpha)*100)}% CI for r (Fisher): [{lo:.3f}, {hi:.3f}]")
    print("-"*70)
    print(f"Alpha (α): {args.alpha}")
    decision = "REJECT H0" if p < args.alpha else "fail to reject H0"
    print(f"Decision: {decision}")
    direction = "positive" if r > 0 else ("negative" if r < 0 else "none")
    magnitude = "negligible"
    ar = abs(r)
    if ar >= 0.5: magnitude = "large"
    elif ar >= 0.3: magnitude = "moderate"
    elif ar >= 0.1: magnitude = "small"
    print(f"Interpretation: {direction} association, {magnitude} magnitude.")
    print("="*70)

    if args.plot:
        scatter_with_fit(x, y, cols, os.path.join("..","results","p2c_scatter.png"),
                         title=f"p2c: {cols[0]} vs {cols[1]}")
        print("Saved scatter to: ../results/p2c_scatter.png")

if __name__ == "__main__":
    main()
