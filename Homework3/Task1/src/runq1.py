import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from metrics import (
    contingency_from_binary,
    mi_from_counts,
    jaccard_index_from_counts,
    chi_square_from_counts,
    permutation_test,
    mi_stat_from_vectors,
    jaccard_stat_from_vectors,
)

def load_two_columns(csv_path: str, col_names=None):
    df = pd.read_csv(csv_path)
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
    x = df[cols[0]].to_numpy()
    y = df[cols[1]].to_numpy()
    return x, y, cols

def ensure_binary(vec: np.ndarray, name: str):
    u = np.unique(vec)
    if not set(u).issubset({0, 1}):
        raise ValueError(f"Column '{name}' must be binary (0/1). Found values: {u}")

def plot_perm(sorted_perm, observed, title, outpath):
    plt.figure(figsize=(7, 4.2))
    idx = np.arange(1, len(sorted_perm) + 1)
    plt.scatter(idx, sorted_perm, s=8)
    plt.axhline(observed, linestyle="--", linewidth=2)
    plt.xlabel("Permutation index (sorted)")
    plt.ylabel("Statistic value")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="P1a: Association between two binary variables.")
    parser.add_argument("--data", default=os.path.join("..", "data", "p1a.csv"),
                        help="Path to p1a.csv (default: ../data/p1a.csv from src/)")
    parser.add_argument("--cols", nargs=2, default=None, help="Two column names in CSV (default: take the only two).")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level alpha (default: 0.05).")
    parser.add_argument("--perms", type=int, default=10000, help="Number of permutations for MI and Jaccard (default: 10000).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for permutation test.")
    parser.add_argument("--plots", action="store_true", help="If set, save permutation plots to ../results/")
    args = parser.parse_args()

    # Paths
    results_dir = os.path.join("..", "results")
    if args.plots:
        os.makedirs(results_dir, exist_ok=True)

    # Load data
    x, y, cols = load_two_columns(args.data, args.cols)
    ensure_binary(x, cols[0])
    ensure_binary(y, cols[1])

    # Build contingency
    counts = contingency_from_binary(x, y)
    N = counts.sum()

    # Statistics
    # Mutual Information (bits)
    mi_obs, mi_p, mi_perm = permutation_test(
        stat_fn=mi_stat_from_vectors,
        x=x, y=y,
        n_perms=args.perms,
        rng=np.random.default_rng(args.seed),
        alternative="greater"
    )

    # Jaccard Index
    ji_obs, ji_p, ji_perm = permutation_test(
        stat_fn=jaccard_stat_from_vectors,
        x=x, y=y,
        n_perms=args.perms,
        rng=np.random.default_rng(args.seed + 1),
        alternative="greater"
    )

    # Pearson's chi-square (parametric)
    chi2_stat, chi2_p, expected = chi_square_from_counts(counts)

    # report
    print("=" * 70)
    print("P1a: Association between two binary variables")
    print(f"File: {args.data}")
    print(f"Columns: {cols[0]} (X), {cols[1]} (Y)")
    print(f"Samples (N): {int(N)}")
    print("-" * 70)
    print("Contingency table (rows=X, cols=Y):")
    print(pd.DataFrame(counts, index=[f"{cols[0]}=0", f"{cols[0]}=1"], columns=[f"{cols[1]}=0", f"{cols[1]}=1"]))
    print("-" * 70)
    print(f"Mutual Information (bits): {mi_obs:.6f} | perm p-value: {mi_p:.6g} | perms: {args.perms}")
    print(f"Jaccard Index:             {ji_obs:.6f} | perm p-value: {ji_p:.6g} | perms: {args.perms}")
    print(f"Pearson chi-square:        {chi2_stat:.6f} | param p-value: {chi2_p:.6g}")
    print("-" * 70)
    print(f"Alpha (α): {args.alpha}")
    print("Decisions:")
    print(f"  MI:       {'REJECT H0' if mi_p < args.alpha else 'fail to reject H0'}")
    print(f"  Jaccard:  {'REJECT H0' if ji_p < args.alpha else 'fail to reject H0'}")
    print(f"  χ²:       {'REJECT H0' if chi2_p < args.alpha else 'fail to reject H0'}")
    print("=" * 70)

    # plots
    if args.plots:
        mi_sorted = np.sort(mi_perm)
        ji_sorted = np.sort(ji_perm)
        plot_perm(mi_sorted, mi_obs,
                  title=f"MI permutation null (N={args.perms})",
                  outpath=os.path.join(results_dir, "p1a_mi_permutation.png"))
        plot_perm(ji_sorted, ji_obs,
                  title=f"Jaccard permutation null (N={args.perms})",
                  outpath=os.path.join(results_dir, "p1a_jaccard_permutation.png"))
        print(f"Saved plots to: {results_dir}")

if __name__ == "__main__":
    main()
