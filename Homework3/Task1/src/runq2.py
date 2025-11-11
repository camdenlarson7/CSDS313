import argparse
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import (
    contingency_from_binary,
    mi_from_counts,
    jaccard_index_from_counts,
    chi_square_from_counts,
)

# Benjamini–Hochberg (FDR)
def benjamini_hochberg(pvals, alpha=0.05):
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ro = np.empty_like(order)
    ro[order] = np.arange(n) + 1
    # adjusted p-values
    adj = p * n / ro
    adj_sorted = adj[order]
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj = np.empty_like(adj_sorted)
    adj[order] = adj_sorted
    thresh = (ro / n) * alpha
    reject = p <= thresh
    return reject, np.minimum(adj, 1.0)

# Permutation helpers
def mi_stat_from_vectors(x, y):
    counts = contingency_from_binary(x, y)
    return mi_from_counts(counts, base=2.0)

def jaccard_stat_from_vectors(x, y):
    counts = contingency_from_binary(x, y)
    return jaccard_index_from_counts(counts)

def permutation_pvalue(stat_fn, x, y, perm_mat_y, alternative="greater"):
    obs = stat_fn(x, y)
    perm_stats = np.array([stat_fn(x, y_perm) for y_perm in perm_mat_y])
    if alternative == "greater":
        c = int(np.sum(perm_stats >= obs))
    else:
        raise ValueError("This script uses 'greater' alternative for MI/JI.")
    pval = (c + 1) / (perm_mat_y.shape[0] + 1)
    return float(obs), float(pval), perm_stats

# Plots
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def scatter_xy(x, y, xlabel, ylabel, title, outpath):
    plt.figure(figsize=(6.2, 4.4))
    plt.scatter(x, y, s=12, alpha=0.8)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="P1b: All-pairs association with FDR control.")
    ap.add_argument("--data", default=os.path.join("..", "data", "p1b.csv"),
                    help="Path to p1b.csv (default: ../data/p1b.csv from src/)")
    ap.add_argument("--alpha", type=float, default=0.05, help="FDR level alpha (default: 0.05).")
    ap.add_argument("--perms", type=int, default=50000,
                    help="Number of permutations for MI & JI (default: 50,000). Increase if needed.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--plots", action="store_true", help="Save comparison plots to ../results/")
    args = ap.parse_args()
    results_dir = os.path.join("..", "results")
    if args.plots:
        ensure_dir(results_dir)

    # Load data
    df = pd.read_csv(args.data)
    cols = list(df.columns)
    ncols = len(cols)
    if ncols < 2:
        raise ValueError("p1b.csv must have >= 2 columns.")
    # Ensure binary
    for c in cols:
        u = set(pd.unique(df[c]))
        if not u.issubset({0, 1}):
            raise ValueError(f"Column '{c}' must be binary 0/1. Found: {sorted(u)}")

    X = df.to_numpy(dtype=int)
    n_samples, n_vars = X.shape

    # Precompute permutations for each column (permute column independently)
    rng = np.random.default_rng(args.seed)
    perm_indices_by_col = [
        np.vstack([rng.permutation(n_samples) for _ in range(args.perms)])
        for _ in range(n_vars)
    ]

    # Iterate over all pairs
    records = []
    for i, j in itertools.combinations(range(n_vars), 2):
        xi = X[:, i]
        yj = X[:, j]
        # Observed counts and stats
        counts = contingency_from_binary(xi, yj)
        mi_obs = mi_from_counts(counts, base=2.0)
        ji_obs = jaccard_index_from_counts(counts)
        chi2_stat, chi2_p, _ = chi_square_from_counts(counts)

        # Permutations for MI & JI: permute Y only (pairwise, reuse precomputed column-j perms)
        perm_mat_y = yj[perm_indices_by_col[j]]
        mi_obs2, mi_p, _ = permutation_pvalue(mi_stat_from_vectors, xi, yj, perm_mat_y, "greater")
        ji_obs2, ji_p, _ = permutation_pvalue(jaccard_stat_from_vectors, xi, yj, perm_mat_y, "greater")
        assert abs(mi_obs - mi_obs2) < 1e-12 and abs(ji_obs - ji_obs2) < 1e-12

        records.append({
            "col_i": cols[i],
            "col_j": cols[j],
            "mi": mi_obs,
            "mi_p": mi_p,
            "ji": ji_obs,
            "ji_p": ji_p,
            "chi2": chi2_stat,
            "chi2_p": chi2_p,
        })

    res = pd.DataFrame.from_records(records)

    # BH FDR at alpha for each family of tests
    mi_reject, mi_q = benjamini_hochberg(res["mi_p"].values, alpha=args.alpha)
    ji_reject, ji_q = benjamini_hochberg(res["ji_p"].values, alpha=args.alpha)
    chi_reject, chi_q = benjamini_hochberg(res["chi2_p"].values, alpha=args.alpha)

    res["mi_q"] = mi_q
    res["ji_q"] = ji_q
    res["chi2_q"] = chi_q
    res["mi_sig"] = mi_reject
    res["ji_sig"] = ji_reject
    res["chi2_sig"] = chi_reject

    # Counts and overlaps
    mi_set = set(res.index[res["mi_sig"]])
    ji_set = set(res.index[res["ji_sig"]])
    chi_set = set(res.index[res["chi2_sig"]])

    n_mi = len(mi_set)
    n_ji = len(ji_set)
    n_chi = len(chi_set)
    n_mi_ji = len(mi_set & ji_set)
    n_mi_chi = len(mi_set & chi_set)
    n_ji_chi = len(ji_set & chi_set)
    n_all3 = len(mi_set & ji_set & chi_set)

    # Save CSV
    out_csv = os.path.join("..", "results", "p1b_summary.csv")
    ensure_dir(os.path.dirname(out_csv))
    res.to_csv(out_csv, index=False)

    # Report
    print("=" * 80)
    print(f"P1b results for {args.data}  |  pairs = {len(res)}  |  N_samples = {n_samples}")
    print(f"Alpha (FDR): {args.alpha}  |  Permutations: {args.perms}")
    print("-" * 80)
    print(f"Significant pairs (BH-FDR @ alpha):")
    print(f"  MI   : {n_mi}")
    print(f"  JI   : {n_ji}")
    print(f"  χ²   : {n_chi}")
    print("-" * 80)
    print("Overlaps among significant sets:")
    print(f"  MI ∩ JI       : {n_mi_ji}")
    print(f"  MI ∩ χ²       : {n_mi_chi}")
    print(f"  JI ∩ χ²       : {n_ji_chi}")
    print(f"  MI ∩ JI ∩ χ²  : {n_all3}")
    print("-" * 80)
    print(f"Saved per-pair table to: {out_csv}")
    print("=" * 80)

    if args.plots:
        # Scatter plots comparing statistics
        scatter_xy(res["mi"], res["ji"],
                   "MI (bits)", "JI",
                   "MI vs JI (statistics)",
                   os.path.join(results_dir, "p1b_scatter_mi_vs_ji_stats.png"))
        scatter_xy(res["mi"], res["chi2"],
                   "MI (bits)", "chi-square",
                   "MI vs chi-square (statistics)",
                   os.path.join(results_dir, "p1b_scatter_mi_vs_chi2_stats.png"))
        scatter_xy(res["ji"], res["chi2"],
                   "JI", "chi-square",
                   "JI vs chi-square (statistics)",
                   os.path.join(results_dir, "p1b_scatter_ji_vs_chi2_stats.png"))

        # Scatter plots comparing -log10 p-values
        eps = 1e-300 
        scatter_xy(-np.log10(res["mi_p"] + eps), -np.log10(res["ji_p"] + eps),
                   r"$-\,\log_{10} p_{\mathrm{MI}}$", r"$-\,\log_{10} p_{\mathrm{JI}}$",
                   "MI vs JI (-log10 p-values)",
                   os.path.join(results_dir, "p1b_scatter_mi_vs_ji_pvals.png"))
        scatter_xy(-np.log10(res["mi_p"] + eps), -np.log10(res["chi2_p"] + eps),
                   r"$-\,\log_{10} p_{\mathrm{MI}}$", r"$-\,\log_{10} p_{\chi^2}$",
                   "MI vs chi-square (-log10 p-values)",
                   os.path.join(results_dir, "p1b_scatter_mi_vs_chi2_pvals.png"))
        scatter_xy(-np.log10(res["ji_p"] + eps), -np.log10(res["chi2_p"] + eps),
                   r"$-\,\log_{10} p_{\mathrm{JI}}$", r"$-\,\log_{10} p_{\chi^2}$",
                   "JI vs chi-square (-log10 p-values)",
                   os.path.join(results_dir, "p1b_scatter_ji_vs_chi2_pvals.png"))

        print(f"Saved plots to: {results_dir}")

if __name__ == "__main__":
    main()
