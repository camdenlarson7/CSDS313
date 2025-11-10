#!/usr/bin/env python
# task3_cars_mileage_test.py
"""
Analyze 'cars.csv' where:
- Column 1 = US car mileage (numeric)
- Column 2 = Japanese car mileage (numeric)

Outputs:
(a) Sample sizes
(b) Hypotheses
(c) Test statistic + reference distribution (Welch's t with Satterthwaite df)
(d) Statistic value + p-value (one-sided, H1: mu_Japan > mu_US)
(e) Conclusion at chosen alpha

Also prints descriptive stats and optional robustness (Mann–Whitney U).
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, shapiro, mannwhitneyu

def load_cars_csv(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load first two columns as US, Japan arrays (drop NaNs)."""
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError("Expected at least 2 columns: [US, Japan].")
    col_names = list(df.columns[:2])
    us = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().to_numpy(dtype=float)
    jp = pd.to_numeric(df.iloc[:, 1], errors="coerce").dropna().to_numpy(dtype=float)
    return us, jp, col_names

def welch_df(s1_sq: float, n1: int, s2_sq: float, n2: int) -> float:
    """Satterthwaite approximation for Welch's t degrees of freedom."""
    num = (s1_sq / n1 + s2_sq / n2) ** 2
    den = (s1_sq**2 / (n1**2 * (n1 - 1))) + (s2_sq**2 / (n2**2 * (n2 - 1)))
    return num / den

def describe(arr: np.ndarray) -> dict:
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)) if arr.size > 0 else np.nan,
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else np.nan,
        "min": float(np.min(arr)) if arr.size > 0 else np.nan,
        "q25": float(np.percentile(arr, 25)) if arr.size > 0 else np.nan,
        "median": float(np.median(arr)) if arr.size > 0 else np.nan,
        "q75": float(np.percentile(arr, 75)) if arr.size > 0 else np.nan,
        "max": float(np.max(arr)) if arr.size > 0 else np.nan,
    }

def main():
    ap = argparse.ArgumentParser(description="Task 3: Compare US vs Japanese car mileage.")
    ap.add_argument("--csv", type=str, default="cars.csv", help="Path to cars.csv")
    ap.add_argument("--alpha", type=float, default=0.05, help="Significance level (default 0.05)")
    ap.add_argument("--robust", action="store_true", help="Also run Mann–Whitney U (nonparametric) as a robustness check.")
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(1)

    us, jp, cols = load_cars_csv(path)

    # (a) How many US / Japanese cars?
    desc_us = describe(us)
    desc_jp = describe(jp)

    print("=== (a) Sample sizes ===")
    print(f"US count (column '{cols[0]}'):      n_us = {desc_us['n']}")
    print(f"Japanese count (column '{cols[1]}'): n_jp = {desc_jp['n']}\n")

    # (b) Hypotheses (sales claim: Japanese > US)
    print("=== (b) Hypothesis ===")
    print("H0: μ_Japan ≤ μ_US")
    print("H1: μ_Japan  > μ_US\n")

    # (c) Test statistic & theoretical distribution
    print("=== (c) Test statistic & reference distribution ===")
    print("Statistic: Welch's two-sample t-statistic")
    print("Reference distribution: approximately t with Satterthwaite (Welch) degrees of freedom\n")

    # Descriptives
    print("Descriptive statistics:")
    print(f"US:     mean = {desc_us['mean']:.6g}, std = {desc_us['std']:.6g}, n = {desc_us['n']}")
    print(f"Japan:  mean = {desc_jp['mean']:.6g}, std = {desc_jp['std']:.6g}, n = {desc_jp['n']}\n")

    # Assumption checks (optional; Welch t is robust to unequal variances)
    # Normality checks are often low-power or too sensitive; use as a light sanity check.
    if 3 <= desc_us['n'] <= 5000:
        sw_us = shapiro(us)
        print(f"Shapiro–Wilk (US):    W = {sw_us.statistic:.4f}, p = {sw_us.pvalue:.4g}")
    if 3 <= desc_jp['n'] <= 5000:
        sw_jp = shapiro(jp)
        print(f"Shapiro–Wilk (Japan): W = {sw_jp.statistic:.4f}, p = {sw_jp.pvalue:.4g}")
    print()

    # (d) Compute Welch t-test (one-sided, H1: mu_Japan > mu_US)
    res = ttest_ind(jp, us, equal_var=False, alternative="greater")
    t_stat = float(res.statistic)
    p_val  = float(res.pvalue)

    # Welch df
    s1_sq, s2_sq = np.var(jp, ddof=1), np.var(us, ddof=1)
    df = welch_df(s1_sq, desc_jp['n'], s2_sq, desc_us['n'])

    mean_diff = desc_jp['mean'] - desc_us['mean']

    print("=== (d) Test result ===")
    print(f"Welch t-statistic = {t_stat:.6g}, df ≈ {df:.3f}, one-sided p-value = {p_val:.6g}")
    print(f"Mean difference (Japan - US) = {mean_diff:.6g}\n")

    # Optional robustness: Mann–Whitney U (one-sided, Japan > US)
    if args.robust:
        # Note: For large samples, use 'asymptotic' method; exact disabled if ties.
        mwu = mannwhitneyu(jp, us, alternative="greater", method="asymptotic")
        print("Robustness (Mann–Whitney U, one-sided H1: Japan > US):")
        print(f"U = {mwu.statistic:.6g}, p-value = {mwu.pvalue:.6g}\n")

    # (e) Conclusion
    print("=== (e) Conclusion ===")
    if p_val < args.alpha:
        print(f"At α = {args.alpha}, reject H0. The data provide statistical evidence that Japanese cars")
        print("have higher mileage on average than US cars (in this sample).")
    else:
        print(f"At α = {args.alpha}, fail to reject H0. The data do not provide sufficient evidence that")
        print("Japanese cars have higher mileage on average than US cars (in this sample).")

    print("\n[Note] Welch's t-test does not assume equal variances; it assumes independent samples and that")
    print("the sampling distribution of the mean is approximately normal (often reasonable by CLT for")
    print("moderate n). Consider outliers/heavy tails; the Mann–Whitney U provides a rank-based check.")

if __name__ == "__main__":
    main()
