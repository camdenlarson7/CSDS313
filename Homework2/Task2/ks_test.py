# ks_test_sample_means.py
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import argparse

def parse_reps_samps(name: str):
    """Extract '<reps>x<samples>' from filename."""
    m = re.search(r"(\d+)x(\d+)", name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

def load_means_dir(mean_dir: Path):
    """
    Load all '*_means_*.csv' in mean_dir.
    Returns a dict: {(reps, samples): np.ndarray of means}
    """
    out = {}
    for f in sorted(mean_dir.glob("*_means_*.csv")):
        df = pd.read_csv(f)
        if "Sample Mean" in df.columns:
            vals = df["Sample Mean"].to_numpy(dtype=float)
        else:
            vals = df.select_dtypes(include="number").to_numpy(dtype=float).ravel()
        reps, samps = parse_reps_samps(f.name)
        if reps is not None:
            out[(reps, samps)] = vals
    return out

def ecdf(values: np.ndarray):
    """Empirical CDF data for step plot."""
    x = np.sort(values)
    n = x.size
    y = np.arange(1, n+1) / n
    return x, y

def plot_ecdf_compare(u_vals, p_vals, title: str, out_png: Path, D: float, pval: float):
    """ECDF overlay for two samples with annotation of KS stats."""
    x_u, y_u = ecdf(u_vals)
    x_p, y_p = ecdf(p_vals)

    plt.figure(figsize=(9, 6))
    plt.step(x_u, y_u, where="post", label="Uniform means", linewidth=2)
    plt.step(x_p, y_p, where="post", label="Power-law means", linewidth=2)
    plt.xlabel("Sample Mean")
    plt.ylabel("Empirical CDF")
    plt.title(title)
    plt.legend()
    # Annotate stats
    txt = f"KS D = {D:.4f}\nKS p-value = {pval:.4g}"
    plt.gca().text(0.02, 0.02, txt, transform=plt.gca().transAxes,
                   ha="left", va="bottom",
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    plt.grid(alpha=0.25)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="KS test comparing Uniform vs Power-law sample-mean distributions.")
    parser.add_argument("--base", type=str, default=None,
                        help="Project base directory (defaults to this script's folder).")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for reject/accept flag.")
    args = parser.parse_args()

    base = Path(args.base).resolve() if args.base else Path(__file__).resolve().parent

    mean_uniform = base / "data" / "uniform"   / "mean_distribution"
    mean_power   = base / "data" / "power_law" / "mean_distribution"
    out_dir      = base / "data" / "ks_test"

    u_dict = load_means_dir(mean_uniform)
    p_dict = load_means_dir(mean_power)

    # Match pairs by (replicates, samples_per_replicate)
    common_keys = sorted(set(u_dict.keys()) & set(p_dict.keys()))
    if not common_keys:
        print(f"[WARN] No matching (replicates, samples) pairs between {mean_uniform} and {mean_power}.")
        return

    rows = []
    for (reps, samps) in common_keys:
        u_vals = u_dict[(reps, samps)]
        p_vals = p_dict[(reps, samps)]
        res = ks_2samp(u_vals, p_vals, alternative="two-sided", mode="auto")
        D, pval = float(res.statistic), float(res.pvalue)
        reject = pval < args.alpha

        # Save ECDF comparison plot
        title = f"KS ECDF — Uniform vs Power-law (means), {reps}x{samps}"
        out_png = out_dir / f"ks_ecdf_{reps}x{samps}.png"
        plot_ecdf_compare(u_vals, p_vals, title=title, out_png=out_png, D=D, pval=pval)
        print(f"[OK] {reps}x{samps}: D={D:.4f}, p={pval:.4g}, reject@{args.alpha}={reject} -> {out_png}")

        rows.append({
            "replicates": reps,
            "samples_per_replicate": samps,
            "n_uniform": len(u_vals),
            "n_power": len(p_vals),
            "ks_D": D,
            "ks_pvalue": pval,
            f"reject@alpha={args.alpha}": reject
        })

    # Write summary CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "ks_results.csv"
    pd.DataFrame(rows).sort_values(["samples_per_replicate", "replicates"]).to_csv(summary_csv, index=False)
    print(f"[OK] Wrote KS summary -> {summary_csv}")

    # Guidance printout explaining CLT expectations (for your report)
    print("\n--- Notes for discussion ---")
    print("* CLT says each distribution of sample means individually tends to Normal as the per-replicate sample size grows.")
    print("* The KS test compares the two sample-mean distributions to each other—not to Normal.")
    print("* If the two parent distributions have the SAME mean (e.g., Uniform[0.25,1.25] and Power(a=3) on [0,1]),")
    print("  the two sample-mean distributions may still differ in variance/skew; KS power increases with the number of means (replicates).")
    print("* If the parent means differ (e.g., Pareto-based power-law), KS will typically reject even for modest replicates.")
    print("* As replicates increase (10 → 100 → 1000), KS becomes more sensitive (higher power) to any persistent differences.")

if __name__ == "__main__":
    main()
