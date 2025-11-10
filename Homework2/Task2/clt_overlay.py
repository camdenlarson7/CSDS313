# clt_overlay.py
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import argparse

def parse_reps_samps_from_name(name: str):
    m = re.search(r"(\d+)x(\d+)", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

def hist_with_normal_overlay(values: np.ndarray, title: str, out_png: Path, bins: int = 30):
    """Plot histogram (density) and overlay Normal(mu_hat, sigma_hat) PDF. Save to out_png."""
    mu_hat = float(np.mean(values))
    sigma_hat = float(np.std(values, ddof=1))

    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=bins, edgecolor="black", alpha=0.7, density=True)
    plt.title(title)
    plt.xlabel("Sample Mean")
    plt.ylabel("Density")

    # x-grid for PDF overlay
    x_min, x_max = np.min(values), np.max(values)
    pad = 0.25 * (x_max - x_min + 1e-12)
    x = np.linspace(x_min - pad, x_max + pad, 600)

    if sigma_hat > 0:
        pdf = (1.0 / (sigma_hat * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu_hat) / sigma_hat) ** 2)
        plt.plot(x, pdf, linewidth=2, label=f"Normal($\\mu$={mu_hat:.4f}, $\\sigma$={sigma_hat:.4f})")
        plt.legend()

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()
    return mu_hat, sigma_hat

def boxplot_for_values(values: np.ndarray, title: str, out_png: Path):
    plt.figure(figsize=(8, 5))
    plt.boxplot(values, vert=True, showmeans=True)
    plt.title(title + " (Boxplot)")
    plt.ylabel("Sample Mean")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()

def process_mean_dir(mean_dir: Path, summary_rows: list):
    """
    For each *_means_*.csv in mean_dir:
      - compute empirical mu/sigma
      - save histogram+Normal overlay to *_hist_pdf.png
      - if it's 10x100, also save a boxplot to *_box.png
    Append stats into summary_rows.
    """
    csv_files = sorted(mean_dir.glob("*_means_*.csv"))
    if not csv_files:
        print(f"[WARN] No means CSVs found in {mean_dir}")
        return

    for f in csv_files:
        df = pd.read_csv(f)
        # Accept either a single 'Sample Mean' column or any numeric columns
        if "Sample Mean" in df.columns:
            values = df["Sample Mean"].to_numpy(dtype=float)
        else:
            values = df.select_dtypes(include="number").to_numpy(dtype=float).squeeze()
        if values.ndim > 1:
            values = values.ravel()

        reps, samps = parse_reps_samps_from_name(f.name)
        base = f.with_suffix("").name  # strip .csv
        # Histogram + Normal overlay
        overlay_png = mean_dir / (base + "_hist_pdf.png")
        mu_hat, sigma_hat = hist_with_normal_overlay(
            values,
            title=f"Distribution of Sample Means (+ Normal Overlay): {f.name}",
            out_png=overlay_png,
            bins=30
        )
        print(f"[OK] CLT overlay -> {overlay_png}")

        # Optional boxplot for 10x100
        if reps == 10 and samps == 100:
            box_png = mean_dir / (base + "_box.png")
            boxplot_for_values(values, title=f"Distribution of Sample Means: {f.name}", out_png=box_png)
            print(f"[OK] Boxplot -> {box_png}")

        summary_rows.append({
            "dir": str(mean_dir),
            "file": f.name,
            "replicates": reps,
            "samples_per_replicate": samps,
            "empirical_mu": mu_hat,
            "empirical_sigma": sigma_hat
        })

def main():
    parser = argparse.ArgumentParser(description="CLT overlays for sample-mean distributions.")
    parser.add_argument("--base", type=str, default=None,
                        help="Base directory (defaults to script folder).")
    args = parser.parse_args()

    base = Path(args.base).resolve() if args.base else Path(__file__).resolve().parent

    # Mean CSV locations produced by your earlier script(s)
    mean_uniform = base / "data" / "uniform"   / "mean_distribution"
    mean_power   = base / "data" / "power_law" / "mean_distribution"

    summary_rows = []
    process_mean_dir(mean_uniform, summary_rows)
    process_mean_dir(mean_power,   summary_rows)

    # Save a combined summary of empirical parameters
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        out_summary = base / "data" / "clt_empirical_params_summary.csv"
        out_summary.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_summary, index=False)
        print(f"[OK] Wrote empirical CLT parameter summary -> {out_summary}")

if __name__ == "__main__":
    main()
