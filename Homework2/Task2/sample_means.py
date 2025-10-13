# sample_means.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import re

def write_means_and_plot(in_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(in_dir.glob("*_replicates_*.csv"))

    if not csv_files:
        print(f"[WARN] No replicate CSVs found in {in_dir}")
        return

    for f in csv_files:
        # Load and compute per-replicate means (columns = Replicate_1, Replicate_2, ...)
        df = pd.read_csv(f)
        num_df = df.select_dtypes(include="number")
        means = num_df.mean(axis=0)  # mean of each replicate (down the rows)

        # Save means CSV (replicates -> means)
        out_csv = f.name.replace("replicates", "means")
        out_csv_path = out_dir / out_csv
        means.to_csv(out_csv_path, header=["Sample Mean"], index_label="Replicate")
        print(f"[OK] Computed sample means for {f} -> {out_csv_path}")

        # Detect replicate/sample-size from filename, fallback to column count
        reps, samps = None, None
        m = re.search(r"(\d+)x(\d+)", f.name)
        if m:
            reps = int(m.group(1))
            samps = int(m.group(2))
        else:
            reps = len(num_df.columns)

        use_box = (reps == 10 and samps == 100)

        plot_title = f"Distribution of Sample Means: {f.name}"
        base_out_name = out_csv_path.with_suffix("").name

        if use_box:
            # ---- Box & whisker plot for 10x100 ----
            plt.figure(figsize=(8, 5))
            plt.boxplot(means.values, vert=True, showmeans=True)
            plt.title(plot_title + " (Boxplot)")
            plt.ylabel("Sample Mean")
            plt.tight_layout()
            out_png_path = out_dir / (base_out_name + "_box.png")
            plt.savefig(out_png_path, dpi=120, bbox_inches="tight")
            plt.close()
            print(f"[OK] Saved BOX plot for {out_csv_path} -> {out_png_path}")
        else:
            # ---- Histogram for all other sizes ----
            plt.figure(figsize=(10, 6))
            plt.hist(means.values, bins=30, edgecolor="black", alpha=0.7)
            plt.title(plot_title)
            plt.xlabel("Sample Mean")
            plt.ylabel("Frequency")
            plt.tight_layout()
            out_png_path = out_dir / (base_out_name + "_hist.png")
            plt.savefig(out_png_path, dpi=120, bbox_inches="tight")
            plt.close()
            print(f"[OK] Saved histogram plot for {out_csv_path} -> {out_png_path}")

def main():
    base = Path(__file__).resolve().parent

    in_uniform  = base / "data" / "uniform"    / "sample_distribution"
    in_power    = base / "data" / "power_law"  / "sample_distribution"

    out_uniform = base / "data" / "uniform"    / "mean_distribution"
    out_power   = base / "data" / "power_law"  / "mean_distribution"

    write_means_and_plot(in_uniform, out_uniform)
    write_means_and_plot(in_power,   out_power)

if __name__ == "__main__":
    main()
