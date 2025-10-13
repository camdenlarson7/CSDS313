from scipy.stats.qmc import Sobol
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path

# ---------- Helpers ----------
def slugify(name: str) -> str:
    """Make a filesystem-friendly filename stem."""
    return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in name)

def uniform_sample(start: float, end: float, n: int) -> list[float]:
    """Generate n samples from a uniform distribution [start, end) using Sobol sequence."""
    pow2 = 2 ** int(np.ceil(np.log2(n)))        # next power of 2 (Sobol works best in powers of 2)
    sampler = Sobol(d=1, scramble=True)
    samples = sampler.random(pow2).flatten()[:n]
    scaled = start + (end - start) * samples
    return scaled.tolist()

def powerlaw_sample(a: float, size: int) -> list[float]:
    """
    Generate samples from a heavy-tailed distribution.
    NOTE: Assignment suggests numpy.random.power (a "power" distribution). If you must match that exactly,
    replace the body with: return np.random.power(a, size).tolist()
    Below uses Pareto (a common "power-law" choice): X ~ Pareto(a) shifted to have minimum 1.
    """
    return (np.random.pareto(a, size) + 1).tolist()

def collect_replicates(distribution_func, dist_args, num_replicates: int, sample_size: int) -> list[list[float]]:
    """Collect num_replicates datasets, each with sample_size samples from the given distribution."""
    return [distribution_func(*dist_args, sample_size) for _ in range(num_replicates)]

def plot_sample_distribution(samples, title: str, out_dir: Path, bins: int = 20):
    """Save a histogram plot to out_dir with a sane filename."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(samples, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    fname = slugify(title) + ".png"
    plt.savefig(out_dir / fname, dpi=120, bbox_inches="tight")
    plt.close()

def save_replicates_to_csv(replicates, dist_name: str, num_replicates: int, sample_size: int, out_dir: Path):
    """Save replicates as columns in a CSV file under out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{slugify(dist_name)}_replicates_{num_replicates}x{sample_size}.csv"
    with open(out_dir / fname, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f"Replicate_{i+1}" for i in range(num_replicates)])
        # Transpose replicates so each row = i-th sample across all replicates
        for row in zip(*replicates):
            writer.writerow(row)

def main():
    # Settings
    uniform_args = (0.25, 1.25)  # start, end
    powerlaw_args = (2.3,)       # shape parameter for Pareto ("power-law"-like). Use 3 if matching assignment alpha=3.
    replicate_sizes = [10, 100, 1000]
    sample_size = 100

    # Base path is directory of this script
    base_dir = Path(__file__).resolve().parent

    # Dist-specific output folders
    out_uniform = base_dir / "data" / "uniform" / "sample_distribution"
    out_power   = base_dir / "data" / "power_law" / "sample_distribution"

    # Generate and save for each distribution and replicate size
    configs = [
        ("Uniform",   uniform_sample, uniform_args, out_uniform),
        ("Power-law", powerlaw_sample, powerlaw_args, out_power),
    ]

    for dist_name, dist_func, dist_args, out_dir in configs:
        for num_replicates in replicate_sizes:
            replicates = collect_replicates(dist_func, dist_args, num_replicates, sample_size)
            print(f"{dist_name} -> {num_replicates} replicates of {sample_size} samples each "
                  f"(saving to {out_dir}).")

            # Plot the first replicate
            if replicates:
                title = f"{dist_name} Sample (Replicate 1, {num_replicates}x{sample_size})"
                plot_sample_distribution(replicates[0], title, out_dir=out_dir, bins=20)

            # Save all replicates to CSV
            save_replicates_to_csv(replicates, dist_name, num_replicates, sample_size, out_dir=out_dir)

if __name__ == "__main__":
    main()
