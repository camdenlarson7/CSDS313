import random
from statistics import mean, variance
import matplotlib.pyplot as plt

# Avoid name collision with function arg "M"
M_values = [100, 1000, 10000]
N100   = [5, 10, 20, 25, 50, 75]
N1000  = [50, 100, 200, 250, 500, 750]
N10000 = [500, 1000, 2000, 2500, 5000, 7500]

def serial_list(M: int, N: int):
    """
    Sample N distinct serial numbers from {1, ..., M} (without replacement).
    Returns a sorted list.
    """
    if N > M:
        raise ValueError("N cannot exceed M when sampling without replacement.")
    serials = random.sample(range(1, M + 1), N)  # unique by construction
    serials.sort()
    return serials

# ----- Estimators -----
def est_mle(sample):
    """ MLE: max observed ID """
    return max(sample)

def est_mean(sample):
    """ Mean-based estimator: 2*Xbar - 1 """
    n = len(sample)
    return 2 * (sum(sample) / n) - 1

def est_mvu(sample):
    """ MVU (UMVU) estimator: ((n+1)/n)*max - 1 """
    n = len(sample)
    return ((n + 1) / n) * max(sample) - 1

# helper: safe mean/variance
def _mean_var(values):
    m = mean(values)
    v = variance(values) if len(values) > 1 else 0.0  # sample variance (ddof=1)
    return m, v

def simulate_estimators(M: int, n: int, reps: int = 2000, seed: int | None = None):
    """
    Run `reps` simulations of size n from {1,...,M} (without replacement),
    compute MLE, MEAN, MVU each time, and return empirical mean & variance.
    Variance is sample variance (n-1 in denominator).
    """
    if seed is not None:
        random.seed(seed)

    mle_vals, mean_vals, mvu_vals = [], [], []

    for _ in range(reps):
        s = serial_list(M, n)
        mle_vals.append(est_mle(s))
        mean_vals.append(est_mean(s))
        mvu_vals.append(est_mvu(s))

    mle_mean, mle_var   = _mean_var(mle_vals)
    mean_mean, mean_var = _mean_var(mean_vals)
    mvu_mean, mvu_var   = _mean_var(mvu_vals)

    return {
        "MLE":  {"mean": mle_mean,  "var": mle_var},
        "MEAN": {"mean": mean_mean, "var": mean_var},
        "MVU":  {"mean": mvu_mean,  "var": mvu_var},
        # raw values if you want to plot distributions later:
        "samples": {
            "MLE": mle_vals, "MEAN": mean_vals, "MVU": mvu_vals
        }
    }

def simulate_over_n(M: int, n_list: list[int], reps: int = 2000, seed: int | None = None):
    """
    Run simulations for a fixed M across multiple n values.
    Returns a dict mapping n -> summary stats for each estimator.
    """
    results = {}
    for n in n_list:
        results[n] = simulate_estimators(M, n, reps=reps, seed=seed)
    return results

# -------- Example usage (uncomment to run) --------
res_100  = simulate_over_n(100,  N100,   reps=5000, seed=42)
res_1000 = simulate_over_n(1000, N1000,  reps=4000, seed=42)
res_1e4  = simulate_over_n(10000, N10000, reps=3000, seed=42)
# 
for n, stats in res_100.items():
    print(f"M=100, n={n} -> "
          f"MLE(mean={stats['MLE']['mean']:.2f}, var={stats['MLE']['var']:.2f}); "
          f"MEAN(mean={stats['MEAN']['mean']:.2f}, var={stats['MEAN']['var']:.2f}); "
          f"MVU(mean={stats['MVU']['mean']:.2f}, var={stats['MVU']['var']:.2f})")

def plot_mean_var_for_M(M: int, n_list: list[int], reps: int = 2000, seed: int | None = 42,
                        save: bool = True, show: bool = True):
    """
    For a fixed M, simulate across n_list and plot:
      1) Empirical mean of each estimator vs n
      2) Empirical variance of each estimator vs n
    Creates two separate figures for this M (no subplots).
    """
    results = simulate_over_n(M, n_list, reps=reps, seed=seed)
    ns = sorted(n_list)

    # Collect series for each estimator
    est_names = ["MLE", "MEAN", "MVU"]
    means_by_est = {e: [results[n][e]["mean"] for n in ns] for e in est_names}
    print(means_by_est)
    vars_by_est  = {e: [results[n][e]["var"]  for n in ns] for e in est_names}
    print(vars_by_est)

    # --- Plot: Empirical mean vs n ---
    plt.figure()
    for e in est_names:
        plt.plot(ns, means_by_est[e], marker='o', label=e)
    plt.title(f"German Tank — Empirical Mean vs n (M = {M})")
    plt.xlabel("n (sample size)")
    plt.ylabel("Empirical mean of estimator")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save:
        plt.savefig(f"mean_vs_n_M{M}.png", dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    # --- Plot: Empirical variance vs n ---
    plt.figure()
    for e in est_names:
        plt.plot(ns, vars_by_est[e], marker='o', label=e)
    plt.title(f"German Tank — Empirical Variance vs n (M = {M})")
    plt.xlabel("n (sample size)")
    plt.ylabel("Empirical variance of estimator")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save:
        plt.savefig(f"variance_vs_n_M{M}.png", dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

# ----- Run for each M with your n-sets -----
# (Assuming you defined N100, N1000, N10000 earlier.)
plot_mean_var_for_M(100,   N100,   reps=3000, seed=42)
plot_mean_var_for_M(1000,  N1000,  reps=3000, seed=42)
plot_mean_var_for_M(10000, N10000, reps=3000, seed=42)