# Task1/src/metrics.py
import numpy as np
from typing import Tuple, Optional, Callable
from scipy.stats import chi2_contingency

# Contingency table
def contingency_from_binary(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Build 2x2 contingency table for two binary vectors x, y (values must be 0 or 1).
    Returns:
        counts: np.ndarray shape (2, 2)
                [[n00, n01],
                 [n10, n11]]
    """
    x = np.asarray(x).astype(int)
    y = np.asarray(y).astype(int)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    if not set(np.unique(x)).issubset({0, 1}) or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("x and y must be binary (0/1).")

    n00 = int(np.sum((x == 0) & (y == 0)))
    n01 = int(np.sum((x == 0) & (y == 1)))
    n10 = int(np.sum((x == 1) & (y == 0)))
    n11 = int(np.sum((x == 1) & (y == 1)))
    return np.array([[n00, n01],
                     [n10, n11]], dtype=float)


# Mutual Information
def mi_from_counts(counts: np.ndarray, base: float = 2.0) -> float:
    """
    Mutual Information from a contingency table (2x2 or larger).
    Skips zero cells by convention.
    Returns MI in 'bits' if base=2, 'nats' if base=np.e.
    """
    counts = np.asarray(counts, dtype=float)
    N = counts.sum()
    row = counts.sum(axis=1, keepdims=True)
    col = counts.sum(axis=0, keepdims=True)

    # Avoid divide-by-zero warnings, zeros contribute 0 to the sum
    with np.errstate(divide='ignore', invalid='ignore'):
        p = counts / N
        ratio = (counts * N) / (row * col)
        # Replace invalids for zero cells with 1 (log(1)=0 -> no contribution)
        ratio = np.where(counts > 0, ratio, 1.0)
        log_ratio = np.log(ratio) / np.log(base)
        mi = np.nansum(p * log_ratio)
    return float(mi)

# Jaccard Index
def jaccard_index_from_counts(counts: np.ndarray) -> float:
    """
    Jaccard for two binary variables: intersection/union where '1' is the 'presence'.
    For a 2x2 table:
        counts = [[n00, n01],
                  [n10, n11]]
    Intersection = n11, Union = n01 + n10 + n11
    """
    n00, n01 = counts[0, 0], counts[0, 1]
    n10, n11 = counts[1, 0], counts[1, 1]
    inter = n11
    union = n01 + n10 + n11
    return 0.0 if union == 0 else float(inter / union)

# Pearson's Chi-square (parametric p-value)
def chi_square_from_counts(counts: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    Wrap scipy.stats.chi2_contingency (no Yates correction).
    Returns: chi2_stat, p_value, expected
    """
    chi2, p, dof, expected = chi2_contingency(counts, correction=False)
    return float(chi2), float(p), expected

# Generic permutation test
def permutation_test(
    stat_fn: Callable[[np.ndarray, np.ndarray], float],
    x: np.ndarray,
    y: np.ndarray,
    n_perms: int = 10000,
    rng: Optional[np.random.Generator] = None,
    alternative: str = "greater"
) -> Tuple[float, float, np.ndarray]:
    """
    Compute observed statistic and a permutation p-value.
    alternative in {"greater","less","two-sided"} applies to how 'more significant' is defined.

    Returns:
        obs_stat, p_value, perm_stats (array of length n_perms)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    obs = stat_fn(x, y)
    perm_stats = np.empty(n_perms, dtype=float)

    # Permute one variable (break association, preserve marginals)
    for i in range(n_perms):
        y_perm = np.array(y, copy=True)
        rng.shuffle(y_perm)
        perm_stats[i] = stat_fn(x, y_perm)

    # Compute permutation p-value with +1 correction
    if alternative == "greater":
        c = int(np.sum(perm_stats >= obs))
    elif alternative == "less":
        c = int(np.sum(perm_stats <= obs))
    elif alternative == "two-sided":
        # Two-sided via absolute deviation from the perm mean
        center = np.mean(perm_stats)
        c = int(np.sum(np.abs(perm_stats - center) >= np.abs(obs - center)))
    else:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'.")

    p_val = (c + 1) / (n_perms + 1)
    return float(obs), float(p_val), perm_stats

# Convenience wrappers using counts
def mi_stat_from_vectors(x: np.ndarray, y: np.ndarray) -> float:
    counts = contingency_from_binary(x, y)
    return mi_from_counts(counts, base=2.0)

def jaccard_stat_from_vectors(x: np.ndarray, y: np.ndarray) -> float:
    counts = contingency_from_binary(x, y)
    return jaccard_index_from_counts(counts)

