# save as: partB_congress_clustering_and_permtest.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ============================================================
# Config
# ============================================================
CSV_PATH = "congress_1984_votes_with_parties.csv"  # same folder as this script
OUT_DIR = "graphs"
os.makedirs(OUT_DIR, exist_ok=True)

K = 2
SEED = 42
marker_map = {"Democrat": "o", "Republican": "s"}

# Permutation test settings
B = 500          # increase if you can (e.g., 1000+)
N_INIT = 20      # k-means restarts to reduce randomness
MAX_ITERS = 300
TOL = 1e-6

# ============================================================
# Load data (16 vote columns + party label)
# ============================================================
df = pd.read_csv(CSV_PATH, header=None)
X = df.iloc[:, 0:16].to_numpy(dtype=float)
X = np.nan_to_num(X, nan=0.0)
party = df.iloc[:, 16].astype(str).str.strip()

# ============================================================
# K-means from scratch (squared Euclidean distance)
# ============================================================
def squared_euclidean_distances(A, B):
    """Return D[i,j] = ||A[i] - B[j]||^2 for A (nxd), B (kxd)."""
    A2 = np.sum(A * A, axis=1, keepdims=True)        # (n, 1)
    B2 = np.sum(B * B, axis=1, keepdims=True).T      # (1, k)
    return A2 + B2 - 2 * (A @ B.T)                   # (n, k)

def kmeans_once(Xdata, k=2, max_iters=200, tol=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    n, d = Xdata.shape

    init_idx = rng.choice(n, size=k, replace=False)
    centroids = Xdata[init_idx].copy()
    labels = np.full(n, -1, dtype=int)

    for _ in range(max_iters):
        D = squared_euclidean_distances(Xdata, centroids)
        new_labels = np.argmin(D, axis=1)

        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels

        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            pts = Xdata[labels == j]
            if len(pts) == 0:
                new_centroids[j] = Xdata[rng.integers(0, n)]  # handle empty cluster
            else:
                new_centroids[j] = pts.mean(axis=0)

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            break

    D_final = squared_euclidean_distances(Xdata, centroids)
    sse = float(np.sum(D_final[np.arange(n), labels]))
    return centroids, labels, sse

def kmeans_best_of_n(Xdata, k=2, n_init=10, base_seed=0, max_iters=200, tol=1e-6):
    best = None
    for r in range(n_init):
        c, lab, sse = kmeans_once(Xdata, k=k, max_iters=max_iters, tol=tol, seed=base_seed + r)
        if best is None or sse < best[2]:
            best = (c, lab, sse)
    return best  # (centroids, labels, sse)

# ============================================================
# Part B1: Cluster in 16-D, then visualize on PC1/PC2
# ============================================================
centroids_16d, labels, sse_obs = kmeans_best_of_n(
    X, k=K, n_init=N_INIT, base_seed=SEED, max_iters=MAX_ITERS, tol=TOL
)

print("=== Part B: Clustering on 16-D votes ===")
print(f"K-means SSE (16-D, best of {N_INIT} runs): {sse_obs:.3f}")

# Agreement with party (best label swap)
y_true = (party == "Republican").astype(int).to_numpy()  # Rep=1, Dem=0
acc_same = (labels == y_true).mean()
acc_swap = ((1 - labels) == y_true).mean()

if acc_swap > acc_same:
    mapped = 1 - labels
    mapping_note = "SWAPPED mapping (cluster labels flipped)"
else:
    mapped = labels
    mapping_note = "DIRECT mapping (cluster labels unchanged)"

best_acc = max(acc_same, acc_swap)
pred_party = np.where(mapped == 1, "Republican", "Democrat")

dem_mask = (party == "Democrat").to_numpy()
rep_mask = (party == "Republican").to_numpy()
dem_acc = (pred_party[dem_mask] == "Democrat").mean() if dem_mask.any() else float("nan")
rep_acc = (pred_party[rep_mask] == "Republican").mean() if rep_mask.any() else float("nan")

print(f"Overall agreement accuracy (best mapping): {best_acc:.3f} | {mapping_note}")
print(f"Democrat accuracy (within actual Dems):   {dem_acc:.3f}  ({dem_mask.sum()} Dems)")
print(f"Republican accuracy (within actual Reps): {rep_acc:.3f}  ({rep_mask.sum()} Reps)")

ct = pd.crosstab(party, labels, rownames=["Party"], colnames=["Cluster"])
print("\nParty vs Cluster table:\n", ct)

def mutual_information(x, y):
    """
    MI(X;Y) in nats using empirical probabilities.
    x, y are 1D arrays of discrete labels (ints/strings ok).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)

    # map labels to 0..k-1
    x_vals, x_inv = np.unique(x, return_inverse=True)
    y_vals, y_inv = np.unique(y, return_inverse=True)

    kx = len(x_vals)
    ky = len(y_vals)

    # joint counts
    joint = np.zeros((kx, ky), dtype=float)
    for i in range(n):
        joint[x_inv[i], y_inv[i]] += 1.0

    pxy = joint / n
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)

    mi = 0.0
    for i in range(kx):
        for j in range(ky):
            if pxy[i, j] > 0:
                mi += pxy[i, j] * np.log(pxy[i, j] / (px[i, 0] * py[0, j]))
    return float(mi)

def entropy(x):
    """H(X) in nats."""
    x = np.asarray(x)
    _, inv = np.unique(x, return_inverse=True)
    n = len(x)
    counts = np.bincount(inv).astype(float)
    p = counts / n
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))

MI = mutual_information(labels, party)
HX = entropy(labels)
HY = entropy(party)

# Normalized MI (common choice): MI / sqrt(HX*HY)
NMI = MI / np.sqrt(HX * HY) if HX > 0 and HY > 0 else float("nan")

print("\n=== Mutual Information Agreement ===")
print(f"MI(Cluster; Party) = {MI:.6f} nats")
print(f"NMI = {NMI:.6f} (0=no agreement, 1=perfect agreement)")

# Fit PCA only for visualization (on ORIGINAL votes)
pca_vis = PCA(n_components=2, svd_solver="full", random_state=SEED)
Z = pca_vis.fit_transform(X)

centroids_pc, labels_pc, sse_pc = kmeans_best_of_n(
    Z, k=K, n_init=N_INIT, base_seed=SEED + 2222, max_iters=MAX_ITERS, tol=TOL
)

print("\n=== Part B: Clustering on (PC1, PC2) only ===")
print(f"K-means SSE (PC1/PC2, best of {N_INIT} runs): {sse_pc:.3f}")

# Agreement with party (best label swap)
acc_same_pc = (labels_pc == y_true).mean()
acc_swap_pc = ((1 - labels_pc) == y_true).mean()

if acc_swap_pc > acc_same_pc:
    mapped_pc = 1 - labels_pc
    mapping_note_pc = "SWAPPED mapping (cluster labels flipped)"
else:
    mapped_pc = labels_pc
    mapping_note_pc = "DIRECT mapping (cluster labels unchanged)"

best_acc_pc = max(acc_same_pc, acc_swap_pc)
pred_party_pc = np.where(mapped_pc == 1, "Republican", "Democrat")

dem_acc_pc = (pred_party_pc[dem_mask] == "Democrat").mean() if dem_mask.any() else float("nan")
rep_acc_pc = (pred_party_pc[rep_mask] == "Republican").mean() if rep_mask.any() else float("nan")

print(f"Overall agreement accuracy (best mapping): {best_acc_pc:.3f} | {mapping_note_pc}")
print(f"Democrat accuracy (within actual Dems):   {dem_acc_pc:.3f}  ({dem_mask.sum()} Dems)")
print(f"Republican accuracy (within actual Reps): {rep_acc_pc:.3f}  ({rep_mask.sum()} Reps)")

ct_pc = pd.crosstab(party, labels_pc, rownames=["Party"], colnames=["Cluster"])
print("\nParty vs Cluster table (PC1/PC2 clustering):\n", ct_pc)

MI_pc = mutual_information(labels_pc, party)
HX_pc = entropy(labels_pc)
# HY is the same party entropy as before
NMI_pc = MI_pc / np.sqrt(HX_pc * HY) if HX_pc > 0 and HY > 0 else float("nan")

print("\n=== Mutual Information Agreement (PC1/PC2 clustering) ===")
print(f"MI(Cluster; Party) = {MI_pc:.6f} nats")
print(f"NMI = {NMI_pc:.6f} (0=no agreement, 1=perfect agreement)")

# Plot clusters found in PC space
fig, ax = plt.subplots(figsize=(8, 6))
for p in ["Democrat", "Republican"]:
    mask = (party == p).to_numpy()
    ax.scatter(
        Z[mask, 0], Z[mask, 1],
        c=labels_pc[mask],
        cmap="viridis",
        marker=marker_map[p],
        s=40,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.4,
        label=p
    )

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("K-means using only (PC1, PC2)\nColor = Cluster, Marker = Party")
ax.grid(True, alpha=0.3)
ax.legend(title="Party (marker)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "partB_clusters_using_PC1_PC2.png"), dpi=140, bbox_inches="tight")
plt.close(fig)

# ============================================================
# Comparison summary
# ============================================================
print("\n=== Comparison: 16-D votes vs PC1/PC2 ===")
print(f"Accuracy (16-D):     {best_acc:.3f} | NMI (16-D):     {NMI:.6f}")
print(f"Accuracy (PC1/PC2):  {best_acc_pc:.3f} | NMI (PC1/PC2):  {NMI_pc:.6f}")

if NMI_pc > NMI:
    winner = "PC1/PC2"
elif NMI_pc < NMI:
    winner = "16-D votes"
else:
    winner = "tie"

print(f"Better agreement (by NMI): {winner}")
print("Comment: PC1/PC2 keeps the two largest-variance directions and can denoise minor/noisy issue variation,")
print("but it also discards information in lower-variance PCs that might still correlate with party.")
print("So agreement can improve (denoising) or worsen (information loss) depending on the dataset.")



# Plot: color = cluster, marker = party
marker_map = {"Democrat": "o", "Republican": "s"}

fig, ax = plt.subplots(figsize=(8, 6))
for p in ["Democrat", "Republican"]:
    mask = (party == p).to_numpy()
    ax.scatter(
        Z[mask, 0], Z[mask, 1],
        c=labels[mask],
        cmap="viridis",
        marker=marker_map[p],
        s=40,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.4,
        label=p
    )

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Clusters from 16-D K-means, Visualized on PC1-PC2\nColor = Cluster, Marker = Party")
ax.grid(True, alpha=0.3)
ax.legend(title="Party (marker)")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "partB_clusters_on_pc1_pc2.png"), dpi=140, bbox_inches="tight")
plt.close(fig)

print("\nPCA (for visualization) explained variance ratios:", pca_vis.explained_variance_ratio_)
print("PCA cumulative explained (PC1+PC2):", float(np.sum(pca_vis.explained_variance_ratio_)))

# ============================================================
# Part B2: Permutation test (cluster on permuted 16-D votes)
# ============================================================
def permute_votes_rowwise(X_votes, rng):
    """Within-row shuffle: preserves each member's -1/0/1 distribution."""
    Xp = X_votes.copy()
    for i in range(Xp.shape[0]):
        Xp[i] = Xp[i, rng.permutation(Xp.shape[1])]
    return Xp

sse_null = np.zeros(B, dtype=float)
rng_master = np.random.default_rng(SEED)

for b in range(B):
    rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
    X_perm = permute_votes_rowwise(X, rng)

    # cluster permuted dataset with SAME algorithm/settings
    _, _, sse_b = kmeans_best_of_n(
        X_perm, k=K, n_init=N_INIT, base_seed=SEED + 10_000 + b, max_iters=MAX_ITERS, tol=TOL
    )
    sse_null[b] = sse_b

# p-value for "more clustered than random": smaller SSE is more extreme
p_value = (1 + np.sum(sse_null <= sse_obs)) / (B + 1)

print("\n=== Permutation Test (16-D votes) ===")
print(f"Observed SSE:   {sse_obs:.3f}")
print(f"Null mean SSE:  {sse_null.mean():.3f}")
print(f"Null std SSE:   {sse_null.std(ddof=1):.3f}")
print(f"p-value (SSE_perm <= SSE_obs): {p_value:.4f}")

# Plot null distribution
plt.figure(figsize=(8, 5))
plt.hist(sse_null, bins=30, alpha=0.85)
plt.axvline(sse_obs, linewidth=2)
plt.title(f"Permutation Test for Clustering (Score = K-means SSE on 16-D Votes)\nB={B}, p={p_value:.4f}")
plt.xlabel("SSE (lower = better clustering)")
plt.ylabel("Count")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "partB_permutation_test_sse_hist.png"), dpi=140, bbox_inches="tight")
plt.show()

# Save scores for writeup
np.savetxt(os.path.join(OUT_DIR, "partB_permutation_sse_scores.txt"), sse_null)
