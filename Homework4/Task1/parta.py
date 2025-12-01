import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

file = "..\datasets-1\datasets\congress\p1_congress_1984_votes.csv"
df = pd.read_csv(file, header=None)

X = df.iloc[:, :16].to_numpy(dtype=float)
X = np.nan_to_num(X, nan=0.0)

pca = PCA(n_components=X.shape[1], svd_solver="full", random_state=42)
pca.fit(X)

explained = pca.explained_variance_ratio_    
cum_explained = np.cumsum(explained)             

k = np.arange(1, X.shape[1] + 1)

plt.figure(figsize=(8, 5))
plt.plot(k, cum_explained, marker="o")
plt.xticks(k)
plt.ylim(0, 1.01)
plt.xlabel("k (number of principal components)")
plt.ylabel("Cumulative variance explained")
plt.title("PCA: Cumulative Variance Explained vs. k")
plt.grid(True, alpha=0.3)
plt.axhline(0.90, linestyle="--")
plt.text(1, 0.905, "90%", va="bottom")
plt.tight_layout()
plt.savefig("graphs/pca_cumulative_variance_explained.png", dpi=120, bbox_inches="tight")

party_affiliations = "../datasets-1/datasets/congress/p1_congress_1984_party_affiliations.csv"
party_df = pd.read_csv(party_affiliations, header=None)
merged_df = pd.concat([df, party_df], axis=1)
merged_df.to_csv("congress_1984_votes_with_parties.csv", index=False, header=False)
print(merged_df.head())

X = merged_df.iloc[:, :16].to_numpy(dtype=float)
X = np.nan_to_num(X, nan=0.0)
party = merged_df.iloc[:, 16].astype(str).str.strip()

pca = PCA(n_components=3, svd_solver="full", random_state=42)
Z = pca.fit_transform(X)  

def scatter_by_party(ax, x, y, xlabel, ylabel, title):
    parties = sorted(party.unique())
    for p in parties:
        mask = (party == p).to_numpy()
        ax.scatter(Z[mask, x], Z[mask, y], label=p, alpha=0.7, s=30)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Party", fontsize=9, title_fontsize=10)

# PC pair plots: (1,2), (1,3), (2,3)
fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

scatter_by_party(axes[0], 0, 1, "PC1", "PC2", "PC1 vs PC2")
scatter_by_party(axes[1], 0, 2, "PC1", "PC3", "PC1 vs PC3")
scatter_by_party(axes[2], 1, 2, "PC2", "PC3", "PC2 vs PC3")
plt.savefig("graphs/pca_pc_pair_plots.png", dpi=120, bbox_inches="tight")


print("Explained variance ratios:", pca.explained_variance_ratio_)
print("Cumulative (first 3 PCs):", np.cumsum(pca.explained_variance_ratio_))
