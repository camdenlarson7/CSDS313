import os
import pandas as pd
import matplotlib.pyplot as plt

WHITE_FILE = "data/winequality-white.csv"
RED_FILE   = "data/winequality-red.csv"
OUT_DIR    = "graphs"
THRESHOLD  = 6  # quality >= 6 => good (1), else bad (0)

os.makedirs(OUT_DIR, exist_ok=True)

white_df = pd.read_csv(WHITE_FILE)
red_df   = pd.read_csv(RED_FILE)


def add_binary_label(df: pd.DataFrame, threshold: int = 6) -> pd.DataFrame:
    df = df.copy()
    df["label"] = (df["quality"] >= threshold).astype(int)
    return df


def plot_quality_hist(df: pd.DataFrame, title: str, color: str, out_file: str):
    qmin, qmax = int(df["quality"].min()), int(df["quality"].max())
    bins = list(range(qmin, qmax + 2))

    plt.figure(figsize=(8, 5))
    plt.hist(df["quality"], bins=bins, color=color, edgecolor="black", alpha=0.8, rwidth=0.9)
    plt.title(title)
    plt.xlabel("Quality score")
    plt.ylabel("Count")
    plt.xticks(range(qmin, qmax + 1))
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()


def plot_class_distribution(df: pd.DataFrame, title: str, out_file: str):
    counts = df["label"].value_counts().reindex([0, 1], fill_value=0)
    total = int(counts.sum())
    labels = ["Bad (0)\nquality ≤ 5", f"Good (1)\nquality ≥ {THRESHOLD}"]
    values = [int(counts.loc[0]), int(counts.loc[1])]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, edgecolor="black")
    plt.title(title)
    plt.ylabel("Count")

    # annotate bars with count + percent
    for bar, v in zip(bars, values):
        pct = (v / total * 100) if total > 0 else 0
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            f"{v}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()


# Discretize (same rule for both datasets)
white_labeled = add_binary_label(white_df, THRESHOLD)
red_labeled   = add_binary_label(red_df, THRESHOLD)

# Plot original quality distributions
plot_quality_hist(
    white_labeled,
    "White wine: Quality score distribution",
    "lightyellow",
    f"{OUT_DIR}/white_quality_hist.png",
)
plot_quality_hist(
    red_labeled,
    "Red wine: Quality score distribution",
    "lightcoral",
    f"{OUT_DIR}/red_quality_hist.png",
)

# Plot class distributions after discretization
plot_class_distribution(
    white_labeled,
    f"White wine: Class distribution (good if quality ≥ {THRESHOLD})",
    f"{OUT_DIR}/white_class_dist.png",
)
plot_class_distribution(
    red_labeled,
    f"Red wine: Class distribution (good if quality ≥ {THRESHOLD})",
    f"{OUT_DIR}/red_class_dist.png",
)

# print counts for reporting
print("White class counts:\n", white_labeled["label"].value_counts().sort_index())
print("Red class counts:\n", red_labeled["label"].value_counts().sort_index())

# save labeled CSVs for later tasks
white_labeled.to_csv("data/white_with_labels.csv", index=False)
red_labeled.to_csv("data/red_with_labels.csv", index=False)
