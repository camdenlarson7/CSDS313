import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Config (edit if needed)
# -----------------------------
METRICS_DIR = "artifacts"
SPLITS_DIR = "artifacts/splits"   # optional; script will skip extra plots if missing
OUT_DIR = "artifacts/partC"
PRIMARY_METRIC = "balanced_accuracy"  # or: "f1", "accuracy", "recall", "precision"

LOGREG_PATH = os.path.join(METRICS_DIR, "logreg_metrics.csv")
RF_PATH     = os.path.join(METRICS_DIR, "rf_metrics.csv")
MLP_PATH    = os.path.join(METRICS_DIR, "mlp_metrics.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
def parse_setting(s: str):
    """
    Parses strings like:
      "Train RED -> Test WHITE (cross-domain)"
    Returns (train_domain, test_domain, domain_type, short_label)
    """
    # Train <X> -> Test <Y> (...)
    m = re.search(r"Train\s+(\w+)\s*->\s*Test\s+(\w+).*\(([^)]+)\)", s, re.IGNORECASE)
    if not m:
        # fallback
        return ("UNK", "UNK", "UNK", s)
    train_d = m.group(1).upper()
    test_d  = m.group(2).upper()
    dtype   = m.group(3).lower()
    short = f"{train_d[0]}→{test_d[0]}"  # R→W etc.
    return train_d, test_d, dtype, short

def load_metrics(path: str, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["model"] = model_name
    parsed = df["setting"].apply(parse_setting)
    df["train_domain"] = parsed.apply(lambda x: x[0])
    df["test_domain"]  = parsed.apply(lambda x: x[1])
    df["domain_type"]  = parsed.apply(lambda x: x[2])
    df["setting_short"] = parsed.apply(lambda x: x[3])
    return df

def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))

def safe_round(x):
    try:
        return float(x)
    except Exception:
        return np.nan

# -----------------------------
# Load & combine metrics
# -----------------------------
metrics = pd.concat([
    load_metrics(LOGREG_PATH, "LogReg"),
    load_metrics(RF_PATH,     "RF"),
    load_metrics(MLP_PATH,    "MLP"),
], ignore_index=True)

# Ensure numeric cols are numeric
for col in ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]:
    metrics[col] = pd.to_numeric(metrics[col], errors="coerce")

# Sanity: ensure we have the 4 settings per model
# (R→R, R→W, W→W, W→R)
print("Loaded rows:", len(metrics))
print(metrics[["model", "setting_short", "domain_type", PRIMARY_METRIC]].sort_values(["model", "setting_short"]))

# -----------------------------
# Visualization 1: Metric by setting (grouped bars)
# -----------------------------
order = ["R→R", "R→W", "W→W", "W→R"]
models_order = ["LogReg", "RF", "MLP"]

plot_df = metrics.copy()
plot_df["setting_short"] = pd.Categorical(plot_df["setting_short"], categories=order, ordered=True)
plot_df["model"] = pd.Categorical(plot_df["model"], categories=models_order, ordered=True)
plot_df = plot_df.sort_values(["setting_short", "model"])

plt.figure(figsize=(9, 5))
x = np.arange(len(order))
width = 0.25

for i, m in enumerate(models_order):
    vals = []
    for s in order:
        row = plot_df[(plot_df["model"] == m) & (plot_df["setting_short"] == s)]
        vals.append(row[PRIMARY_METRIC].iloc[0] if len(row) else np.nan)
    plt.bar(x + (i - 1) * width, vals, width=width, label=m, edgecolor="black")

plt.xticks(x, order)
plt.ylim(0, 1)
plt.ylabel(PRIMARY_METRIC)
plt.title(f"{PRIMARY_METRIC} by setting (in-domain vs cross-domain)")
plt.legend()
plt.tight_layout()
out1 = os.path.join(OUT_DIR, f"metric_by_setting_{PRIMARY_METRIC}.png")
plt.savefig(out1, dpi=160, bbox_inches="tight")
plt.close()

# -----------------------------
# Visualization 2: In-domain avg vs Cross-domain avg (per model)
# -----------------------------
summary = (metrics
           .groupby(["model", "domain_type"], as_index=False)[PRIMARY_METRIC]
           .mean()
           .rename(columns={PRIMARY_METRIC: "avg_metric"}))

pivot = summary.pivot(index="model", columns="domain_type", values="avg_metric")
# Ensure columns exist
for col in ["in-domain", "cross-domain"]:
    if col not in pivot.columns:
        pivot[col] = np.nan
pivot = pivot.loc[models_order]

plt.figure(figsize=(7, 4))
x = np.arange(len(models_order))
plt.bar(x - 0.18, pivot["in-domain"].values, width=0.36, label="in-domain", edgecolor="black")
plt.bar(x + 0.18, pivot["cross-domain"].values, width=0.36, label="cross-domain", edgecolor="black")
plt.xticks(x, models_order)
plt.ylim(0, 1)
plt.ylabel(f"Average {PRIMARY_METRIC}")
plt.title(f"Average {PRIMARY_METRIC}: in-domain vs cross-domain")
plt.legend()
plt.tight_layout()
out2 = os.path.join(OUT_DIR, f"in_vs_cross_avg_{PRIMARY_METRIC}.png")
plt.savefig(out2, dpi=160, bbox_inches="tight")
plt.close()

# Save combined metrics + summary table
combined_out = os.path.join(OUT_DIR, "combined_metrics.csv")
metrics.to_csv(combined_out, index=False)
summary_out = os.path.join(OUT_DIR, f"summary_avgs_{PRIMARY_METRIC}.csv")
pivot.to_csv(summary_out)

# -----------------------------
# Optional: Extra plots for Q3 (label imbalance + feature shift)
# -----------------------------
def try_extra_plots():
    needed = ["red_train.csv", "red_test.csv", "white_train.csv", "white_test.csv"]
    if not os.path.isdir(SPLITS_DIR):
        print(f"Skipping extra plots: '{SPLITS_DIR}' not found.")
        return
    for f in needed:
        if not os.path.exists(os.path.join(SPLITS_DIR, f)):
            print(f"Skipping extra plots: missing {f} in {SPLITS_DIR}.")
            return

    red_train = pd.read_csv(os.path.join(SPLITS_DIR, "red_train.csv"))
    white_train = pd.read_csv(os.path.join(SPLITS_DIR, "white_train.csv"))
    red_test = pd.read_csv(os.path.join(SPLITS_DIR, "red_test.csv"))
    white_test = pd.read_csv(os.path.join(SPLITS_DIR, "white_test.csv"))

    # Label imbalance plot (train + test)
    def label_props(df):
        c = df["label"].value_counts().reindex([0, 1], fill_value=0)
        total = c.sum()
        return (c / total).values  # [bad, good]

    labels = ["Bad (0)", "Good (1)"]
    sets = {
        "Red train": label_props(red_train),
        "Red test":  label_props(red_test),
        "White train": label_props(white_train),
        "White test":  label_props(white_test),
    }

    plt.figure(figsize=(8, 4.5))
    x = np.arange(len(labels))
    width = 0.18
    for i, (name, vals) in enumerate(sets.items()):
        plt.bar(x + (i - 1.5) * width, vals, width=width, label=name, edgecolor="black")
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("Proportion")
    plt.title("Label distribution (good/bad) across domains")
    plt.legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "label_distribution.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()

    # Feature-shift plot: pick top 3 features with largest standardized mean difference (train sets)
    feat_cols = [c for c in red_train.columns if c not in ["quality", "label"]]
    r = red_train[feat_cols]
    w = white_train[feat_cols]

    r_mean, w_mean = r.mean(), w.mean()
    r_std, w_std = r.std(ddof=0).replace(0, np.nan), w.std(ddof=0).replace(0, np.nan)
    pooled = np.sqrt((r_std**2 + w_std**2) / 2.0)
    smd = ((r_mean - w_mean).abs() / pooled).sort_values(ascending=False)
    top_feats = smd.head(3).index.tolist()

    for feat in top_feats:
        plt.figure(figsize=(7.5, 4.5))
        plt.hist(red_train[feat], bins=30, alpha=0.6, label="Red train", edgecolor="black")
        plt.hist(white_train[feat], bins=30, alpha=0.6, label="White train", edgecolor="black")
        plt.title(f"Feature distribution shift: {feat}")
        plt.xlabel(feat)
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        out = os.path.join(OUT_DIR, f"feature_shift_{feat.replace(' ', '_')}.png")
        plt.savefig(out, dpi=160, bbox_inches="tight")
        plt.close()

    print("Extra plots generated: label_distribution.png + feature_shift_*.png")

try_extra_plots()

# -----------------------------
# Auto-drafts for Part C answers (<=100 words each)
# -----------------------------
# Q1: Consistency across in-domain settings (R→R and W→W)
in_domain = metrics[metrics["domain_type"] == "in-domain"].copy()
consistency = (in_domain.groupby("model")[PRIMARY_METRIC]
               .agg(["mean", "std", "min", "max"])
               .sort_values("std", ascending=True))

most_consistent_model = consistency.index[0]
mc_std = consistency.loc[most_consistent_model, "std"]
mc_min = consistency.loc[most_consistent_model, "min"]
mc_max = consistency.loc[most_consistent_model, "max"]

# Overfitting-ish signal: in-domain avg - cross-domain avg
gap = (pivot["in-domain"] - pivot["cross-domain"]).sort_values(ascending=False)
largest_gap_model = gap.index[0]
largest_gap_val = gap.loc[largest_gap_model]

q1 = (
    f"Using {PRIMARY_METRIC} as the primary metric, {most_consistent_model} was the most consistent in-domain "
    f"(std={mc_std:.3f}, range {mc_min:.3f}–{mc_max:.3f} across R→R and W→W). "
    f"The largest drop from in-domain to cross-domain was for {largest_gap_model} (Δ={largest_gap_val:.3f}), "
    f"suggesting it fits domain-specific patterns strongly. This looks more like sensitivity to domain shift "
    f"than classic train-set overfitting, since we only evaluate on held-out tests."
)

# Q2: Best cross-domain + degradation
cross_domain = metrics[metrics["domain_type"] == "cross-domain"].copy()
cross_avg = cross_domain.groupby("model")[PRIMARY_METRIC].mean().sort_values(ascending=False)
best_generalizer = cross_avg.index[0]
best_cross = cross_avg.loc[best_generalizer]
best_in = pivot.loc[best_generalizer, "in-domain"]
degrade = best_in - best_cross

q2 = (
    f"{best_generalizer} generalized best across wine types by average cross-domain {PRIMARY_METRIC} "
    f"({best_cross:.3f}). Compared to its in-domain average ({best_in:.3f}), performance degraded by "
    f"{degrade:.3f} absolute. Overall, all models drop substantially in cross-domain tests, indicating "
    f"the relationship between features and the ‘good’ label differs between red and white wines."
)

# Q3: Factors (generic + optionally references your extra plots)
q3 = (
    "Cross-domain performance differences are likely driven by distribution shift: the same physicochemical "
    "features can have different ranges and predictive relationships in red vs white wines. Label imbalance "
    "(different good/bad rates) can also skew precision/recall and cause conservative models to rarely predict "
    "the positive class. Model sensitivity matters: random forests can over-specialize to domain-specific splits, "
    "while simpler linear models may transfer better but underfit. Differences in dataset size and sampling noise "
    "also affect stability, especially for the smaller red dataset."
)

# Truncate if you want hard <=100 words (usually these are already close)
def truncate_to_words(text, max_words=100):
    words = re.findall(r"\S+", text)
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip() + "…"

q1_t = truncate_to_words(q1, 100)
q2_t = truncate_to_words(q2, 100)
q3_t = truncate_to_words(q3, 100)

print("\n=== AUTO-DRAFT ANSWERS (<=100 words each) ===")
print("\nQ1:", q1_t, f"\n(word count: {word_count(q1_t)})")
print("\nQ2:", q2_t, f"\n(word count: {word_count(q2_t)})")
print("\nQ3:", q3_t, f"\n(word count: {word_count(q3_t)})")

print("\nSaved plots:")
print(" -", out1)
print(" -", out2)
print("Also saved:")
print(" -", combined_out)
print(" -", summary_out)
