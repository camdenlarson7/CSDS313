import os
import pandas as pd
from sklearn.model_selection import train_test_split

OUT_DIR = "artifacts/splits"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2  # 20% test, 80% train

white_file = "data/white_with_labels.csv"
red_file   = "data/red_with_labels.csv"

white = pd.read_csv(white_file)
red   = pd.read_csv(red_file)

def split_features_label(df: pd.DataFrame):
    # Features exclude quality + label
    X = df.drop(columns=["label", "quality"])
    y = df["label"]
    return X, y

# Train/test splits
Xr, yr = split_features_label(red)
Xw, yw = split_features_label(white)

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr, yr, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=yr
)
Xw_train, Xw_test, yw_train, yw_test = train_test_split(
    Xw, yw, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=yw
)

# Reattach quality/label for saving (use indices from split)
red_train_df = red.loc[Xr_train.index].reset_index(drop=True)
red_test_df  = red.loc[Xr_test.index].reset_index(drop=True)

white_train_df = white.loc[Xw_train.index].reset_index(drop=True)
white_test_df  = white.loc[Xw_test.index].reset_index(drop=True)

# Save splits
red_train_df.to_csv(f"{OUT_DIR}/red_train.csv", index=False)
red_test_df.to_csv(f"{OUT_DIR}/red_test.csv", index=False)

white_train_df.to_csv(f"{OUT_DIR}/white_train.csv", index=False)
white_test_df.to_csv(f"{OUT_DIR}/white_test.csv", index=False)

# Quick sanity printout
def summarize(name, df):
    counts = df["label"].value_counts().sort_index()
    total = len(df)
    bad = int(counts.get(0, 0))
    good = int(counts.get(1, 0))
    print(f"{name}: n={total} | bad={bad} ({bad/total:.3f}) | good={good} ({good/total:.3f})")

print("\n=== Split summaries ===")
summarize("RED train", red_train_df)
summarize("RED test ", red_test_df)
summarize("WHITE train", white_train_df)
summarize("WHITE test", white_test_df)

print(f"\nSaved CSV splits to: {OUT_DIR}/")
