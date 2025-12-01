import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix
)

SPLIT_DIR = "artifacts/splits"

def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(f"{SPLIT_DIR}/{name}")

def get_X_y(df: pd.DataFrame):
    X = df.drop(columns=["quality", "label"])
    y = df["label"].astype(int)
    return X, y

def evaluate(model, X, y, title: str):
    preds = model.predict(X)

    acc = accuracy_score(y, preds)
    bal_acc = balanced_accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    cm = confusion_matrix(y, preds)

    print(f"\n=== {title} ===")
    print(f"Accuracy:           {acc:.4f}")
    print(f"Balanced accuracy:  {bal_acc:.4f}")
    print(f"Precision (good=1): {prec:.4f}")
    print(f"Recall (good=1):    {rec:.4f}")
    print(f"F1 (good=1):        {f1:.4f}")
    print("Confusion matrix [[TN FP],[FN TP]]:")
    print(cm)

    return {
        "setting": title,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

# Load splits
red_train = load_split("red_train.csv")
red_test  = load_split("red_test.csv")
white_train = load_split("white_train.csv")
white_test  = load_split("white_test.csv")

Xr_train, yr_train = get_X_y(red_train)
Xr_test,  yr_test  = get_X_y(red_test)
Xw_train, yw_train = get_X_y(white_train)
Xw_test,  yw_test  = get_X_y(white_test)

# Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

results = []

# Train on RED
rf_model.fit(Xr_train, yr_train)
results.append(evaluate(rf_model, Xr_test, yr_test, "Train RED -> Test RED (in-domain)"))
results.append(evaluate(rf_model, Xw_test, yw_test, "Train RED -> Test WHITE (cross-domain)"))

# Train on WHITE
rf_model.fit(Xw_train, yw_train)
results.append(evaluate(rf_model, Xw_test, yw_test, "Train WHITE -> Test WHITE (in-domain)"))
results.append(evaluate(rf_model, Xr_test, yr_test, "Train WHITE -> Test RED (cross-domain)"))

# Save metrics table for Part C
metrics_df = pd.DataFrame(results)
metrics_df.to_csv("artifacts/rf_metrics.csv", index=False)
print("\nSaved metrics to artifacts/rf_metrics.csv")
print("\n", metrics_df)

feat_importances = pd.Series(rf_model.feature_importances_, index=Xw_train.columns).sort_values(ascending=False)
feat_importances.to_csv("artifacts/rf_feature_importances_latest.csv")
print("\nSaved latest feature importances to artifacts/rf_feature_importances_latest.csv")
