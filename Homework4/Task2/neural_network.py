import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
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

# Simple MLP (neural net)
# Small network + early stopping to avoid overfitting
mlp = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(32,),   # 1 hidden layer, 32 neurons (simple)
        activation="relu",
        solver="adam",
        alpha=1e-4,                 # L2 regularization
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42
    ))
])

results = []

# Train on RED
mlp.fit(Xr_train, yr_train)
results.append(evaluate(mlp, Xr_test, yr_test, "Train RED -> Test RED (in-domain)"))
results.append(evaluate(mlp, Xw_test, yw_test, "Train RED -> Test WHITE (cross-domain)"))

# Train on WHITE
mlp.fit(Xw_train, yw_train)
results.append(evaluate(mlp, Xw_test, yw_test, "Train WHITE -> Test WHITE (in-domain)"))
results.append(evaluate(mlp, Xr_test, yr_test, "Train WHITE -> Test RED (cross-domain)"))

metrics_df = pd.DataFrame(results)
metrics_df.to_csv("artifacts/mlp_metrics.csv", index=False)
print("\nSaved metrics to artifacts/mlp_metrics.csv")
print("\n", metrics_df)
