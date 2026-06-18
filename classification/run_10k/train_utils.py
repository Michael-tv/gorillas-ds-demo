import csv as _csv
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA     = os.path.join(BASE_DIR, "training_data", "training_data.csv")

FEATURES = [
    "initial_velocity_ms", "launch_angle_deg", "wind_x_ms",
    "drag_param", "height_diff_m", "target_distance_m",
]
TARGET = "hit_target"


def model_path(filename):
    os.makedirs(MODELS_DIR, exist_ok=True)
    return os.path.join(MODELS_DIR, filename)


def load_data(test_size=0.2, scale=False):
    if not os.path.isfile(DATA):
        print()
        print(f"  [error] Training data not found: {DATA}")
        print(  "  Run:    python generate_data.py")
        print()
        sys.exit(1)
    df = pd.read_csv(DATA)
    X = df[FEATURES].values
    y = df[TARGET].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
    hits = int(y_train.sum())
    print(f"Loaded {len(X_train)} train / {len(X_test)} test samples")
    print(f"  Train hits: {hits}  misses: {len(y_train) - hits}\n")
    return X_train, X_test, y_train, y_test, scaler


def save_metrics(model_name, **kw):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"metrics_{model_name}.csv")
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["model"] + list(kw.keys()))
        w.writerow([model_name] + [f"{v:.6f}" if isinstance(v, float) else str(v) for v in kw.values()])


def print_metrics(y_test, y_pred, y_prob=None):
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob) if y_prob is not None else float("nan")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    if y_prob is not None:
        print(f"  ROC-AUC   : {auc:.4f}")
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"               Miss   Hit")
    print(f"  Actual Miss  {tn:>5}  {fp:>5}")
    print(f"  Actual Hit   {fn:>5}  {tp:>5}")
    print()
    script = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    model_name = script[6:] if script.startswith("train_") else script
    save_metrics(model_name, accuracy=acc, precision=prec, recall=rec, f1=f1, roc_auc=auc)