"""Shared loader for the classification datasets (run_10k/20k/40k).

Replaces the byte-identical train_utils.py copies that used to live in each of
those folders. Reads the raw generated dataset and derives the engineered
feature columns (wind_x_ms, drag_param, height_diff_m) at load time via
feature_engineering.add_engineered_columns -- generated data never stores them.
"""
import csv as _csv
import os
import sys

import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score)

from feature_engineering import add_engineered_columns

FEATURES = [
    "initial_velocity_ms", "launch_angle_deg", "wind_x_ms",
    "drag_param", "height_diff_m", "target_distance_m",
]
TARGET = "hit_target"


def load_data(data_path, n_samples=None):
    """Read, slice, and engineer the pool -- returns (X, y) for the caller to
    split (with `stratify=y`). Splitting is a training decision, not a
    loading one, so it lives in each training script instead of here."""
    if not os.path.isfile(data_path):
        print()
        print(f"  [error] Training data not found: {data_path}")
        print(  "  Run:    dvc repro (or the matching data_generation script)")
        print()
        sys.exit(1)
    df = pd.read_parquet(data_path)
    if n_samples is not None:
        # data_path is the shared, pre-shuffled pool -- slicing a prefix here
        # (rather than caching a separate generated file per size) gives the
        # 10k/20k/40k tiers nested samples of one draw, so growing the sample
        # size is the only thing that changes between tiers.
        df = df.iloc[:n_samples]
    df = add_engineered_columns(df)
    X = df[FEATURES].values
    y = df[TARGET].values
    print(f"Loaded {len(X)} samples -- hits: {int(y.sum())}  misses: {len(y) - int(y.sum())}\n")
    return X, y


def model_path(models_dir, filename):
    os.makedirs(models_dir, exist_ok=True)
    return os.path.join(models_dir, filename)


def save_metrics(models_dir, model_name, **kw):
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"metrics_{model_name}.csv")
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["model"] + list(kw.keys()))
        w.writerow([model_name] + [f"{v:.6f}" if isinstance(v, float) else str(v) for v in kw.values()])


def print_metrics(models_dir, y_test, y_pred, y_prob=None):
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
    save_metrics(models_dir, model_name, accuracy=acc, precision=prec, recall=rec, f1=f1, roc_auc=auc)
