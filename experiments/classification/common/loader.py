"""Shared loader for classification. Reads the raw dataset and derives the
engineered feature columns (wind_x_ms, drag_param, height_diff_m) at load
time via feature_engineering.add_engineered_columns; generated data never
stores them.

model_name is an explicit parameter to print_metrics rather than inferred,
since experiments/classification/train.py calls this module directly.
"""
import csv as _csv
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                              confusion_matrix, f1_score, precision_score,
                              recall_score, roc_auc_score)

from feature_engineering import add_engineered_columns

# params lives at the repo root. The run_*/train_utils.py shims already put it
# on sys.path before importing this module, but doing it here too means the
# loader works when imported directly (e.g. from a test or a notebook).
_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import params
from data_generation.contract import check_contract

FEATURES = [
    "initial_velocity_ms", "launch_angle_deg", "wind_x_ms",
    "drag_param", "height_diff_m", "target_distance_m",
]
TARGET = "hit_target"

GROUP_COLUMN = "group_id"   # generate.COLUMNS' 14th column -- see splitting.py


def _clean_no_outlier(df):
    return df[df["is_outlier"] == "none"]


def load_data(data_path, n_samples=None, clean=""):
    """Read, slice, and engineer the pool -- returns (X, y) for the caller to
    split (with `stratify=y`). Splitting is a training decision, not a
    loading one, so it lives in each training script instead of here."""
    if not os.path.isfile(data_path):
        print()
        print(f"  [error] Training data not found: {data_path}")
        print(  "  Run:    dvc repro (or the matching data_generation script)")
        print()
        sys.exit(1)
    # Validated on read, not just on write: data/ is DVC-cached, so the pool on
    # disk can predate the current code (e.g. missing group_id) with nothing in
    # the working tree showing it.
    df = check_contract(pd.read_parquet(data_path), source=os.path.basename(data_path),
                        verbose=False)
    # data_path is the shared, pre-shuffled pool; prefix-slicing gives nested
    # samples across size tiers. take_samples raises rather than silently
    # returning a short frame if the pool has fewer rows than n_samples.
    df = params.take_samples(df, n_samples, pool_name=os.path.basename(data_path))
    if clean == "no_outlier":
        n_orig = len(df)
        df = _clean_no_outlier(df)
        print(f"  No-outlier cleaning: {n_orig} -> {len(df)} rows ({n_orig - len(df)} removed)")
    df = add_engineered_columns(df)
    X = df[FEATURES].values
    y = df[TARGET].values
    # See the note in experiments/regression/common/loader.py: group_id is not
    # a feature, but a group-aware split is impossible without it.
    groups = df[GROUP_COLUMN].to_numpy()
    print(f"Loaded {len(X)} samples -- hits: {int(y.sum())}  misses: {len(y) - int(y.sum())}\n")
    return X, y, groups


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


def print_metrics(models_dir, model_name, y_test, y_pred, y_prob=None):
    """Report precision/recall/PR-AUC first, not accuracy: with a 5-7% hit rate,
    predicting "miss" every time scores 93-95% accuracy while doing nothing
    useful, so the always-miss baseline is printed alongside it for comparison.
    """
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    acc  = accuracy_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_prob) if y_prob is not None else float("nan")
    # PR-AUC (average precision) is the right summary under imbalance: ROC-AUC
    # is flattered by the huge true-negative pool, PR-AUC is not.
    pr_auc = average_precision_score(y_test, y_prob) if y_prob is not None else float("nan")

    pos_rate  = float(np.mean(y_test))
    baseline  = 1.0 - pos_rate          # accuracy of predicting "miss" every time

    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    if y_prob is not None:
        print(f"  PR-AUC    : {pr_auc:.4f}   (positive rate {pos_rate:.4f} = a random"
              f" classifier's PR-AUC)")
        print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  Accuracy  : {acc:.4f}   (always-miss baseline {baseline:.4f}"
          f"{'  <-- accuracy beats the model here' if baseline > acc else ''})")
    if baseline > acc:
        print("  NOTE: this model is LESS accurate than predicting 'miss' every time. "
              "That is\n        why accuracy is not the metric to judge it by -- see "
              "precision/recall above.")

    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"               Miss   Hit")
    print(f"  Actual Miss  {tn:>5}  {fp:>5}")
    print(f"  Actual Hit   {fn:>5}  {tp:>5}")
    print()
    save_metrics(models_dir, model_name, precision=prec, recall=rec, f1=f1,
                 pr_auc=pr_auc, roc_auc=auc, accuracy=acc,
                 baseline_accuracy=baseline, positive_rate=pos_rate)
