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
    # Validated on read, not only on write: data/ is DVC-cached rather than in
    # Git, so the pool on disk can predate the current code (e.g. a 13-column
    # pool generated before group_id existed) with nothing in the working tree
    # showing it. Both producers write through the same checks -- see
    # data_generation/contract.py (AUDIT.md task 38).
    df = check_contract(pd.read_parquet(data_path), source=os.path.basename(data_path),
                        verbose=False)
    # data_path is the shared, pre-shuffled pool -- prefix-slicing here (rather
    # than caching a separate generated file per size) gives the size tiers
    # nested samples of one draw, so growing the sample size is the only thing
    # that changes between tiers. take_samples raises rather than silently
    # returning a short frame when the pool holds fewer than n_samples rows,
    # which a Gorillas pool (5,000 rows vs n_samples: 40000) does (AUDIT.md
    # task 35 / §5.5).
    df = params.take_samples(df, n_samples, pool_name=os.path.basename(data_path))
    df = add_engineered_columns(df)
    X = df[FEATURES].values
    y = df[TARGET].values
    # See the note in models/regression/common/loader.py: group_id is not a
    # feature, but a group-aware split is impossible without it (AUDIT.md task
    # 36 / finding N2).
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
