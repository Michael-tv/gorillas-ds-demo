"""Shared loader for every regression run (run_raw_10k/20k/40k, run_eng_10k/
20k/40k). Always reads the raw physical columns from the shared, pre-shuffled
pool (see data_generation/generate.py) and cleans/splits them -- it never
derives wind_x_ms/drag_param/height_diff_m itself.

The raw-vs-engineered contrast those run_* pairs exist to demonstrate now
lives entirely in the model's sklearn Pipeline: run_raw_*/train_utils.py sets
FEATURE_STEP = ("engineer", "passthrough"), run_eng_*/train_utils.py sets it
to ("engineer", feature_engineering.EngineeredFeatures(...)) -- same raw
input, different first pipeline step.
"""
import csv as _csv
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

# params lives at the repo root. The run_*/train_utils.py shims already put it
# on sys.path before importing this module, but doing it here too means the
# loader works when imported directly (e.g. from a test or a notebook).
_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import params
from data_generation.contract import check_contract

FEATURES = ["launch_angle_deg", "wind_speed_ms", "wind_direction_norm",
            "mass_kg", "radius_m", "drag_coeff", "launch_height_m", "landing_height_m", "landing_distance_m"]
TARGET = "initial_velocity_ms"

# Valid physical ranges for each raw feature and the target. Cleaning always
# runs on these raw columns, before the pipeline's engineering step -- so
# run_raw_* and run_eng_* apply the exact same cleaning.
# None means no bound on that side.
FEATURE_RANGES = {
    "launch_angle_deg":    (0,    90),
    "wind_speed_ms":       (0,    None),
    "wind_direction_norm": (-1,   1),
    "mass_kg":             (0,    None),
    "radius_m":            (0,    None),
    "drag_coeff":          (0,    None),
    "launch_height_m":     (0,    None),
    "landing_height_m":    (0,    None),
    "landing_distance_m":  (0,    None),
    "initial_velocity_ms": (0,    None),
}


def _clean_iqr(df, multiplier=1.5):
    mask = pd.Series(True, index=df.index)
    for col in FEATURES + [TARGET]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue   # skip constant / categorical columns
        mask &= df[col].between(q1 - multiplier * iqr, q3 + multiplier * iqr)
    return df[mask]


def _clean_range(df):
    mask = pd.Series(True, index=df.index)
    for col, (lo, hi) in FEATURE_RANGES.items():
        if col not in df.columns:
            continue
        if lo is not None:
            mask &= df[col] >= lo
        if hi is not None:
            mask &= df[col] <= hi
    return df[mask]


def _clean_no_outlier(df):
    return df[df["is_outlier"] == "none"]


def load_data(data_path, n_samples=None):
    """Read, slice, and clean the pool -- returns (X, y) for the caller to
    split. Splitting is a training decision (test_size, stratify, ...), not a
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
    n_orig = len(df)
    clean  = os.environ.get("TRAIN_CLEAN", "")
    if clean == "iqr":
        df = _clean_iqr(df)
        print(f"  IQR cleaning   : {n_orig} -> {len(df)} rows ({n_orig - len(df)} removed)")
    elif clean == "range":
        df = _clean_range(df)
        print(f"  Range cleaning : {n_orig} -> {len(df)} rows ({n_orig - len(df)} removed)")
    elif clean == "no_outlier":
        df = _clean_no_outlier(df)
        print(f"  No-outlier cleaning: {n_orig} -> {len(df)} rows ({n_orig - len(df)} removed)")
    X = df[FEATURES]
    y = df[TARGET].values
    print(f"Loaded {len(X)} samples\n")
    return X, y


def model_path(models_dir, filename):
    os.makedirs(models_dir, exist_ok=True)
    name = os.environ.get("TRAIN_MODEL_NAME", "")
    if name:
        return os.path.join(models_dir, f"model_{name}.joblib")
    return os.path.join(models_dir, filename)


def save_metrics(models_dir, model_name, **kw):
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"metrics_{model_name}.csv")
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["model"] + list(kw.keys()))
        w.writerow([model_name] + [f"{v:.6f}" if isinstance(v, float) else str(v) for v in kw.values()])


def print_metrics(models_dir, y_test, y_pred):
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    print(f"  MAE  : {mae:.3f} m/s")
    print(f"  MSE  : {mse:.3f} m^2/s^2")
    print(f"  RMSE : {rmse:.3f} m/s\n")
    print(f"  {'Actual':>10}  {'Predicted':>10}  {'Error':>8}")
    for a, p in zip(y_test[:6], y_pred[:6]):
        print(f"  {a:>10.2f}  {p:>10.2f}  {a - p:>+8.2f} m/s")
    model_name = os.environ.get("TRAIN_MODEL_NAME", "")
    if not model_name:
        script = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        model_name = script[6:] if script.startswith("train_") else script
    save_metrics(models_dir, model_name, mae=mae, mse=mse, rmse=rmse)
