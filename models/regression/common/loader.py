"""Shared loader for every regression run. Always reads the raw physical
columns from the pool it's pointed at and cleans/splits them -- it never
derives wind_x_ms/drag_param/height_diff_m itself.

The raw-vs-engineered contrast (raw/eng scripts) lives entirely in the
model's sklearn Pipeline: each `experiment_raw`/`experiment_eng` script
builds `("features", EngineeredFeatures(output_columns=RAW_FEATURES_or_
ENG_FEATURES))` -- same raw input, different first Pipeline step, built
explicitly in the one script that trains that model on that scheme (see
models/regression/standard/experiment_raw/train_linear_regression.py).

`clean` and `model_name` are explicit parameters, not environment variables
(TRAIN_CLEAN / TRAIN_MODEL_NAME) -- the previous design read them out of
os.environ because the injected-PYTHONPATH call chain had no other way to
pass per-run values into a shared script (AUDIT.md C1 / the run_*/
train_utils.py cleanup). Every `models/<domain>/<mode>/experiment_<name>/
train_<model>.py` script calls this module directly and just passes them.
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

GROUP_COLUMN = "group_id"   # generate.COLUMNS' 14th column -- see splitting.py

# Valid physical ranges for each raw feature and the target. Cleaning always
# runs on these raw columns, before the pipeline's engineering step -- so
# run_raw and run_eng apply the exact same cleaning.
# None means no bound on that side.
#
# landing_distance_m has NO lower bound, and that is the whole point of this
# block (AUDIT.md task 13b). It used to be bounded at 0, on the assumption that a
# negative landing distance had to be corrupt. It does not: a high-angle shot
# into a strong headwind genuinely lands behind the launch point -- 73 m/s at 79
# degrees into 17.4 m/s of headwind lands at -65.0 m, where the same shot
# windless lands at +30.8 m.
#
# Measured over the 50,000-row pool, that one bound was doing most of the damage:
#
#   clean:"range" dropped                          964 rows
#     of which the landing_distance_m >= 0 bound    807  (84%)
#       of which is_outlier == "none"               727  <- physically valid
#     every other bound, clean rows dropped           0  <- all doing their job
#
#   with the bound removed, clean:"range" drops     167 rows, 100% data_error
#
# So the demo was mostly deleting real data while claiming to remove corrupt
# data. It is now a precise corrupt-row detector rather than a blunt one: of the
# ~500 deliberately corrupted rows it catches 167, all of them genuinely corrupt.
# Less sensitive, far more precise -- and that trade is itself worth showing, as
# what bounds-based cleaning can and cannot do.
FEATURE_RANGES = {
    "launch_angle_deg":    (0,    90),
    "wind_speed_ms":       (0,    None),
    "wind_direction_norm": (-1,   1),
    "mass_kg":             (0,    None),
    "radius_m":            (0,    None),
    "drag_coeff":          (0,    None),
    "launch_height_m":     (0,    None),
    "landing_height_m":    (0,    None),
    "landing_distance_m":  (None, None),   # signed: see the note above
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


def load_data(data_path, n_samples=None, clean=""):
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
    # group_id travels with (X, y) rather than being dropped here: it is not a
    # feature, but the caller cannot build a group-aware split without it, and
    # dropping it at the loader is what made task 36 impossible even after task
    # 32 created the key (AUDIT.md finding N2). splitting.split() decides what to
    # do with it -- a Gorillas pool groups 32 throws per board, the Python pool
    # gives every row its own id and so degrades to an ordinary random split.
    groups = df[GROUP_COLUMN].to_numpy()
    print(f"Loaded {len(X)} samples\n")
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


def print_metrics(models_dir, model_name, y_test, y_pred):
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    print(f"  MAE  : {mae:.3f} m/s")
    print(f"  MSE  : {mse:.3f} m^2/s^2")
    print(f"  RMSE : {rmse:.3f} m/s\n")
    print(f"  {'Actual':>10}  {'Predicted':>10}  {'Error':>8}")
    for a, p in zip(y_test[:6], y_pred[:6]):
        print(f"  {a:>10.2f}  {p:>10.2f}  {a - p:>+8.2f} m/s")
    save_metrics(models_dir, model_name, mae=mae, mse=mse, rmse=rmse)
