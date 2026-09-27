import csv as _csv
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(BASE_DIR, "..", "..", "..")
sys.path.insert(0, REPO_ROOT)

import params
import splitting
from feature_engineering import add_engineered_columns
from models.regression.common import loader

MODELS_DIR = os.path.join(REPO_ROOT, "experiments", "regression", "run_leakage", "models")
DATA       = params.data_path()

FEATURES = ["launch_angle_deg", "wind_x_ms", "drag_param",
            "height_diff_m", "landing_distance_m"]
TARGET   = "initial_velocity_ms"

# Small sample so the fully-grown tree memorisation effect is clearly visible
N_SAMPLES = 10_000


def load_data(test_size=0.2, seed=42):
    if not os.path.isfile(DATA):
        print()
        print(f"  [error] Training data not found: {DATA}")
        print(  "  Run:    dvc repro generate")
        print()
        sys.exit(1)
    df = pd.read_parquet(DATA)
    df = add_engineered_columns(df)
    df = df.sample(n=N_SAMPLES, random_state=seed).reset_index(drop=True)
    X  = df[FEATURES].values
    y  = df[TARGET].values
    groups = df[loader.GROUP_COLUMN].to_numpy()
    # Group-aware whenever the active pool has groups -- see splitting.py and
    # AUDIT.md task 36/§5.1. groups_train is returned alongside the split so a
    # caller that cross-validates can pass it to splitting.cv_for().
    return splitting.split(X, y, groups, test_size=test_size, random_state=seed)


def model_path(filename):
    os.makedirs(MODELS_DIR, exist_ok=True)
    return os.path.join(MODELS_DIR, filename)


def save_metrics(model_name, **kw):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"metrics_{model_name}.csv")
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["model"] + list(kw.keys()))
        w.writerow([model_name] + [f"{v:.6f}" if isinstance(v, float) else str(v) for v in kw.values()])


def print_metrics(y_test, y_pred, model_name=None):
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    print(f"  MAE  : {mae:.3f} m/s")
    print(f"  MSE  : {mse:.3f} m^2/s^2")
    print(f"  RMSE : {rmse:.3f} m/s\n")
    print(f"  {'Actual':>10}  {'Predicted':>10}  {'Error':>8}")
    for a, p in zip(y_test[:6], y_pred[:6]):
        print(f"  {a:>10.2f}  {p:>10.2f}  {a - p:>+8.2f} m/s")
    if model_name:
        save_metrics(model_name, mae=mae, mse=mse, rmse=rmse)
