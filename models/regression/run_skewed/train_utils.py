import csv as _csv
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(BASE_DIR, "..", "..", "..")
sys.path.insert(0, REPO_ROOT)

import params
from feature_engineering import add_engineered_columns

MODELS_DIR = os.path.join(REPO_ROOT, "experiments", "regression", "run_skewed", "models")
DATA       = params.data_path()
MAX_ELEVATION_DEG = 30  # skewed = a low-angle-only slice of the standard pool, not a separate draw

FEATURES = ["launch_angle_deg", "wind_x_ms", "drag_param",
            "height_diff_m", "landing_distance_m"]
TARGET = "initial_velocity_ms"


def model_path(filename):
    os.makedirs(MODELS_DIR, exist_ok=True)
    name = os.environ.get("TRAIN_MODEL_NAME", "")
    if name:
        return os.path.join(MODELS_DIR, f"model_{name}.joblib")
    return os.path.join(MODELS_DIR, filename)


def _clean_no_outlier(df):
    return df[df["is_outlier"] == "none"]


def load_data(test_size=0.2):
    if not os.path.isfile(DATA):
        print()
        print(f"  [error] Training data not found: {DATA}")
        print(  "  Run:    dvc repro generate")
        print()
        sys.exit(1)
    df = pd.read_parquet(DATA)
    n_pool = len(df)
    df = df[df["launch_angle_deg"] <= MAX_ELEVATION_DEG]
    print(f"  Skewed filter  : {n_pool} -> {len(df)} rows (kept launch_angle_deg <= {MAX_ELEVATION_DEG})")
    df = add_engineered_columns(df)
    n_orig = len(df)
    clean  = os.environ.get("TRAIN_CLEAN", "")
    if clean == "no_outlier":
        df = _clean_no_outlier(df)
        print(f"  No-outlier cleaning: {n_orig} -> {len(df)} rows ({n_orig - len(df)} removed)")
    X = df[FEATURES].values
    y = df[TARGET].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print(f"Loaded {len(X_train)} train / {len(X_test)} test samples\n")
    return X_train, X_test, y_train, y_test


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
    if model_name is None:
        model_name = os.environ.get("TRAIN_MODEL_NAME", "")
        if not model_name:
            script = os.path.splitext(os.path.basename(sys.argv[0]))[0]
            model_name = script[6:] if script.startswith("train_") else script
    save_metrics(model_name, mae=mae, mse=mse, rmse=rmse)
