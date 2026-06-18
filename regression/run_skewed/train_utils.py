import csv as _csv
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA     = os.path.join(BASE_DIR, "training_data", "training_data.csv")

FEATURES = ["launch_angle_deg", "wind_x_ms", "drag_param",
            "height_diff_m", "landing_distance_m"]
TARGET   = "initial_velocity_ms"


def model_path(filename):
    os.makedirs(MODELS_DIR, exist_ok=True)
    name = os.environ.get("TRAIN_MODEL_NAME", "")
    if name:
        return os.path.join(MODELS_DIR, f"model_{name}.joblib")
    return os.path.join(MODELS_DIR, filename)


def _clean_no_outlier(df):
    return df[df["is_outlier"] == "none"]


def load_data(test_size=0.2, scale=False):
    if not os.path.isfile(DATA):
        print()
        print(f"  [error] Training data not found: {DATA}")
        print( "  Run:    python generate_data.py")
        print()
        sys.exit(1)
    df = pd.read_csv(DATA)
    n_orig = len(df)
    clean  = os.environ.get("TRAIN_CLEAN", "")
    if clean == "no_outlier":
        df = _clean_no_outlier(df)
        print(f"  No-outlier cleaning: {n_orig} → {len(df)} rows ({n_orig - len(df)} removed)")
    X = df[FEATURES].values
    y = df[TARGET].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
    print(f"Loaded {len(X_train)} train / {len(X_test)} test samples\n")
    return X_train, X_test, y_train, y_test, scaler


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
    print(f"  MSE  : {mse:.3f} m²/s²")
    print(f"  RMSE : {rmse:.3f} m/s\n")
    print(f"  {'Actual':>10}  {'Predicted':>10}  {'Error':>8}")
    for a, p in zip(y_test[:6], y_pred[:6]):
        print(f"  {a:>10.2f}  {p:>10.2f}  {a - p:>+8.2f} m/s")
    if model_name:
        save_metrics(model_name, mae=mae, mse=mse, rmse=rmse)
