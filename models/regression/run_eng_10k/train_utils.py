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

# Valid physical ranges for each feature and the target.
# None means no bound on that side.
FEATURE_RANGES = {
    "launch_angle_deg":    (0,    90),
    "wind_x_ms":           (None, None),  # signed — headwind is negative
    "drag_param":          (0,    None),  # (0.5*rho*Cd*pi*r²)/mass — must be positive
    "height_diff_m":       (None, None),  # landing − launch height, can be negative
    "landing_distance_m":  (0,    None),
    "initial_velocity_ms": (0,    None),
}


def model_path(filename):
    os.makedirs(MODELS_DIR, exist_ok=True)
    name = os.environ.get("TRAIN_MODEL_NAME", "")
    if name:
        return os.path.join(MODELS_DIR, f"model_{name}.joblib")
    return os.path.join(MODELS_DIR, filename)


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


def load_data(test_size=0.2, scale=False):
    if not os.path.isfile(DATA):
        print()
        print(f"  [error] Training data not found: {DATA}")
        print(  "  Run:    python generate_data.py")
        print()
        sys.exit(1)
    df = pd.read_csv(DATA)
    n_orig = len(df)
    clean  = os.environ.get("TRAIN_CLEAN", "")
    if clean == "iqr":
        df = _clean_iqr(df)
        print(f"  IQR cleaning   : {n_orig} → {len(df)} rows ({n_orig - len(df)} removed)")
    elif clean == "range":
        df = _clean_range(df)
        print(f"  Range cleaning : {n_orig} → {len(df)} rows ({n_orig - len(df)} removed)")
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


def print_metrics(y_test, y_pred):
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    print(f"  MAE  : {mae:.3f} m/s")
    print(f"  MSE  : {mse:.3f} m²/s²")
    print(f"  RMSE : {rmse:.3f} m/s\n")
    print(f"  {'Actual':>10}  {'Predicted':>10}  {'Error':>8}")
    for a, p in zip(y_test[:6], y_pred[:6]):
        print(f"  {a:>10.2f}  {p:>10.2f}  {a - p:>+8.2f} m/s")
    model_name = os.environ.get("TRAIN_MODEL_NAME", "")
    if not model_name:
        script = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        model_name = script[6:] if script.startswith("train_") else script
    save_metrics(model_name, mae=mae, mse=mse, rmse=rmse)
