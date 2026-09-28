"""UNDERFITTING (high bias) -- DecisionTree with max_depth=1.

One split, one threshold. The model cannot capture the non-linear physics of
projectile motion. Both train MAE and test MAE are high and close together --
the classic high-bias signature. See train_overfitting.py (this folder) for
the high-variance contrast.
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

import params
import splitting
from feature_engineering import add_engineered_columns
from models.regression.common import loader

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DATA       = os.path.join(_REPO_ROOT, "data", "standard_velocity.parquet")
MODELS_DIR = os.path.join(_REPO_ROOT, "experiments_results", "regression", "standard", "experiment_bias_variance", "models")
FEATURES   = ["launch_angle_deg", "wind_x_ms", "drag_param", "height_diff_m", "landing_distance_m"]
TARGET     = "initial_velocity_ms"
DEPTH      = 1


def _print_only(y_true, y_pred, label):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    print(f"  {label} MAE  : {mae:.3f} m/s")
    print(f"  {label} RMSE : {rmse:.3f} m/s")


def main():
    n_samples = params.load_params()["n_samples"]
    df = add_engineered_columns(pd.read_parquet(DATA))
    df = params.take_samples(df, n_samples).reset_index(drop=True)
    X, y = df[FEATURES].values, df[TARGET].values
    groups = df[loader.GROUP_COLUMN].to_numpy()
    X_train, X_test, y_train, y_test, groups_train = splitting.split(
        X, y, groups, test_size=params.load_params()["test_size"], random_state=42)

    print(f"Train set size : {len(X_train):,}")
    print(f"Test  set size : {len(X_test):,}")
    print(f"max_depth      : {DEPTH}  (underfitting)")

    model = DecisionTreeRegressor(max_depth=DEPTH, random_state=42)
    model.fit(X_train, y_train)

    print(f"\nDecision Tree (depth={DEPTH}) -- train set")
    _print_only(y_train, model.predict(X_train), "Train")

    print(f"Decision Tree (depth={DEPTH}) -- test set")
    loader.print_metrics(MODELS_DIR, "underfitting", y_test, model.predict(X_test))
    print("NOTE: train MAE ~= test MAE, both high -- model too simple to fit the data.")

    joblib.dump(model, loader.model_path(MODELS_DIR, "model_underfitting.joblib"))
    print("Saved model_underfitting.joblib")


if __name__ == "__main__":
    main()
