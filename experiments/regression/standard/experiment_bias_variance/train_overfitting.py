"""OVERFITTING (high variance) -- DecisionTree with max_depth=None (fully grown).

The tree splits until every leaf holds one sample -- zero training error.
Test MAE is much higher than train MAE because the model memorised noise
rather than the underlying physics. See train_underfitting.py (this folder)
for the high-bias contrast.
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
from experiments.regression.common import loader

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DATA       = os.path.join(_REPO_ROOT, "data", "standard_velocity.parquet")
MODELS_DIR = os.path.join(_REPO_ROOT, "experiments", "regression", "standard", "experiment_bias_variance", "results", "models")
METRICS_DIR = os.path.join(_REPO_ROOT, "experiments", "regression", "standard", "experiment_bias_variance", "results", "metrics")
FEATURES   = ["launch_angle_deg", "wind_x_ms", "drag_param", "height_diff_m", "landing_distance_m"]
TARGET     = "initial_velocity_ms"


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
    print("max_depth      : None  (overfitting)")

    model = DecisionTreeRegressor(max_depth=None, random_state=42)
    model.fit(X_train, y_train)

    print("\nDecision Tree (depth=None) -- train set")
    _print_only(y_train, model.predict(X_train), "Train")

    print("Decision Tree (depth=None) -- test set")
    loader.print_metrics(METRICS_DIR, "overfitting", y_test, model.predict(X_test))
    print("NOTE: train MAE is 0 (memorisation) but test MAE is much higher -- overfitting.")

    joblib.dump(model, loader.model_path(MODELS_DIR, "model_overfitting.joblib"))
    print("Saved model_overfitting.joblib")


if __name__ == "__main__":
    main()
