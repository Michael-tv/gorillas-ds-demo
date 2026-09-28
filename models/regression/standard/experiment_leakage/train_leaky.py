"""LEAKY pipeline -- decision tree trained on contaminated data.

The test rows are accidentally included in the training set (e.g. feature
engineering was run on the full dataset and saved before the train/test
split was applied).

A fully-grown tree memorises every training example, so because the test
rows are also in the training set the reported test MAE collapses to zero --
a completely fraudulent result. See train_clean.py (this folder) for the
uncontaminated control.
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
MODELS_DIR = os.path.join(_REPO_ROOT, "experiments_results", "regression", "standard", "experiment_leakage", "models")
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

    X_train_leaky = np.vstack([X_train, X_test])
    y_train_leaky = np.concatenate([y_train, y_test])

    print(f"Train set size (leaky) : {len(X_train_leaky):,}  ({len(X_test):,} test rows included)")
    print(f"Test  set size         : {len(X_test):,}")

    model = DecisionTreeRegressor(max_depth=None, random_state=42)
    model.fit(X_train_leaky, y_train_leaky)

    print("\nDecision Tree (leaky) -- train set")
    _print_only(y_train_leaky, model.predict(X_train_leaky), "Train")

    print("Decision Tree (leaky) -- test set")
    loader.print_metrics(MODELS_DIR, "dt_leaky", y_test, model.predict(X_test))
    print("NOTE: near-zero test MAE is fraudulent -- model memorised the test rows.")

    joblib.dump(model, loader.model_path(MODELS_DIR, "model_dt_leaky.joblib"))
    print("Saved model_dt_leaky.joblib")


if __name__ == "__main__":
    main()
