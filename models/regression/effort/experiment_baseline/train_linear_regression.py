"""Train Linear Regression on a fixed 1000-row, no-outlier baseline slice of
the effort pool -- the simplest possible run on this pool: a default,
unmodified LinearRegression().fit(), no hyperparameter search, no CV, no
sweep.

Cleaning runs over the full pool before slicing, not the other way around,
so n_samples means 1000 clean rows, not 1000 raw rows that shrink once
outlier rows are dropped -- same clean-then-slice order as
experiment_row_count/train_linear_regression.py, just at one fixed size
instead of a sweep of tiers. See ../experiment_raw/train_linear_regression.py
for the general experiment_<name>/train_<model>.py pattern this follows.

Also reports train_mae and variance_proxy (= mae - train_mae) alongside the
usual test mae/mse/rmse -- a cheap bias/variance proxy (train_mae close to
test mae means bias-, not variance-, limited; a wide variance_proxy gap
means the opposite). Not a real decomposition -- see
experiment_row_count/train_linear_regression.py's docstring for why.
Computed inline here rather than via loader.print_metrics, since that only
knows about the three test-set metrics; loader.save_metrics is still used
to write the CSV, since it takes free-form columns.
"""
import os

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error

import params
import splitting
from feature_engineering import EngineeredFeatures
from models.regression.algorithms import linear_regression as algo
from models.regression.common import loader

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DATA       = os.path.join(_REPO_ROOT, "data", "gorillas_effort.parquet")
MODELS_DIR = os.path.join(_REPO_ROOT, "experiments_results", "regression", "effort", "experiment_baseline", "models")
KEY        = "linear_regression"


def main():
    n_samples = params.load_experiment_params(__file__)["n_samples"]
    X_full, y_full, groups_full = loader.load_data(DATA, clean="no_outlier")
    X, y, groups = X_full.iloc[:n_samples], y_full[:n_samples], groups_full[:n_samples]

    X_train, X_test, y_train, y_test, _ = splitting.split(
        X, y, groups, test_size=params.load_params()["test_size"], random_state=42)

    feature_step = ("features", EngineeredFeatures(output_columns=loader.FEATURES))
    print(f"{algo.NAME} -- experiment_baseline (n_samples={n_samples})\n")
    model = algo.fit(X_train, y_train, feature_step, {})

    y_pred_test = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred_test)
    mse  = float(np.mean((y_test - y_pred_test) ** 2))
    rmse = mse ** 0.5
    train_mae = mean_absolute_error(y_train, model.predict(X_train))
    variance_proxy = mae - train_mae

    print(f"{algo.NAME} -- test set")
    print(f"  MAE  : {mae:.3f} m/s")
    print(f"  MSE  : {mse:.3f} m^2/s^2")
    print(f"  RMSE : {rmse:.3f} m/s")
    print(f"  Train MAE      : {train_mae:.3f} m/s  (bias proxy)")
    print(f"  Variance proxy : {variance_proxy:.3f} m/s  (mae - train_mae)")

    loader.save_metrics(MODELS_DIR, KEY, mae=mae, mse=mse, rmse=rmse,
                         train_mae=train_mae, variance_proxy=variance_proxy)

    joblib.dump(model, loader.model_path(MODELS_DIR, f"model_{KEY}.joblib"))
    print(f"Saved model_{KEY}.joblib")


if __name__ == "__main__":
    main()
