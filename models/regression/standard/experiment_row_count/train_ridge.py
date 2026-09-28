"""Train Ridge Regression at increasing sample sizes, to show how test
performance improves with more data.

TIERS are nested prefix slices of one seeded, pre-shuffled pool
(data/standard_velocity.parquet, 50000 rows) -- growing the sample size is
the only thing that changes between tiers, so a later tier's extra rows are
exactly the earlier tier's rows plus more, not a different draw. Same
principle as params.yaml's n_samples convergence knob, swept here in one
script instead of requiring `dvc exp run --set-param n_samples=...` once per
tier. See ../experiment_skew/train_linear_regression.py for the general
experiment_<name>/train_<model>.py pattern.

Writes one row per tier to metrics_<key>.csv (n_samples, mae, mse, rmse) and
one model_<key>_n<size>.joblib per tier, so every tier's model stays
inspectable, not just the metrics curve.
"""
import csv
import os

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error

import params
import splitting
from feature_engineering import EngineeredFeatures
from models.regression.algorithms import ridge as algo
from models.regression.common import loader

_REPO_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DATA        = os.path.join(_REPO_ROOT, "data", "standard_velocity.parquet")
MODELS_DIR  = os.path.join(_REPO_ROOT, "experiments_results", "regression", "standard", "experiment_row_count", "models")
KEY         = "ridge"
TIERS       = [1000, 2000, 5000, 10000, 20000, 50000]


def main():
    X_full, y_full, groups_full = loader.load_data(DATA)
    search = params.search_params("regression", KEY)
    test_size = params.load_params()["test_size"]

    os.makedirs(MODELS_DIR, exist_ok=True)
    rows = []
    for n in TIERS:
        X, y, groups = X_full.iloc[:n], y_full[:n], groups_full[:n]
        print(f"\n=== {algo.NAME} -- n_samples={n} ===")
        X_train, X_test, y_train, y_test, groups_train = splitting.split(
            X, y, groups, test_size=test_size, random_state=42)
        cv_folds = splitting.cv_for(search["cv"], X_train, y_train, groups_train)

        feature_step = ("features", EngineeredFeatures(output_columns=loader.FEATURES))
        model = algo.fit(X_train, y_train, feature_step, {"n_iter": search["n_iter"], "cv": cv_folds})
        y_pred = model.predict(X_test)
        mae  = mean_absolute_error(y_test, y_pred)
        mse  = float(np.mean((y_test - y_pred) ** 2))
        rmse = mse ** 0.5
        print(f"  MAE={mae:.3f}  MSE={mse:.3f}  RMSE={rmse:.3f}")
        rows.append({"n_samples": n, "mae": mae, "mse": mse, "rmse": rmse})

        joblib.dump(model, os.path.join(MODELS_DIR, f"model_{KEY}_n{n}.joblib"))

    metrics_path = os.path.join(MODELS_DIR, f"metrics_{KEY}.csv")
    with open(metrics_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_samples", "mae", "mse", "rmse"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved metrics_{KEY}.csv ({len(rows)} tiers) and {len(rows)} models to {MODELS_DIR}")


if __name__ == "__main__":
    main()
