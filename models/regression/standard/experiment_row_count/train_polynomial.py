"""Train Polynomial Regression at increasing sample sizes, to show how test
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
import glob
import os
import re

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error

import params
import splitting
from feature_engineering import EngineeredFeatures
from models.regression.algorithms import polynomial as algo
from models.regression.common import loader

_REPO_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DATA        = os.path.join(_REPO_ROOT, "data", "standard_velocity.parquet")
EXPERIMENT_DIR = os.path.join(_REPO_ROOT, "experiments_results", "regression", "standard", "experiment_row_count")
MODELS_DIR  = os.path.join(EXPERIMENT_DIR, "models")
KEY         = "polynomial"
TIERS_DIR   = os.path.join(EXPERIMENT_DIR, "tiers", KEY)
TIERS       = params.load_experiment_params(__file__)["tiers"]


def _clean_stale_tiers():
    """Delete any model_n<size>.joblib in TIERS_DIR whose size is no longer in
    TIERS -- this is what makes tiers/<key>/ (a single directory dvc.yaml
    outs: entry) correctly shrink when a tier is removed from params.yaml,
    instead of leaving the old tier's model behind as an orphan."""
    os.makedirs(TIERS_DIR, exist_ok=True)
    for path in glob.glob(os.path.join(TIERS_DIR, "model_n*.joblib")):
        m = re.fullmatch(r"model_n(\d+)\.joblib", os.path.basename(path))
        if m and int(m.group(1)) not in TIERS:
            os.remove(path)
            print(f"  Removed stale {os.path.basename(path)} (no longer in TIERS)")


def main():
    X_full, y_full, groups_full = loader.load_data(DATA, clean="no_outlier")
    search = params.search_params("regression", KEY)
    test_size = params.load_params()["test_size"]

    os.makedirs(MODELS_DIR, exist_ok=True)
    _clean_stale_tiers()
    rows = []
    for n in TIERS:
        X, y, groups = X_full.iloc[:n], y_full[:n], groups_full[:n]
        print(f"\n=== {algo.NAME} -- n_samples={n} ===")
        X_train, X_test, y_train, y_test, groups_train = splitting.split(
            X, y, groups, test_size=test_size, random_state=42)
        cv_folds = splitting.single_split_cv()

        feature_step = ("features", EngineeredFeatures(output_columns=loader.FEATURES))
        model = algo.fit(X_train, y_train, feature_step, {"n_iter": search["n_iter"], "cv": cv_folds})
        y_pred = model.predict(X_test)
        mae  = mean_absolute_error(y_test, y_pred)
        mse  = float(np.mean((y_test - y_pred) ** 2))
        rmse = mse ** 0.5
        print(f"  MAE={mae:.3f}  MSE={mse:.3f}  RMSE={rmse:.3f}")
        rows.append({"n_samples": n, "mae": mae, "mse": mse, "rmse": rmse})

        joblib.dump(model, os.path.join(TIERS_DIR, f"model_n{n}.joblib"))

    metrics_path = os.path.join(MODELS_DIR, f"metrics_{KEY}.csv")
    with open(metrics_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_samples", "mae", "mse", "rmse"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved metrics_{KEY}.csv ({len(rows)} tiers) and {len(rows)} models to {TIERS_DIR}")


if __name__ == "__main__":
    main()
