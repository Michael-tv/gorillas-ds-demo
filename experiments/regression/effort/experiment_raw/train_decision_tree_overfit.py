"""Worst-case overfitting demo, on the effort pool's 9 raw columns.

Converted from experiments/regression/train_decision_tree_overfit.py (--run raw
--key decision_tree_overfit) to this experiment_raw/ folder's standalone-
script pattern -- see ../experiment_skew/train_linear_regression.py for the
general rationale.

Every other model here trains on X_train and scores on held-out X_test; this
one deliberately does neither -- it fits on the ENTIRE dataset with no split
at all, sweeping a param grid and keeping whichever combination gets the
lowest TRAINING error, to show what a fully-grown tree with zero validation
looks like.
"""
import itertools
import os

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

import params
from feature_engineering import EngineeredFeatures
from experiments.regression.common import loader

_REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DATA         = os.path.join(_REPO_ROOT, "data", "gorillas_effort.parquet")
MODELS_DIR   = os.path.join(_REPO_ROOT, "experiments", "regression", "effort", "experiment_raw", "results", "models")
METRICS_DIR  = os.path.join(_REPO_ROOT, "experiments", "regression", "effort", "experiment_raw", "results", "metrics")
KEY          = "decision_tree_overfit"
RAW_FEATURES = loader.FEATURES  # the 9 raw columns, as-is

PARAM_GRID = {
    "max_depth":         [8, 10, 12, 15, 20, 25, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf":  [1, 2, 4, 8],
}


def main():
    X_raw, y, _groups = loader.load_data(DATA)
    X = EngineeredFeatures(output_columns=RAW_FEATURES).fit_transform(X_raw)

    # Evaluate every param combination on training data -- no validation whatsoever
    best_rmse, best_params, best_model = float("inf"), None, None
    keys   = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))

    for values in combos:
        m = DecisionTreeRegressor(**dict(zip(keys, values)), random_state=42)
        m.fit(X, y)
        rmse = np.sqrt(np.mean((y - m.predict(X)) ** 2))
        if rmse < best_rmse:
            best_rmse, best_params, best_model = rmse, dict(zip(keys, values)), m

    print(f"Best params (by train RMSE): {best_params}")
    print(f"Best train RMSE            : {best_rmse:.3f} m/s  ({len(combos)} combos tried)\n")

    y_pred = best_model.predict(X)
    mae    = mean_absolute_error(y, y_pred)
    mse    = np.mean((y - y_pred) ** 2)
    rmse   = np.sqrt(mse)

    print(f"Decision Tree overfit (n={len(X)}, scored on training data only)")
    print(f"  Train MAE  : {mae:.3f} m/s")
    print(f"  Train MSE  : {mse:.3f} m^2/s^2")
    print(f"  Train RMSE : {rmse:.3f} m/s")
    print("  (no held-out test set -- worst-case overfitting)")

    joblib.dump(best_model, loader.model_path(MODELS_DIR, f"model_{KEY}.joblib"))
    print(f"Saved model_{KEY}.joblib")
    loader.save_metrics(METRICS_DIR, KEY, mae=mae, mse=mse, rmse=rmse)


if __name__ == "__main__":
    main()
