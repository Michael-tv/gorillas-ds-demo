"""Worst-case overfitting demo, on the velocity pool's 5 engineered columns.

Fits on the entire dataset with no train/test split, sweeping a param grid
and keeping whichever combination gets the lowest TRAINING error -- shows
what a fully-grown tree with zero validation looks like.
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
DATA         = os.path.join(_REPO_ROOT, "data", "gorillas_velocity.parquet")
MODELS_DIR   = os.path.join(_REPO_ROOT, "experiments", "regression", "velocity", "experiment_eng", "results", "models")
METRICS_DIR  = os.path.join(_REPO_ROOT, "experiments", "regression", "velocity", "experiment_eng", "results", "metrics")
KEY          = "decision_tree_overfit"
ENG_FEATURES = ["launch_angle_deg", "wind_x_ms", "drag_param", "height_diff_m", "landing_distance_m"]  # the 5 engineered columns

PARAM_GRID = {
    "max_depth":         [8, 10, 12, 15, 20, 25, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf":  [1, 2, 4, 8],
}


def main():
    X_raw, y, _groups = loader.load_data(DATA)
    X = EngineeredFeatures(output_columns=ENG_FEATURES).fit_transform(X_raw)

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
