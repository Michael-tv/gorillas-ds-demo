"""Worst-case overfitting demo -- kept standalone, not folded into
models/regression/train.py's generic algorithm harness.

Every other algorithm trains on X_train and scores on a held-out X_test;
this one deliberately does neither -- it fits on the ENTIRE dataset with no
split at all, and reports training-set error, to show what a fully-grown
tree with zero validation looks like. Forcing that into a harness built
around "split, cross-validate, score on test" would either quietly break the
one thing this script exists to demonstrate, or make the harness worse for
every other algorithm to accommodate one exception -- so it keeps its own
`main()`, importing run config from models/regression/runs.py instead of an
injected train_utils (AUDIT.md C1).

While rewriting the imports, also fixed two bugs the original audit found in
this file (AUDIT.md task 6): it read the parquet directly, bypassing
loader.load_data() entirely, so --clean was accepted on the command line and
silently ignored, and the pool never went through the 14-column contract
check on read. Both now go through the same loader every other run uses.
"""
import argparse
import itertools

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

from feature_engineering import add_engineered_columns
from models.regression.common import loader
from models.regression.runs import RUNS

PARAM_GRID = {
    "max_depth":         [8, 10, 12, 15, 20, 25, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf":  [1, 2, 4, 8],
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, choices=sorted(RUNS))
    p.add_argument("--key", required=True)
    p.add_argument("--clean", default="", choices=["", "iqr", "range", "no_outlier"])
    args = p.parse_args()

    cfg = RUNS[args.run]

    # No split: load_data returns the raw columns for the WHOLE pool (no
    # train/test division), which is the point of this demo.
    X_raw, y, _groups = loader.load_data(cfg.data, n_samples=cfg.n_samples, clean=args.clean)
    X = add_engineered_columns(X_raw)[cfg.features].values

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

    joblib.dump(best_model, loader.model_path(cfg.models_dir, f"model_{args.key}.joblib"))
    print(f"Saved model_{args.key}.joblib")
    loader.save_metrics(cfg.models_dir, args.key, mae=mae, mse=mse, rmse=rmse)


if __name__ == "__main__":
    main()
