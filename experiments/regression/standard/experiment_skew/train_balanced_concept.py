"""BALANCED model -- the control for the skewed demo (train_skewed_concept.py,
this folder). Same architecture, same number of rows, but drawn from the full
launch-angle range instead of a low-angle slice.

Predictions at a steep angle come from interpolation rather than
extrapolation, so this model stays accurate exactly where the skewed one
collapses. Three things are held equal so the holdout MAE difference is
attributable to the training distribution and nothing else (AUDIT.md task 11):
same row count (matched to the skewed run below), same test_size, same
feature engineering (EngineeredFeatures, as the skewed model uses).
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

import params
import splitting
from feature_engineering import EngineeredFeatures
from experiments.regression.common import loader

_REPO_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SKEWED_DATA   = os.path.join(_REPO_ROOT, "data", "skewed_training_data.parquet")
HOLDOUT_DATA  = os.path.join(_REPO_ROOT, "data", "skewed_holdout.parquet")
FULL_POOL     = os.path.join(_REPO_ROOT, "data", "standard_velocity.parquet")
MODELS_DIR    = os.path.join(_REPO_ROOT, "experiments", "regression", "standard", "experiment_skew", "results", "models")
METRICS_DIR   = os.path.join(_REPO_ROOT, "experiments", "regression", "standard", "experiment_skew", "results", "metrics")
ENG_FEATURES  = ["launch_angle_deg", "wind_x_ms", "drag_param", "height_diff_m", "landing_distance_m"]


def main():
    max_angle = params.load_params()["skew"]["max_angle_deg"]
    test_size = params.load_params()["test_size"]

    # Matched to the skewed run rather than chosen, so the comparison isolates
    # the training distribution -- the pool is pre-shuffled, so a prefix is a
    # fair sample.
    n_samples = len(pd.read_parquet(SKEWED_DATA))
    print(f"Sized to match the skewed run : {n_samples:,} rows")

    X, y, groups = loader.load_data(FULL_POOL, n_samples=n_samples)
    X_train, X_test, y_train, y_test, groups_train = splitting.split(
        X, y, groups, test_size=test_size, random_state=42)

    print(f"Train set size (balanced) : {len(X_train):,}  (full angle range)")
    print(f"Test  set size            : {len(X_test):,}")

    feature_step = ("features", EngineeredFeatures(output_columns=ENG_FEATURES))
    model = Pipeline([feature_step, ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
    model.fit(X_train, y_train)

    print("\nRandom Forest (balanced) -- test set, IN-distribution")
    mae_in = mean_absolute_error(y_test, model.predict(X_test))
    print(f"  MAE : {mae_in:.3f} m/s")

    # Scored on the same holdout the skewed model is scored on, so the two
    # numbers are directly comparable.
    X_ood, y_ood, _ = loader.load_data(HOLDOUT_DATA)
    pred_ood = model.predict(X_ood)
    mae_ood  = mean_absolute_error(y_ood, pred_ood)

    print(f"\nRandom Forest (balanced) -- same holdout as the skewed model "
          f"(launch_angle_deg > {max_angle}, n={len(X_ood):,})")
    print(f"  MAE  : {mae_ood:.3f} m/s")
    print(f"  RMSE : {np.sqrt(np.mean((y_ood - pred_ood) ** 2)):.3f} m/s")

    ratio = (mae_ood / mae_in) if mae_in else float("nan")
    print(f"\n  In-distribution MAE     : {mae_in:.3f} m/s")
    print(f"  Out-of-distribution MAE : {mae_ood:.3f} m/s   ({ratio:.1f}x)")
    print("  Compare this ratio with the skewed model's. Same architecture, same row\n"
          "  count, same split -- only the training distribution differs.")

    loader.save_metrics(METRICS_DIR, "balanced", mae=mae_in, mae_holdout=mae_ood,
                        holdout_mae_ratio=ratio, n_samples=float(n_samples))
    joblib.dump(model, loader.model_path(MODELS_DIR, "model_balanced.joblib"))
    print("\nSaved model_balanced.joblib")


if __name__ == "__main__":
    main()
