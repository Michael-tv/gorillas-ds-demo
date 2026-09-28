"""SKEWED model -- a random forest trained on a low-angle slice of the
standard pool: only rows with launch_angle_deg <= params.yaml's
skew.max_angle_deg, produced by pipelines/standard/dvc.yaml's filter_skewed
stage.

The lesson is extrapolation. Training data is concentrated at low launch
angles, so when the model is asked about a steep shot it has never seen, it
cannot extrapolate -- a random forest flatlines at its boundary leaf mean,
producing a confidently wrong velocity.

Reports MAE on BOTH test sets, which is the whole point (AUDIT.md task 10):

  * the in-distribution split, drawn from the same filtered slice -- where
    the model looks fine;
  * data/skewed_holdout.parquet, the rows the filter EXCLUDED -- where it
    does not.

See train_balanced_concept.py (this folder) for the control -- same
architecture, full angle range instead.
"""
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

import params
import splitting
from feature_engineering import EngineeredFeatures
from models.regression.common import loader

_REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DATA         = os.path.join(_REPO_ROOT, "data", "skewed_training_data.parquet")
HOLDOUT_DATA = os.path.join(_REPO_ROOT, "data", "skewed_holdout.parquet")
MODELS_DIR   = os.path.join(_REPO_ROOT, "experiments_results", "regression", "standard", "experiment_skew", "models")
ENG_FEATURES = ["launch_angle_deg", "wind_x_ms", "drag_param", "height_diff_m", "landing_distance_m"]


def main():
    max_angle = params.load_params()["skew"]["max_angle_deg"]
    X, y, groups = loader.load_data(DATA)
    X_train, X_test, y_train, y_test, groups_train = splitting.split(
        X, y, groups, test_size=params.load_params()["test_size"], random_state=42)

    print(f"Train set size (skewed) : {len(X_train):,}  (launch_angle_deg <= {max_angle})")
    print(f"Test  set size          : {len(X_test):,}   (same slice -- in-distribution)")

    feature_step = ("features", EngineeredFeatures(output_columns=ENG_FEATURES))
    model = Pipeline([feature_step, ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
    model.fit(X_train, y_train)

    print("\nRandom Forest (skewed) -- test set, IN-distribution")
    mae_in = mean_absolute_error(y_test, model.predict(X_test))
    print(f"  MAE : {mae_in:.3f} m/s")

    X_ood, y_ood, _ = loader.load_data(HOLDOUT_DATA)
    pred_ood = model.predict(X_ood)
    mae_ood  = mean_absolute_error(y_ood, pred_ood)

    print(f"\nRandom Forest (skewed) -- holdout, OUT-of-distribution "
          f"(launch_angle_deg > {max_angle}, n={len(X_ood):,})")
    print(f"  MAE  : {mae_ood:.3f} m/s")
    print(f"  RMSE : {np.sqrt(np.mean((y_ood - pred_ood) ** 2)):.3f} m/s")

    ratio = (mae_ood / mae_in) if mae_in else float("nan")
    print(f"\n  In-distribution MAE     : {mae_in:.3f} m/s")
    print(f"  Out-of-distribution MAE : {mae_ood:.3f} m/s   ({ratio:.1f}x worse)")
    print("  The model looks fine on a test set that shares its blind spot, and is\n"
          f"  {ratio:.1f}x worse on the angles it never saw. That gap is the demo -- not\n"
          "  the in-distribution number.")

    loader.save_metrics(MODELS_DIR, "skewed", mae=mae_in, mae_holdout=mae_ood,
                        holdout_mae_ratio=ratio, max_angle_deg=float(max_angle))
    joblib.dump(model, loader.model_path(MODELS_DIR, "model_skewed.joblib"))
    print("\nSaved model_skewed.joblib")


if __name__ == "__main__":
    main()
