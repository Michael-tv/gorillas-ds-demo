"""
SKEWED model — a random forest trained on a low-angle slice of the standard
pool: only rows with launch_angle_deg <= params.yaml's skew.max_angle_deg,
produced by the `filter_skewed` stage.

The lesson is extrapolation. Training data is concentrated at low launch angles,
so when the model is asked about a steep shot it has never seen, it cannot
extrapolate -- a random forest flatlines at its boundary leaf mean, producing a
confidently wrong velocity.

This script reports MAE on BOTH test sets, which is the whole point (AUDIT.md
task 10):

  * the in-distribution split, drawn from the same filtered slice -- where the
    model looks fine;
  * data/skewed_holdout.parquet, the rows the filter EXCLUDED -- where it does
    not.

Until the filter became its own stage, the excluded rows were dropped at load
time and no stage ever evaluated above the cap. The demo asserted "extrapolates
poorly beyond 30 degrees" while measuring nothing of the kind. The gap between
the two numbers below is the demo.
"""
import os
import sys

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_utils import (FEATURE_STEP, load_data, load_holdout, model_path,
                         print_metrics, save_metrics)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import params
import splitting

MAX_ANGLE = params.load_params()["skew"]["max_angle_deg"]

X, y, groups = load_data()
X_train, X_test, y_train, y_test, groups_train = splitting.split(
    X, y, groups, test_size=params.load_params()["test_size"], random_state=42)

print(f"Train set size (skewed) : {len(X_train):,}  (launch_angle_deg <= {MAX_ANGLE})")
print(f"Test  set size          : {len(X_test):,}   (same slice -- in-distribution)")

# Engineering inside the Pipeline, so the saved model takes raw physical inputs
# and cannot be fed a differently-derived feature set at serving time.
model = Pipeline([FEATURE_STEP,
                  ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
model.fit(X_train, y_train)

print("\nRandom Forest (skewed) -- train set")
print_metrics("skewed", y_train, model.predict(X_train))

print("Random Forest (skewed) -- test set, IN-distribution")
print_metrics("skewed", y_test, model.predict(X_test))
mae_in = mean_absolute_error(y_test, model.predict(X_test))

# The claim, measured.
X_ood, y_ood, _ = load_holdout()
pred_ood = model.predict(X_ood)
mae_ood  = mean_absolute_error(y_ood, pred_ood)

print(f"Random Forest (skewed) -- holdout, OUT-of-distribution "
      f"(launch_angle_deg > {MAX_ANGLE}, n={len(X_ood):,})")
print(f"  MAE  : {mae_ood:.3f} m/s")
print(f"  RMSE : {np.sqrt(np.mean((y_ood - pred_ood) ** 2)):.3f} m/s")

ratio = (mae_ood / mae_in) if mae_in else float("nan")
print(f"\n  In-distribution MAE     : {mae_in:.3f} m/s")
print(f"  Out-of-distribution MAE : {mae_ood:.3f} m/s   ({ratio:.1f}x worse)")
print(f"  The model looks fine on a test set that shares its blind spot, and is "
      f"{ratio:.1f}x worse\n  on the angles it never saw. That gap is the demo -- not the "
      f"in-distribution number.")

# Written last, deliberately overwriting the mae/mse/rmse that print_metrics
# records, so the file carries the in/out-of-distribution comparison rather than
# just whichever split was printed last. Both numbers then survive in
# dvc metrics diff / dvc exp show rather than only on stdout.
save_metrics("skewed", mae=mae_in, mae_holdout=mae_ood, holdout_mae_ratio=ratio,
             max_angle_deg=float(MAX_ANGLE))

joblib.dump(model, model_path("model_skewed.joblib"))
print("\nSaved model_skewed.joblib")
