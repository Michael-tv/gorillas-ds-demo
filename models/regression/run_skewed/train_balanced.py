"""
BALANCED model — the control for the skewed demo. Same architecture, same
number of rows, but drawn from the full launch-angle range instead of a
low-angle slice.

Predictions at a steep angle come from interpolation rather than extrapolation,
so this model stays accurate exactly where the skewed one collapses. Run both
and compare their holdout MAE: that difference is attributable to the training
distribution and nothing else, which is only true because three things are held
equal (AUDIT.md task 11):

  * **Same row count.** It reads how many rows the skewed run trains on and
    matches it. It used to hardcode 10,000 against the skewed run's 8,525, so
    part of any difference was simply "one saw more data".
  * **Same split.** `test_size` comes from params.yaml rather than a hardcoded
    0.2, so sweeping it moves both runs together.
  * **Same pipeline.** Feature engineering happens in FEATURE_STEP, as it does
    for the skewed model, rather than at load time here and in the Pipeline
    there.
"""
import os
import sys

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_utils import (FEATURE_STEP, load_full_pool, load_holdout, model_path,
                         print_metrics, save_metrics, skewed_row_count)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import numpy as np
import params
import splitting

MAX_ANGLE = params.load_params()["skew"]["max_angle_deg"]
TEST_SIZE = params.load_params()["test_size"]

# Matched to the skewed run rather than chosen, so the comparison isolates the
# training distribution. The pool is pre-shuffled, so a prefix is a fair sample.
N_SAMPLES = skewed_row_count()
print(f"Sized to match the skewed run : {N_SAMPLES:,} rows")

X, y, groups = load_full_pool(n_samples=N_SAMPLES)
X_train, X_test, y_train, y_test, groups_train = splitting.split(
    X, y, groups, test_size=TEST_SIZE, random_state=42)

print(f"Train set size (balanced) : {len(X_train):,}  (full angle range)")
print(f"Test  set size            : {len(X_test):,}")

model = Pipeline([FEATURE_STEP,
                  ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
model.fit(X_train, y_train)

print("\nRandom Forest (balanced) -- train set")
print_metrics("balanced", y_train, model.predict(X_train))

print("Random Forest (balanced) -- test set, IN-distribution")
print_metrics("balanced", y_test, model.predict(X_test))
mae_in = mean_absolute_error(y_test, model.predict(X_test))

# Scored on the same holdout the skewed model is scored on, so the two numbers
# are directly comparable. This model saw those angles in training, so it should
# be close to its in-distribution figure -- that is the contrast.
X_ood, y_ood, _ = load_holdout()
pred_ood = model.predict(X_ood)
mae_ood  = mean_absolute_error(y_ood, pred_ood)

print(f"Random Forest (balanced) -- same holdout as the skewed model "
      f"(launch_angle_deg > {MAX_ANGLE}, n={len(X_ood):,})")
print(f"  MAE  : {mae_ood:.3f} m/s")
print(f"  RMSE : {np.sqrt(np.mean((y_ood - pred_ood) ** 2)):.3f} m/s")

ratio = (mae_ood / mae_in) if mae_in else float("nan")
print(f"\n  In-distribution MAE     : {mae_in:.3f} m/s")
print(f"  Out-of-distribution MAE : {mae_ood:.3f} m/s   ({ratio:.1f}x)")
print("  Compare this ratio with the skewed model's. Same architecture, same row\n"
      "  count, same split -- only the training distribution differs.")

# Written last, deliberately overwriting the mae/mse/rmse that print_metrics
# records, so the file carries the comparison rather than just the last split.
save_metrics("balanced", mae=mae_in, mae_holdout=mae_ood, holdout_mae_ratio=ratio,
             n_samples=float(N_SAMPLES))

joblib.dump(model, model_path("model_balanced.joblib"))
print("\nSaved model_balanced.joblib")
