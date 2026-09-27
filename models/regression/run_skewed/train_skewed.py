"""
SKEWED model — random forest trained on a filtered slice of the standard
pool: only rows with launch_angle_deg <= 30 (see train_utils.MAX_ELEVATION_DEG).

Training data is concentrated at low launch angles. When the model is
asked to predict at 40° (outside its training distribution) it
extrapolates poorly, producing a biased velocity estimate.
"""
import os
import sys
import joblib
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_utils import load_data, model_path, print_metrics

X_train, X_test, y_train, y_test, groups_train = load_data()

print(f"Train set size (skewed) : {len(X_train):,}  (elevation 5–30°)")
print(f"Test  set size          : {len(X_test):,}")

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("\nRandom Forest (skewed) -- train set")
print_metrics(y_train, model.predict(X_train))

print("Random Forest (skewed) -- test set")
print_metrics(y_test, model.predict(X_test), model_name="skewed")
print("NOTE: test MAE looks fine in-distribution but model extrapolates poorly beyond 30° elevation.")

joblib.dump(model, model_path("model_skewed.joblib"))
print("Saved model_skewed.joblib")
