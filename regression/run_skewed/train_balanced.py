"""
BALANCED model — random forest trained on data spanning the full
elevation range (5–85°).

Same architecture as the skewed model. Predictions at 40° elevation
come from interpolation rather than extrapolation, so accuracy is much
higher at angles the skewed model has never seen.
"""
import math
import os
import sys
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_utils import model_path, print_metrics, FEATURES, TARGET

RHO = 1.225  # kg/m³

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "run_raw_10k", "training_data", "training_data.csv")

if not os.path.isfile(DATA):
    print()
    print(f"  [error] Balanced training data not found: {DATA}")
    print("  Run:    python regression/run_raw_10k/generate_data.py")
    print()
    sys.exit(1)

df = pd.read_csv(DATA)
df["wind_x_ms"]    = df["wind_speed_ms"] * df["wind_direction_norm"]
df["drag_param"]   = (0.5 * RHO * df["drag_coeff"] * math.pi * df["radius_m"] ** 2) / df["mass_kg"]
df["height_diff_m"] = df["landing_height_m"] - df["launch_height_m"]
X = df[FEATURES].values
y = df[TARGET].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train set size (balanced) : {len(X_train):,}  (elevation 5–85°)")
print(f"Test  set size            : {len(X_test):,}")

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("\nRandom Forest (balanced) -- train set")
print_metrics(y_train, model.predict(X_train))

print("Random Forest (balanced) -- test set")
print_metrics(y_test, model.predict(X_test), model_name="balanced")

joblib.dump({"model": model}, model_path("model_balanced.joblib"))
print("Saved model_balanced.joblib")
