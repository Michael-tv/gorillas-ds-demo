"""
BALANCED model — random forest trained on data spanning the full
elevation range (5–85°).

Same architecture as the skewed model. Predictions at 40° elevation
come from interpolation rather than extrapolation, so accuracy is much
higher at angles the skewed model has never seen.
"""
import os
import sys
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_utils import model_path, print_metrics, FEATURES, TARGET

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, REPO_ROOT)
import params
from feature_engineering import add_engineered_columns

DATA      = params.data_path()
N_SAMPLES = 10_000  # same size as the skewed comparison set

if not os.path.isfile(DATA):
    print()
    print(f"  [error] Balanced training data not found: {DATA}")
    print("  Run:    dvc repro generate")
    print()
    sys.exit(1)

df = pd.read_parquet(DATA).iloc[:N_SAMPLES]
df = add_engineered_columns(df)
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

joblib.dump(model, model_path("model_balanced.joblib"))
print("Saved model_balanced.joblib")
