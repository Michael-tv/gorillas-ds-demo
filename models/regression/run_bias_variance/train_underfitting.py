"""
UNDERFITTING (high bias) -- DecisionTree with max_depth=1.

One split, one threshold. The model cannot capture the non-linear
physics of projectile motion. Both train MAE and test MAE are high
and close together -- the classic high-bias signature.
"""
import os
import sys
import joblib
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_utils import load_data, model_path, print_metrics

DEPTH = 1

X_train, X_test, y_train, y_test = load_data()

print(f"Train set size : {len(X_train):,}")
print(f"Test  set size : {len(X_test):,}")
print(f"max_depth      : {DEPTH}  (underfitting)")

model = DecisionTreeRegressor(max_depth=DEPTH, random_state=42)
model.fit(X_train, y_train)

print(f"\nDecision Tree (depth={DEPTH}) -- train set")
print_metrics(y_train, model.predict(X_train))

print(f"Decision Tree (depth={DEPTH}) -- test set")
print_metrics(y_test, model.predict(X_test), model_name="underfitting")
print("NOTE: train MAE ~= test MAE, both high -- model too simple to fit the data.")

joblib.dump(model, model_path("model_underfitting.joblib"))
print("Saved model_underfitting.joblib")
