"""
OVERFITTING (high variance) -- DecisionTree with max_depth=None (fully grown).

The tree splits until every leaf holds one sample -- zero training error.
Test MAE is much higher than train MAE because the model memorised
noise rather than the underlying physics.
"""
import os
import sys
import joblib
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_utils import load_data, model_path, print_metrics

X_train, X_test, y_train, y_test = load_data()

print(f"Train set size : {len(X_train):,}")
print(f"Test  set size : {len(X_test):,}")
print(f"max_depth      : None  (overfitting)")

model = DecisionTreeRegressor(max_depth=None, random_state=42)
model.fit(X_train, y_train)

print("\nDecision Tree (depth=None) -- train set")
print_metrics(y_train, model.predict(X_train))

print("Decision Tree (depth=None) -- test set")
print_metrics(y_test, model.predict(X_test), model_name="overfitting")
print("NOTE: train MAE is 0 (memorisation) but test MAE is much higher -- overfitting.")

joblib.dump({"model": model}, model_path("model_overfitting.joblib"))
print("Saved model_overfitting.joblib")
