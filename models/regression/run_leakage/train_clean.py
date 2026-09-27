"""
CLEAN pipeline — decision tree trained on uncontaminated data.

The test set is held out before any fitting. Same architecture as
the leaky model so the only difference is what the model trained on.
Test MAE here reflects genuine out-of-sample generalisation.
"""
import os
import sys
import joblib
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_utils import load_data, model_path, print_metrics

X_train, X_test, y_train, y_test, groups_train = load_data()

print(f"Train set size (clean) : {len(X_train):,}")
print(f"Test  set size         : {len(X_test):,}")

model = DecisionTreeRegressor(max_depth=None, random_state=42)
model.fit(X_train, y_train)

print("\nDecision Tree (clean) -- train set")
print_metrics(y_train, model.predict(X_train))

print("Decision Tree (clean) -- test set")
print_metrics(y_test, model.predict(X_test), model_name="dt_clean")

joblib.dump(model, model_path("model_dt_clean.joblib"))
print("Saved model_dt_clean.joblib")
