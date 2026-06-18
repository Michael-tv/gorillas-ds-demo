"""
LEAKY pipeline — decision tree trained on contaminated data.

The test rows are accidentally included in the training set
(e.g. feature engineering was run on the full dataset and saved
before the train/test split was applied).

A fully-grown tree memorises every training example, so because
the test rows are also in the training set the reported test MAE
collapses to zero — a completely fraudulent result.
"""
import os
import sys
import numpy as np
import joblib
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_utils import load_data, model_path, print_metrics

X_train, X_test, y_train, y_test = load_data()

X_train_leaky = np.vstack([X_train, X_test])
y_train_leaky = np.concatenate([y_train, y_test])

print(f"Train set size (leaky) : {len(X_train_leaky):,}  ({len(X_test):,} test rows included)")
print(f"Test  set size         : {len(X_test):,}")

model = DecisionTreeRegressor(max_depth=None, random_state=42)
model.fit(X_train_leaky, y_train_leaky)

print("\nDecision Tree (leaky) -- train set")
print_metrics(y_train_leaky, model.predict(X_train_leaky))

print("Decision Tree (leaky) -- test set")
print_metrics(y_test, model.predict(X_test), model_name="dt_leaky")
print("NOTE: near-zero test MAE is fraudulent -- model memorised the test rows.")

joblib.dump({"model": model}, model_path("model_dt_leaky.joblib"))
print("Saved model_dt_leaky.joblib")
