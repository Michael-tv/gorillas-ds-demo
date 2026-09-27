import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from train_utils import load_data, print_metrics, model_path

import params
import splitting

X, y, groups = load_data()
# Group-aware when the active pool has groups -- a Gorillas pool shares one
# board's wind and skyline across all 32 of its throws, so a random split would
# put the same board on both sides and score against rows the model has
# effectively already seen. Degrades to an ordinary random split when the pool
# has no group structure (the Python pool's ids are unique per row), so the same
# call is correct for both producers. See splitting.py -- AUDIT.md task 36/§5.1.
X_train, X_test, y_train, y_test, groups_train = splitting.split(
    X, y, groups, test_size=params.load_params()["test_size"], random_state=42, stratify=True)

# class_weight="balanced" reweights the loss by inverse class frequency, so the
# 5-7% hit rate in a Gorillas pool does not let the model win by always predicting
# "miss" (AUDIT.md task 37 / §5.4). A no-op on the Python pool, whose labels are
# exactly 50/50 -- which is the point: the setting is correct for both.
model = Pipeline([("scaler", StandardScaler()),
                  ("model", LogisticRegression(max_iter=1000, random_state=42,
                                               class_weight="balanced"))])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Logistic Regression -- test set")
print_metrics(y_test, y_pred, y_prob)

joblib.dump(model, model_path("model_logistic_regression.joblib"))
print("Saved model_logistic_regression.joblib")
