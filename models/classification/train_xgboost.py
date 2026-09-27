import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from train_utils import load_data, print_metrics, model_path

import params
import splitting

_search = params.search_params("classification", "xgboost")
N_ITER  = _search["n_iter"]
CV      = _search["cv"]

# Tune on PR-AUC, not ROC-AUC: with a 5-7% positive rate ROC-AUC is flattered by
# the large true-negative pool, while average precision tracks the thing we
# actually care about -- how good the positive predictions are (AUDIT.md task 37).
SCORING = "average_precision"

X, y, groups = load_data()
# Group-aware when the active pool has groups -- a Gorillas pool shares one
# board's wind and skyline across all 32 of its throws, so a random split would
# put the same board on both sides and score against rows the model has
# effectively already seen. Degrades to an ordinary random split when the pool
# has no group structure (the Python pool's ids are unique per row), so the same
# call is correct for both producers. See splitting.py -- AUDIT.md task 36/§5.1.
X_train, X_test, y_train, y_test, groups_train = splitting.split(
    X, y, groups, test_size=params.load_params()["test_size"], random_state=42, stratify=True)

# Group-aware folds too: cross-validating with a random KFold inside a
# group-aware split would leak across folds instead of across the test set.
CV_FOLDS = splitting.cv_for(CV, X_train, y_train, groups_train, stratify=True)

# Negative/positive ratio on the TRAINING labels only. 1.0 when the pool is
# balanced, ~13 at a 7% hit rate.
_pos = int(y_train.sum())
_neg = len(y_train) - _pos
SCALE_POS_WEIGHT = (_neg / _pos) if _pos else 1.0
print(f"  scale_pos_weight : {SCALE_POS_WEIGHT:.2f}  "
      f"({_neg} miss / {_pos} hit in train)")

param_dist = {
    "n_estimators":     [200, 300, 500],
    "learning_rate":    [0.01, 0.05, 0.1, 0.2],
    "max_depth":        [3, 4, 5, 6, 8],
    "subsample":        [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "reg_alpha":        [0, 0.1, 1.0],
    "reg_lambda":       [1.0, 2.0, 5.0],
}

search = RandomizedSearchCV(
    # XGBoost's equivalent of class_weight="balanced" is scale_pos_weight, which
    # takes the negative/positive ratio rather than a keyword -- computed from the
    # TRAINING labels only, so nothing about the test set leaks into the model
    # (AUDIT.md task 37 / §5.4). 1.0 on the balanced Python pool, ~13 on a Gorillas
    # pool at a 7% hit rate.
    XGBClassifier(random_state=42, verbosity=0, eval_metric="logloss",
                  scale_pos_weight=SCALE_POS_WEIGHT),
    param_distributions=param_dist,
    n_iter=N_ITER, cv=CV_FOLDS, scoring=SCORING,
    random_state=42, n_jobs=-1, verbose=1,
)
search.fit(X_train, y_train)

print(f"\nBest params : {search.best_params_}")
print(f"Best CV {SCORING}: {search.best_score_:.4f}\n")

best   = search.best_estimator_
y_pred = best.predict(X_test)
y_prob = best.predict_proba(X_test)[:, 1]

print("XGBoost -- test set")
print_metrics(y_test, y_pred, y_prob)

joblib.dump(best, model_path("model_xgboost.joblib"))
print("Saved model_xgboost.joblib")
