import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from train_utils import load_data, print_metrics, model_path

import params
import splitting

_search = params.search_params("classification", "decision_tree")
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

param_dist = {
    "max_depth":         [4, 6, 8, 10, 12, 15, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf":  [1, 2, 4, 8],
}

search = RandomizedSearchCV(
    # class_weight="balanced" reweights the loss by inverse class frequency, so the
# 5-7% hit rate in a Gorillas pool does not let the model win by always predicting
# "miss" (AUDIT.md task 37 / §5.4). A no-op on the Python pool, whose labels are
# exactly 50/50 -- which is the point: the setting is correct for both.
    DecisionTreeClassifier(random_state=42, class_weight="balanced"),
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

print("Decision Tree -- test set")
print_metrics(y_test, y_pred, y_prob)

joblib.dump(best, model_path("model_decision_tree.joblib"))
print("Saved model_decision_tree.joblib")
