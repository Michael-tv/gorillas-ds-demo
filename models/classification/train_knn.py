import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from train_utils import load_data, print_metrics, model_path

import params
import splitting

_search = params.search_params("classification", "knn")
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

# KNeighborsClassifier has NO class_weight parameter -- unlike the logistic,
# tree and forest models, it cannot be reweighted for the 5-7% hit rate
# (AUDIT.md task 37 / §5.4). Left as-is deliberately rather than papered over:
# "not every model exposes the knob" is worth saying out loud, and the
# precision/recall/PR-AUC reported below show what that costs. Resampling the
# training set would be the alternative, at the price of a second mechanism to
# explain. weights="distance" is in the search space but weights NEIGHBOURS by
# distance, not CLASSES by frequency -- it is not a substitute.
pipeline = Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier())])

param_dist = {
    "model__n_neighbors": [3, 5, 7, 10, 15, 20],
    "model__weights":     ["uniform", "distance"],
    "model__metric":      ["euclidean", "manhattan"],
}

search = RandomizedSearchCV(
    pipeline,
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

print("KNN -- test set")
print_metrics(y_test, y_pred, y_prob)

joblib.dump(best, model_path("model_knn.joblib"))
print("Saved model_knn.joblib")
