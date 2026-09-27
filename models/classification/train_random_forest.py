import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from train_utils import load_data, print_metrics, model_path

import params

_search = params.search_params("classification", "random_forest")
N_ITER  = _search["n_iter"]
CV      = _search["cv"]

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params.load_params()["test_size"], random_state=42, stratify=y
)

param_dist = {
    "n_estimators":      [100, 200, 300],
    "max_depth":         [8, 12, 16, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
}

search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=N_ITER, cv=CV, scoring="roc_auc",
    random_state=42, n_jobs=-1, verbose=1,
)
search.fit(X_train, y_train)

print(f"\nBest params : {search.best_params_}")
print(f"Best CV AUC : {search.best_score_:.4f}\n")

best   = search.best_estimator_
y_pred = best.predict(X_test)
y_prob = best.predict_proba(X_test)[:, 1]

print("Random Forest -- test set")
print_metrics(y_test, y_pred, y_prob)

joblib.dump(best, model_path("model_random_forest.joblib"))
print("Saved model_random_forest.joblib")
