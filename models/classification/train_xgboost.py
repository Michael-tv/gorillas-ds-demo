import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from train_utils import load_data, print_metrics, model_path

import params

_search = params.search_params("classification", "xgboost")
N_ITER  = _search["n_iter"]
CV      = _search["cv"]

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params.load_params()["test_size"], random_state=42, stratify=y
)

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
    XGBClassifier(random_state=42, verbosity=0, eval_metric="logloss"),
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

print("XGBoost -- test set")
print_metrics(y_test, y_pred, y_prob)

joblib.dump(best, model_path("model_xgboost.joblib"))
print("Saved model_xgboost.joblib")
