import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from train_utils import load_data, print_metrics, model_path, FEATURE_STEP

import params

_search = params.search_params("regression", "decision_tree")
N_ITER  = _search["n_iter"]
CV      = _search["cv"]

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params.load_params()["test_size"], random_state=42)

pipeline = Pipeline([FEATURE_STEP, ("model", DecisionTreeRegressor(random_state=42))])

param_dist = {
    "model__max_depth":         [8, 10, 12, 15, 20, 25, None],
    "model__min_samples_split": [2, 5, 10, 20],
    "model__min_samples_leaf":  [1, 2, 4, 8],
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=N_ITER, cv=CV, scoring="neg_mean_squared_error",
    random_state=42, n_jobs=-1, verbose=1,
)
search.fit(X_train, y_train)

print(f"\nBest params : {search.best_params_}")
print(f"Best CV RMSE: {(-search.best_score_) ** 0.5:.3f} m\n")

print("Decision Tree -- test set")
print_metrics(y_test, search.best_estimator_.predict(X_test))

joblib.dump(search.best_estimator_, model_path("model_decision_tree.joblib"))
print("Saved model_decision_tree.joblib")
