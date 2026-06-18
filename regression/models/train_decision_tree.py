import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from train_utils import load_data, print_metrics, model_path

N_ITER = 30
CV     = 5

X_train, X_test, y_train, y_test, _ = load_data()

param_dist = {
    "max_depth":         [8, 10, 12, 15, 20, 25, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf":  [1, 2, 4, 8],
}

search = RandomizedSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=N_ITER, cv=CV, scoring="neg_mean_squared_error",
    random_state=42, n_jobs=-1, verbose=1,
)
search.fit(X_train, y_train)

print(f"\nBest params : {search.best_params_}")
print(f"Best CV RMSE: {(-search.best_score_) ** 0.5:.3f} m\n")

print("Decision Tree -- test set")
print_metrics(y_test, search.best_estimator_.predict(X_test))

joblib.dump({"model": search.best_estimator_}, model_path("model_decision_tree.joblib"))
print("Saved model_decision_tree.joblib")
