import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from train_utils import load_data, print_metrics, model_path

N_ITER = 20
CV     = 5

X_train, X_test, y_train, y_test, _ = load_data()

param_dist = {
    "n_estimators":      [50, 100, 200, 300],
    "max_depth":         [10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      ["sqrt", "log2", None],
}

search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=N_ITER, cv=CV, scoring="neg_mean_squared_error",
    random_state=42, n_jobs=-1, verbose=1,
)
search.fit(X_train, y_train)

print(f"\nBest params : {search.best_params_}")
print(f"Best CV RMSE: {(-search.best_score_) ** 0.5:.3f} m\n")

print("Random Forest -- test set")
print_metrics(y_test, search.best_estimator_.predict(X_test))

joblib.dump({"model": search.best_estimator_}, model_path("model_random_forest.joblib"))
print("Saved model_random_forest.joblib")
