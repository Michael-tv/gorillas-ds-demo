import joblib
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV
from train_utils import load_data, print_metrics, model_path

N_ITER = 20
CV     = 5

X_train, X_test, y_train, y_test, scaler = load_data(scale=True)

param_dist = {
    "n_neighbors": [3, 5, 7, 10, 15, 20, 30],
    "weights":     ["uniform", "distance"],
    "metric":      ["euclidean", "manhattan"],
}

search = RandomizedSearchCV(
    KNeighborsRegressor(),
    param_distributions=param_dist,
    n_iter=N_ITER, cv=CV, scoring="neg_mean_squared_error",
    random_state=42, n_jobs=-1, verbose=1,
)
search.fit(X_train, y_train)

print(f"\nBest params : {search.best_params_}")
print(f"Best CV RMSE: {(-search.best_score_) ** 0.5:.3f}\n")

print("KNN Regressor -- test set")
print_metrics(y_test, search.best_estimator_.predict(X_test))

joblib.dump({"model": search.best_estimator_, "scaler": scaler}, model_path("model_knn.joblib"))
print("Saved model_knn.joblib")
