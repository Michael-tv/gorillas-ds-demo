import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
from train_utils import load_data, print_metrics, model_path

N_ITER = 15
CV     = 5

X_train, X_test, y_train, y_test, scaler = load_data(scale=True)

param_dist = {
    "hidden_layer_sizes": [(64, 32), (128, 64), (128, 64, 32), (256, 128), (256, 128, 64)],
    "alpha":              [1e-4, 1e-3, 1e-2],
    "learning_rate_init": [1e-3, 5e-3, 1e-2],
}

search = RandomizedSearchCV(
    MLPRegressor(max_iter=600, random_state=42),
    param_distributions=param_dist,
    n_iter=N_ITER, cv=CV, scoring="neg_mean_squared_error",
    random_state=42, n_jobs=-1, verbose=1,
)
search.fit(X_train, y_train)

print(f"\nBest params : {search.best_params_}")
print(f"Best CV RMSE: {(-search.best_score_) ** 0.5:.3f} m\n")

print("MLP -- test set")
print_metrics(y_test, search.best_estimator_.predict(X_test))

joblib.dump({"model": search.best_estimator_, "scaler": scaler}, model_path("model_mlp.joblib"))
print("Saved model_mlp.joblib")
