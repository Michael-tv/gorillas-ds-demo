import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from train_utils import load_data, print_metrics, model_path, FEATURE_STEP

import params

_search = params.search_params("regression", "mlp")
N_ITER  = _search["n_iter"]
CV      = _search["cv"]

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params.load_params()["test_size"], random_state=42)

pipeline = Pipeline([FEATURE_STEP, ("scaler", StandardScaler()),
                      ("model", MLPRegressor(max_iter=600, random_state=42))])

param_dist = {
    "model__hidden_layer_sizes": [(64, 32), (128, 64), (128, 64, 32), (256, 128), (256, 128, 64)],
    "model__alpha":              [1e-4, 1e-3, 1e-2],
    "model__learning_rate_init": [1e-3, 5e-3, 1e-2],
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

print("MLP -- test set")
print_metrics(y_test, search.best_estimator_.predict(X_test))

joblib.dump(search.best_estimator_, model_path("model_mlp.joblib"))
print("Saved model_mlp.joblib")
