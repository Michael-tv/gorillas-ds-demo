import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from train_utils import load_data, print_metrics, model_path, FEATURE_STEP

import params

_search = params.search_params("regression", "xgboost")
N_ITER  = _search["n_iter"]
CV      = _search["cv"]

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params.load_params()["test_size"], random_state=42)

pipeline = Pipeline([FEATURE_STEP, ("model", XGBRegressor(random_state=42, verbosity=0))])

param_dist = {
    "model__n_estimators":     [200, 300, 500],
    "model__learning_rate":    [0.01, 0.05, 0.1, 0.2],
    "model__max_depth":        [3, 4, 5, 6, 8],
    "model__subsample":        [0.7, 0.8, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 1.0],
    "model__reg_alpha":        [0, 0.1, 1.0],
    "model__reg_lambda":       [1.0, 2.0, 5.0],
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=N_ITER, cv=CV, scoring="neg_mean_squared_error",
    random_state=42, n_jobs=-1, verbose=1,
)
search.fit(X_train, y_train)

print(f"\nBest params : {search.best_params_}")
print(f"Best CV RMSE: {(-search.best_score_) ** 0.5:.3f} m/s\n")

print("XGBoost -- test set")
print_metrics(y_test, search.best_estimator_.predict(X_test))

joblib.dump(search.best_estimator_, model_path("model_xgboost.joblib"))
print("Saved model_xgboost.joblib")
