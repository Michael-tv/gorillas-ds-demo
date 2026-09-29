from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NAME = "KNN Regressor"
PARAM_DIST = {
    "model__n_neighbors": [3, 5, 7, 10, 15, 20, 30],
    "model__weights":     ["uniform", "distance"],
    "model__metric":      ["euclidean", "manhattan"],
}


def fit(X_train, y_train, feature_step, search_cfg):
    pipeline = Pipeline([feature_step, ("scaler", StandardScaler()), ("model", KNeighborsRegressor())])
    search = RandomizedSearchCV(
        pipeline, param_distributions=PARAM_DIST,
        n_iter=search_cfg["n_iter"], cv=search_cfg["cv"], scoring="neg_mean_squared_error",
        random_state=42, n_jobs=-1, verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"\nBest params : {search.best_params_}")
    print(f"Best CV RMSE: {(-search.best_score_) ** 0.5:.3f}\n")
    return search.best_estimator_
