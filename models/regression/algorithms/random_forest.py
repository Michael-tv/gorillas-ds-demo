from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

NAME = "Random Forest"
PARAM_DIST = {
    "model__n_estimators":      [50, 100, 200, 300],
    "model__max_depth":         [10, 15, 20, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf":  [1, 2, 4],
    "model__max_features":      ["sqrt", "log2", None],
}


def fit(X_train, y_train, feature_step, search_cfg):
    pipeline = Pipeline([feature_step, ("model", RandomForestRegressor(random_state=42, n_jobs=-1))])
    search = RandomizedSearchCV(
        pipeline, param_distributions=PARAM_DIST,
        n_iter=search_cfg["n_iter"], cv=search_cfg["cv"], scoring="neg_mean_squared_error",
        random_state=42, n_jobs=-1, verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"\nBest params : {search.best_params_}")
    print(f"Best CV RMSE: {(-search.best_score_) ** 0.5:.3f} m\n")
    return search.best_estimator_
