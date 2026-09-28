from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

NAME = "Decision Tree"
PARAM_DIST = {
    "model__max_depth":         [8, 10, 12, 15, 20, 25, None],
    "model__min_samples_split": [2, 5, 10, 20],
    "model__min_samples_leaf":  [1, 2, 4, 8],
}


def fit(X_train, y_train, feature_step, search_cfg):
    pipeline = Pipeline([feature_step, ("model", DecisionTreeRegressor(random_state=42))])
    search = RandomizedSearchCV(
        pipeline, param_distributions=PARAM_DIST,
        n_iter=search_cfg["n_iter"], cv=search_cfg["cv"], scoring="neg_mean_squared_error",
        random_state=42, n_jobs=-1, verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"\nBest params : {search.best_params_}")
    print(f"Best CV RMSE: {(-search.best_score_) ** 0.5:.3f} m\n")
    return search.best_estimator_
