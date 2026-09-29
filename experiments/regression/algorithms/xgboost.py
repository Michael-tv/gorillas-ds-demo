from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

NAME = "XGBoost"
PARAM_DIST = {
    "model__n_estimators":     [200, 300, 500],
    "model__learning_rate":    [0.01, 0.05, 0.1, 0.2],
    "model__max_depth":        [3, 4, 5, 6, 8],
    "model__subsample":        [0.7, 0.8, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 1.0],
    "model__reg_alpha":        [0, 0.1, 1.0],
    "model__reg_lambda":       [1.0, 2.0, 5.0],
}


def fit(X_train, y_train, feature_step, search_cfg):
    pipeline = Pipeline([feature_step, ("model", XGBRegressor(random_state=42, verbosity=0))])
    search = RandomizedSearchCV(
        pipeline, param_distributions=PARAM_DIST,
        n_iter=search_cfg["n_iter"], cv=search_cfg["cv"], scoring="neg_mean_squared_error",
        random_state=42, n_jobs=-1, verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"\nBest params : {search.best_params_}")
    print(f"Best CV RMSE: {(-search.best_score_) ** 0.5:.3f} m/s\n")
    return search.best_estimator_
