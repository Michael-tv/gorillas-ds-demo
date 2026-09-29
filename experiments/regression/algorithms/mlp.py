from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NAME = "MLP"
PARAM_DIST = {
    "model__hidden_layer_sizes": [(64, 32), (128, 64), (128, 64, 32), (256, 128), (256, 128, 64)],
    "model__alpha":              [1e-4, 1e-3, 1e-2],
    "model__learning_rate_init": [1e-3, 5e-3, 1e-2],
}


def fit(X_train, y_train, feature_step, search_cfg):
    pipeline = Pipeline([feature_step, ("scaler", StandardScaler()),
                          ("model", MLPRegressor(max_iter=600, random_state=42))])
    search = RandomizedSearchCV(
        pipeline, param_distributions=PARAM_DIST,
        n_iter=search_cfg["n_iter"], cv=search_cfg["cv"], scoring="neg_mean_squared_error",
        random_state=42, n_jobs=-1, verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"\nBest params : {search.best_params_}")
    print(f"Best CV RMSE: {(-search.best_score_) ** 0.5:.3f} m\n")
    return search.best_estimator_
