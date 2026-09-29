from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NAME = "MLP"
SCORING = "average_precision"
PARAM_DIST = {
    "model__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64), (64, 64, 32)],
    "model__activation":         ["relu", "tanh"],
    "model__alpha":              [1e-4, 1e-3, 1e-2],
    "model__learning_rate_init": [1e-3, 5e-4],
}


def fit(X_train, y_train, search_cfg):
    """MLPClassifier has no class_weight parameter (see knn's note); judge it
    on precision/recall/PR-AUC, not accuracy."""
    pipeline = Pipeline([("scaler", StandardScaler()),
                          ("model", MLPClassifier(max_iter=500, early_stopping=True, random_state=42))])
    search = RandomizedSearchCV(
        pipeline, param_distributions=PARAM_DIST,
        n_iter=search_cfg["n_iter"], cv=search_cfg["cv"], scoring=SCORING,
        random_state=42, n_jobs=-1, verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"\nBest params : {search.best_params_}")
    print(f"Best CV {SCORING}: {search.best_score_:.4f}\n")
    return search.best_estimator_
