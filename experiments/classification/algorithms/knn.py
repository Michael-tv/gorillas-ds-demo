from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NAME = "KNN"
SCORING = "average_precision"
PARAM_DIST = {
    "model__n_neighbors": [3, 5, 7, 10, 15, 20],
    "model__weights":     ["uniform", "distance"],
    "model__metric":      ["euclidean", "manhattan"],
}


def fit(X_train, y_train, search_cfg):
    """KNeighborsClassifier has NO class_weight parameter -- unlike the
    logistic, tree and forest models, it cannot be reweighted for the 5-7%
    hit rate. Left as-is deliberately: "not every model exposes the knob" is
    worth saying out loud, and the precision/recall/PR-AUC reported
    downstream show what that costs. weights="distance" weights NEIGHBOURS
    by distance, not CLASSES by frequency -- it is not a substitute."""
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier())])
    search = RandomizedSearchCV(
        pipeline, param_distributions=PARAM_DIST,
        n_iter=search_cfg["n_iter"], cv=search_cfg["cv"], scoring=SCORING,
        random_state=42, n_jobs=-1, verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"\nBest params : {search.best_params_}")
    print(f"Best CV {SCORING}: {search.best_score_:.4f}\n")
    return search.best_estimator_
