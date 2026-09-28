from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

NAME = "Random Forest"
SCORING = "average_precision"
PARAM_DIST = {
    "n_estimators":      [100, 200, 300],
    "max_depth":         [8, 12, 16, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
}


def fit(X_train, y_train, search_cfg):
    search = RandomizedSearchCV(
        # class_weight="balanced" -- see decision_tree.py's note.
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        param_distributions=PARAM_DIST,
        n_iter=search_cfg["n_iter"], cv=search_cfg["cv"], scoring=SCORING,
        random_state=42, n_jobs=-1, verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"\nBest params : {search.best_params_}")
    print(f"Best CV {SCORING}: {search.best_score_:.4f}\n")
    return search.best_estimator_
