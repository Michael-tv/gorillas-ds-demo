from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

NAME = "Decision Tree"
# Tune on PR-AUC, not ROC-AUC: with a 5-7% positive rate, ROC-AUC is flattered
# by the large true-negative pool; PR-AUC tracks precision on the positive
# class instead.
SCORING = "average_precision"
PARAM_DIST = {
    "max_depth":         [4, 6, 8, 10, 12, 15, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf":  [1, 2, 4, 8],
}


def fit(X_train, y_train, search_cfg):
    search = RandomizedSearchCV(
        # class_weight="balanced" reweights by inverse class frequency so the
        # model can't win by always predicting "miss"; a no-op on the 50/50
        # Python pool, which is fine -- the setting is correct for both.
        DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        param_distributions=PARAM_DIST,
        n_iter=search_cfg["n_iter"], cv=search_cfg["cv"], scoring=SCORING,
        random_state=42, n_jobs=-1, verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"\nBest params : {search.best_params_}")
    print(f"Best CV {SCORING}: {search.best_score_:.4f}\n")
    return search.best_estimator_
