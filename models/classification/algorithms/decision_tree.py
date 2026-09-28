from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

NAME = "Decision Tree"
# Tune on PR-AUC, not ROC-AUC: with a 5-7% positive rate ROC-AUC is flattered by
# the large true-negative pool, while average precision tracks the thing we
# actually care about -- how good the positive predictions are (AUDIT.md task 37).
SCORING = "average_precision"
PARAM_DIST = {
    "max_depth":         [4, 6, 8, 10, 12, 15, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf":  [1, 2, 4, 8],
}


def fit(X_train, y_train, search_cfg):
    search = RandomizedSearchCV(
        # class_weight="balanced" reweights the loss by inverse class frequency,
        # so the 5-7% hit rate in a Gorillas pool does not let the model win by
        # always predicting "miss" (AUDIT.md task 37 / §5.4). A no-op on the
        # Python pool, whose labels are exactly 50/50 -- which is the point: the
        # setting is correct for both.
        DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        param_distributions=PARAM_DIST,
        n_iter=search_cfg["n_iter"], cv=search_cfg["cv"], scoring=SCORING,
        random_state=42, n_jobs=-1, verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"\nBest params : {search.best_params_}")
    print(f"Best CV {SCORING}: {search.best_score_:.4f}\n")
    return search.best_estimator_
