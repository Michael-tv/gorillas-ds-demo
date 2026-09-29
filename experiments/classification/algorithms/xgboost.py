from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

NAME = "XGBoost"
SCORING = "average_precision"
PARAM_DIST = {
    "n_estimators":     [200, 300, 500],
    "learning_rate":    [0.01, 0.05, 0.1, 0.2],
    "max_depth":        [3, 4, 5, 6, 8],
    "subsample":        [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "reg_alpha":        [0, 0.1, 1.0],
    "reg_lambda":       [1.0, 2.0, 5.0],
}


def fit(X_train, y_train, search_cfg):
    # Negative/positive ratio on the TRAINING labels only. 1.0 when the pool is
    # balanced, ~13 at a 7% hit rate.
    pos = int(y_train.sum())
    neg = len(y_train) - pos
    scale_pos_weight = (neg / pos) if pos else 1.0
    print(f"  scale_pos_weight : {scale_pos_weight:.2f}  ({neg} miss / {pos} hit in train)")

    search = RandomizedSearchCV(
        # XGBoost's equivalent of class_weight="balanced" is scale_pos_weight,
        # which takes the negative/positive ratio rather than a keyword --
        # computed from the TRAINING labels only, so nothing about the test set
        # leaks into the model (AUDIT.md task 37 / §5.4).
        XGBClassifier(random_state=42, verbosity=0, eval_metric="logloss",
                      scale_pos_weight=scale_pos_weight),
        param_distributions=PARAM_DIST,
        n_iter=search_cfg["n_iter"], cv=search_cfg["cv"], scoring=SCORING,
        random_state=42, n_jobs=-1, verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"\nBest params : {search.best_params_}")
    print(f"Best CV {SCORING}: {search.best_score_:.4f}\n")
    return search.best_estimator_
