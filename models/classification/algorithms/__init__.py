"""One module per classification algorithm, each exposing
fit(X_train, y_train, search_cfg) -> fitted_estimator. See
models/regression/algorithms/__init__.py's docstring for the rationale --
same pattern, no feature_step here since classification has no raw-vs-
engineered split.
"""
from models.classification.algorithms import (decision_tree, knn,
                                              logistic_regression, mlp,
                                              random_forest, xgboost)

ALGORITHMS = {
    "logistic_regression": logistic_regression,
    "decision_tree":       decision_tree,
    "knn":                 knn,
    "random_forest":       random_forest,
    "mlp":                 mlp,
    "xgboost":              xgboost,
}
