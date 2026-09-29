"""One module per regression algorithm, each exposing fit(X_train, y_train,
feature_step, search_cfg) -> fitted_estimator. train.py owns everything
common (load, split, CV_FOLDS, print_metrics, save); each module here owns
only what genuinely differs per algorithm: the estimator, its Pipeline shape,
its search space, and how it searches (RandomizedSearchCV, GridSearchCV, a
*CV meta-estimator, or no search at all) -- collapsing that into one shared
shape would have hidden real differences rather than removed duplication.

ALGORITHMS maps the same keys params.yaml's search.regression block and
dvc_models_regression.yaml's `model:` field use.
"""
from experiments.regression.algorithms import (decision_tree, knn, lasso,
                                          linear_regression, mlp, polynomial,
                                          random_forest, ridge, xgboost)

ALGORITHMS = {
    "linear_regression": linear_regression,
    "ridge":              ridge,
    "lasso":              lasso,
    "polynomial":         polynomial,
    "decision_tree":      decision_tree,
    "knn":                knn,
    "random_forest":      random_forest,
    "mlp":                mlp,
    "xgboost":            xgboost,
}
