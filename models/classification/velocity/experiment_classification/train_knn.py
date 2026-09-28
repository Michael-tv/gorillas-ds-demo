"""Train KNN on the velocity pool's engineered feature columns.

Converted from the train_classification@<model> foreach matrix
(models/classification/train.py --run run --algorithm knn)
to this experiment_classification/ folder's one-script-per-model pattern --
see models/regression/standard/experiment_skew/train_linear_regression.py
for the general experiment_<name>/train_<model>.py rationale.

Reads data/gorillas_velocity.parquet -- fixed-size, not switchable, matching the
pre-existing train_classification_velocity convention (permanent Gorillas-domain
pipelines don't take an n_samples slice).
"""
import os

import joblib

import params
import splitting
from models.classification.algorithms import knn as algo
from models.classification.common import loader

_REPO_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DATA        = os.path.join(_REPO_ROOT, "data", "gorillas_velocity.parquet")
MODELS_DIR  = os.path.join(_REPO_ROOT, "experiments_results", "classification", "velocity", "experiment_classification", "models")
KEY         = "knn"


def main():
    X, y, groups = loader.load_data(DATA)
    X_train, X_test, y_train, y_test, groups_train = splitting.split(
        X, y, groups, test_size=params.load_params()["test_size"], random_state=42, stratify=True)

    search = params.search_params("classification", KEY)
    cv = params.load_experiment_params(__file__)["cv"]
    cv_folds = splitting.cv_for(cv, X_train, y_train, groups_train, stratify=True)

    print(f"{algo.NAME} -- experiment_classification\n")
    model = algo.fit(X_train, y_train, {"n_iter": search["n_iter"], "cv": cv_folds})

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"{algo.NAME} -- test set")
    loader.print_metrics(MODELS_DIR, KEY, y_test, y_pred, y_prob)

    joblib.dump(model, loader.model_path(MODELS_DIR, f"model_{KEY}.joblib"))
    print(f"Saved model_{KEY}.joblib")


if __name__ == "__main__":
    main()
