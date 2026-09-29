"""Train MLP on the velocity pool's engineered feature columns.

Reads data/gorillas_velocity.parquet, a fixed-size pool (no n_samples slicing).
"""
import os

import joblib

import params
import splitting
from experiments.classification.algorithms import mlp as algo
from experiments.classification.common import loader

_REPO_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DATA        = os.path.join(_REPO_ROOT, "data", "gorillas_velocity.parquet")
MODELS_DIR  = os.path.join(_REPO_ROOT, "experiments", "classification", "velocity", "experiment_classification", "results", "models")
METRICS_DIR = os.path.join(_REPO_ROOT, "experiments", "classification", "velocity", "experiment_classification", "results", "metrics")
KEY         = "mlp"


def main():
    X, y, groups = loader.load_data(DATA)
    X_train, X_test, y_train, y_test, groups_train = splitting.split(
        X, y, groups, test_size=params.load_params()["test_size"], random_state=42, stratify=True)

    n_iter = params.load_experiment_params(__file__)["n_iter"][KEY]
    cv = params.load_experiment_params(__file__)["cv"]
    cv_folds = splitting.cv_for(cv, X_train, y_train, groups_train, stratify=True)

    print(f"{algo.NAME} -- experiment_classification\n")
    model = algo.fit(X_train, y_train, {"n_iter": n_iter, "cv": cv_folds})

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"{algo.NAME} -- test set")
    loader.print_metrics(METRICS_DIR, KEY, y_test, y_pred, y_prob)

    joblib.dump(model, loader.model_path(MODELS_DIR, f"model_{KEY}.joblib"))
    print(f"Saved model_{KEY}.joblib")


if __name__ == "__main__":
    main()
