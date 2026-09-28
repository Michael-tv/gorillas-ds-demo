"""Train Logistic Regression on the standard pool's engineered feature columns.

Converted from the train_classification@<model> foreach matrix
(models/classification/train.py --run run --algorithm logistic_regression)
to this experiment_classification/ folder's one-script-per-model pattern --
see models/regression/standard/experiment_skew/train_linear_regression.py
for the general experiment_<name>/train_<model>.py rationale.

Reads data/standard_velocity.parquet, sliced to the first `n_samples` rows
(params.yaml) -- kept switchable here (unlike the Gorillas-domain scripts,
which are fixed-size) mainly so a full sweep can be shrunk for a faster run.
See models/classification/*/experiment_row_count/ for the dedicated
sample-size-convergence story; this script trains at a single size.
"""
import os

import joblib

import params
import splitting
from models.classification.algorithms import logistic_regression as algo
from models.classification.common import loader

_REPO_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DATA        = os.path.join(_REPO_ROOT, "data", "standard_velocity.parquet")
MODELS_DIR  = os.path.join(_REPO_ROOT, "experiments_results", "classification", "standard", "experiment_classification", "models")
KEY         = "logistic_regression"


def main():
    n_samples = params.load_params()["n_samples"]
    X, y, groups = loader.load_data(DATA, n_samples=n_samples)
    X_train, X_test, y_train, y_test, groups_train = splitting.split(
        X, y, groups, test_size=params.load_params()["test_size"], random_state=42, stratify=True)

    search = params.search_params("classification", KEY)
    cv_folds = splitting.cv_for(search["cv"], X_train, y_train, groups_train, stratify=True)

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
