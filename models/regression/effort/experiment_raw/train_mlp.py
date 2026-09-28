"""Train MLP on the effort pool's 9 raw feature columns.

Converted from the train_raw@<model> foreach matrix (models/regression/
train.py --run raw --algorithm mlp) to this experiment_raw/
folder's one-script-per-model pattern -- see
../experiment_skew/train_mlp.py for the general
experiment_<name>/train_<model>.py rationale.

Reads data/gorillas_effort.parquet -- fixed-size, not switchable, matching the
pre-existing raw_effort/eng_effort convention (permanent Gorillas-domain
pipelines don't take an n_samples slice).
"""
import os

import joblib

import params
import splitting
from feature_engineering import EngineeredFeatures
from models.regression.algorithms import mlp as algo
from models.regression.common import loader

_REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DATA         = os.path.join(_REPO_ROOT, "data", "gorillas_effort.parquet")
MODELS_DIR   = os.path.join(_REPO_ROOT, "experiments_results", "regression", "effort", "experiment_raw", "models")
KEY          = "mlp"
RAW_FEATURES = loader.FEATURES  # the 9 raw columns, as-is


def main():
    X, y, groups = loader.load_data(DATA)
    X_train, X_test, y_train, y_test, groups_train = splitting.split(
        X, y, groups, test_size=params.load_params()["test_size"], random_state=42)

    search = params.search_params("regression", KEY)
    cv = params.load_experiment_params(__file__)["cv"]
    cv_folds = splitting.cv_for(cv, X_train, y_train, groups_train)

    feature_step = ("features", EngineeredFeatures(output_columns=RAW_FEATURES))
    print(f"{algo.NAME} -- experiment_raw\n")
    model = algo.fit(X_train, y_train, feature_step, {"n_iter": search["n_iter"], "cv": cv_folds})

    print(f"{algo.NAME} -- test set")
    loader.print_metrics(MODELS_DIR, KEY, y_test, model.predict(X_test))

    joblib.dump(model, loader.model_path(MODELS_DIR, f"model_{KEY}.joblib"))
    print(f"Saved model_{KEY}.joblib")


if __name__ == "__main__":
    main()
