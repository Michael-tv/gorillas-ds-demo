"""Train XGBoost on the standard pool's 9 raw feature columns.

Converted from the train_raw@<model> foreach matrix (models/regression/
train.py --run raw --algorithm xgboost) to this experiment_cv_baseline/
folder's one-script-per-model pattern -- see
../experiment_skew/train_xgboost.py for the general
experiment_<name>/train_<model>.py rationale.

Reads data/standard_velocity.parquet, sliced to the first `n_samples` rows
(params.yaml) -- kept switchable here (unlike the Gorillas-domain raw/eng
scripts, which are fixed-size) mainly so a full sweep can be shrunk for a
faster run without editing this file. See experiment_row_count/ for the
dedicated sample-size-convergence story; this script trains at a single size.

Kept identical to experiment_raw except for retaining 5-fold cross-validation
in the hyperparameter search (see splitting.cv_for) -- a deliberate "before"
baseline so experiment_raw (single validation split) can be compared against
the original CV-based search for cost/robustness.
"""
import os

import joblib

import params
import splitting
from feature_engineering import EngineeredFeatures
from models.regression.algorithms import xgboost as algo
from models.regression.common import loader

_REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DATA         = os.path.join(_REPO_ROOT, "data", "standard_velocity.parquet")
MODELS_DIR   = os.path.join(_REPO_ROOT, "experiments_results", "regression", "standard", "experiment_cv_baseline", "models")
KEY          = "xgboost"
RAW_FEATURES = loader.FEATURES  # the 9 raw columns, as-is


def main():
    n_samples = params.load_params()["n_samples"]
    X, y, groups = loader.load_data(DATA, n_samples=n_samples)
    X_train, X_test, y_train, y_test, groups_train = splitting.split(
        X, y, groups, test_size=params.load_params()["test_size"], random_state=42)

    search = params.search_params("regression", KEY)
    cv = params.load_experiment_params(__file__)["cv"]
    cv_folds = splitting.cv_for(cv, X_train, y_train, groups_train)

    feature_step = ("features", EngineeredFeatures(output_columns=RAW_FEATURES))
    print(f"{algo.NAME} -- experiment_cv_baseline\n")
    model = algo.fit(X_train, y_train, feature_step, {"n_iter": search["n_iter"], "cv": cv_folds})

    print(f"{algo.NAME} -- test set")
    loader.print_metrics(MODELS_DIR, KEY, y_test, model.predict(X_test))

    joblib.dump(model, loader.model_path(MODELS_DIR, f"model_{KEY}.joblib"))
    print(f"Saved model_{KEY}.joblib")


if __name__ == "__main__":
    main()
