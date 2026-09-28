"""Train Ridge Regression on the effort pool's 5 engineered feature columns.

Converted from the train_eng@<model> foreach matrix (models/regression/
train.py --run eng --algorithm ridge) to this experiment_eng/
folder's one-script-per-model pattern. See ../experiment_raw/
train_ridge.py for the n_samples rationale, and
../experiment_skew/train_ridge.py for the general
experiment_<name>/train_<model>.py pattern.
"""
import os

import joblib

import params
import splitting
from feature_engineering import EngineeredFeatures
from models.regression.algorithms import ridge as algo
from models.regression.common import loader

_REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DATA         = os.path.join(_REPO_ROOT, "data", "gorillas_effort.parquet")
MODELS_DIR   = os.path.join(_REPO_ROOT, "experiments_results", "regression", "effort", "experiment_eng", "models")
KEY          = "ridge"
ENG_FEATURES = ["launch_angle_deg", "wind_x_ms", "drag_param", "height_diff_m", "landing_distance_m"]


def main():
    X, y, groups = loader.load_data(DATA)
    X_train, X_test, y_train, y_test, groups_train = splitting.split(
        X, y, groups, test_size=params.load_params()["test_size"], random_state=42)

    search = params.search_params("regression", KEY)
    cv_folds = splitting.cv_for(search["cv"], X_train, y_train, groups_train)

    feature_step = ("features", EngineeredFeatures(output_columns=ENG_FEATURES))
    print(f"{algo.NAME} -- experiment_eng\n")
    model = algo.fit(X_train, y_train, feature_step, {"n_iter": search["n_iter"], "cv": cv_folds})

    print(f"{algo.NAME} -- test set")
    loader.print_metrics(MODELS_DIR, KEY, y_test, model.predict(X_test))

    joblib.dump(model, loader.model_path(MODELS_DIR, f"model_{KEY}.joblib"))
    print(f"Saved model_{KEY}.joblib")


if __name__ == "__main__":
    main()
