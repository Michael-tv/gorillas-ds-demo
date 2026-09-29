"""Train Random Forest on the effort pool's 5 engineered feature columns."""
import os

import joblib

import params
import splitting
from feature_engineering import EngineeredFeatures
from experiments.regression.algorithms import random_forest as algo
from experiments.regression.common import loader

_REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DATA         = os.path.join(_REPO_ROOT, "data", "gorillas_effort.parquet")
MODELS_DIR   = os.path.join(_REPO_ROOT, "experiments", "regression", "effort", "experiment_eng", "results", "models")
METRICS_DIR  = os.path.join(_REPO_ROOT, "experiments", "regression", "effort", "experiment_eng", "results", "metrics")
KEY          = "random_forest"
ENG_FEATURES = ["launch_angle_deg", "wind_x_ms", "drag_param", "height_diff_m", "landing_distance_m"]


def main():
    X, y, groups = loader.load_data(DATA)
    X_train, X_test, y_train, y_test, groups_train = splitting.split(
        X, y, groups, test_size=params.load_params()["test_size"], random_state=42)

    n_iter = params.load_experiment_params(__file__)["n_iter"][KEY]
    cv = params.load_experiment_params(__file__)["cv"]
    cv_folds = splitting.cv_for(cv, X_train, y_train, groups_train)

    feature_step = ("features", EngineeredFeatures(output_columns=ENG_FEATURES))
    print(f"{algo.NAME} -- experiment_eng\n")
    model = algo.fit(X_train, y_train, feature_step, {"n_iter": n_iter, "cv": cv_folds})

    print(f"{algo.NAME} -- test set")
    loader.print_metrics(METRICS_DIR, KEY, y_test, model.predict(X_test))

    joblib.dump(model, loader.model_path(MODELS_DIR, f"model_{KEY}.joblib"))
    print(f"Saved model_{KEY}.joblib")


if __name__ == "__main__":
    main()
