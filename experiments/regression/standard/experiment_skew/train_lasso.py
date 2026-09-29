"""Train Lasso Regression on the skewed-extrapolation demo pool.

See train_ridge.py and train_linear_regression.py in this folder for why
Ridge/Lasso appear only in this experiment, and for the full explanation of
this experiment_<name>/train_<model>.py pattern and this demo's data.
"""
import os

import joblib

import params
import splitting
from feature_engineering import EngineeredFeatures
from experiments.regression.algorithms import lasso as algo
from experiments.regression.common import loader

_REPO_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DATA        = os.path.join(_REPO_ROOT, "data", "skewed_training_data.parquet")
MODELS_DIR  = os.path.join(_REPO_ROOT, "experiments", "regression", "standard", "experiment_skew", "results", "models")
METRICS_DIR = os.path.join(_REPO_ROOT, "experiments", "regression", "standard", "experiment_skew", "results", "metrics")
KEY         = "lasso"
ENG_FEATURES = ["launch_angle_deg", "wind_x_ms", "drag_param", "height_diff_m", "landing_distance_m"]


def main():
    X, y, groups = loader.load_data(DATA)
    X_train, X_test, y_train, y_test, groups_train = splitting.split(
        X, y, groups, test_size=params.load_params()["test_size"], random_state=42)

    n_iter = params.load_experiment_params(__file__)["n_iter"][KEY]
    cv_folds = splitting.single_split_cv()

    feature_step = ("features", EngineeredFeatures(output_columns=ENG_FEATURES))
    print(f"{algo.NAME} -- experiment_skew\n")
    model = algo.fit(X_train, y_train, feature_step, {"n_iter": n_iter, "cv": cv_folds})

    print(f"{algo.NAME} -- test set")
    loader.print_metrics(METRICS_DIR, KEY, y_test, model.predict(X_test))

    joblib.dump(model, loader.model_path(MODELS_DIR, f"model_{KEY}.joblib"))
    print(f"Saved model_{KEY}.joblib")


if __name__ == "__main__":
    main()
