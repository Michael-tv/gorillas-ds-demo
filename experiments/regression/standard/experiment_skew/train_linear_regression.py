"""Train Linear Regression on the skewed-extrapolation demo pool.

Worked example of the experiment_<name>/train_<model>.py pattern: one
standalone, explicit script per model, no --run/--algorithm dispatch through
train.py/runs.py to trace. Still imports the shared, already-one-file-per-model
experiments/regression/algorithms/<name>.py (its Pipeline shape and search space
aren't duplicated here) and the shared loader/splitting utilities -- only the
per-experiment orchestration (which data, which features, which output
directory) is explicit and local to this file.

data/skewed_training_data.parquet is the low-angle slice filter_skewed
produces (this folder's dvc.yaml, `filter_skewed` stage); rows above
skew.max_angle_deg are held out separately as data/skewed_holdout.parquet, the
out-of-distribution test set this demo measures against (AUDIT.md tasks
8-10). Always the 5 engineered features -- this demo has no raw variant.
"""
import os

import joblib

import params
import splitting
from feature_engineering import EngineeredFeatures
from experiments.regression.algorithms import linear_regression as algo
from experiments.regression.common import loader

_REPO_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DATA        = os.path.join(_REPO_ROOT, "data", "skewed_training_data.parquet")
MODELS_DIR  = os.path.join(_REPO_ROOT, "experiments", "regression", "standard", "experiment_skew", "results", "models")
METRICS_DIR = os.path.join(_REPO_ROOT, "experiments", "regression", "standard", "experiment_skew", "results", "metrics")
KEY         = "linear_regression"
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
