"""Central registry of regression run configs.

Replaces the nine run_raw/run_raw_effort/run_raw_velocity/run_eng/
run_eng_effort/run_eng_velocity/run_skewed train_utils.py shims -- each was a
near-identical file whose only job was to set DATA/MODELS_DIR/FEATURES/
FEATURE_STEP, importable only because scripts/run_with_pythonpath.py had put
its folder on PYTHONPATH ahead of every other run's folder. A shared script
(train_random_forest.py, say) doing `from train_utils import ...` could not
tell you, by reading it, which of the nine it would get -- that was decided
entirely by which DVC stage launched it (AUDIT.md C1, "the worst clarity
offender"). One file, one dict, an explicit `--run` argument to
models/regression/train.py instead.

Each entry says: which pool to read, where models/metrics land, which feature
columns the model sees (used as-is if they're the 9 raw columns, or engineered
inside the model's own Pipeline if they're the 5 derived ones -- see
feature_step() below), and how many rows to train on.
"""
import os
from dataclasses import dataclass
from typing import List, Optional

import params
from models.regression.common import loader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW_FEATURES = loader.FEATURES  # the 9 raw columns, as-is
ENG_FEATURES = ["launch_angle_deg", "wind_x_ms", "drag_param", "height_diff_m", "landing_distance_m"]


@dataclass(frozen=True)
class RunConfig:
    name: str
    data: str                  # absolute path to the parquet this run trains on
    models_dir: str            # absolute path to experiments/regression/<run>/models
    features: List[str]        # RAW_FEATURES or ENG_FEATURES
    n_samples: Optional[int]   # None = train on the whole pool, unsliced


def _exp_dir(run_dir_name):
    return os.path.join(REPO_ROOT, "experiments", "regression", run_dir_name, "models")


def _data(name):
    return os.path.join(REPO_ROOT, "data", f"{name}.parquet")


def _build_runs():
    n_samples = params.load_params()["n_samples"]
    standard  = params.data_path()  # training_data param -- may itself be a Gorillas pool
    return {
        "raw":          RunConfig("raw",          standard,                               _exp_dir("run_raw"),          RAW_FEATURES, n_samples),
        "raw_effort":   RunConfig("raw_effort",   _data("gorillas_effort_regression"),     _exp_dir("run_raw_effort"),   RAW_FEATURES, None),
        "raw_velocity": RunConfig("raw_velocity", _data("gorillas_velocity_regression"),   _exp_dir("run_raw_velocity"), RAW_FEATURES, None),
        "eng":          RunConfig("eng",          standard,                               _exp_dir("run_eng"),          ENG_FEATURES, n_samples),
        "eng_effort":   RunConfig("eng_effort",   _data("gorillas_effort_regression"),     _exp_dir("run_eng_effort"),   ENG_FEATURES, None),
        "eng_velocity": RunConfig("eng_velocity", _data("gorillas_velocity_regression"),   _exp_dir("run_eng_velocity"), ENG_FEATURES, None),
        # skewed reads the FILTERED slice (filter_skewed's output), not the raw
        # pool -- n_samples is deliberately not applied, since this run's size
        # is already set by skew.max_angle_deg and slicing it again would
        # confound the two knobs (see run_skewed/train_utils.py, which still
        # holds the skewed-specific helpers load_holdout()/skewed_row_count()/
        # load_full_pool() used by the two standalone concept scripts).
        "skewed":       RunConfig("skewed",       _data("skewed_training_data"),           _exp_dir("run_skewed"),       ENG_FEATURES, None),
    }


RUNS = _build_runs()


def feature_step(cfg):
    """The Pipeline's first step for a given run: raw columns passed through,
    or the 3 derived ones computed inside the Pipeline -- see
    feature_engineering.EngineeredFeatures. Same step TYPE either way, only
    the column list differs, so there is no separate "passthrough" sentinel."""
    from feature_engineering import EngineeredFeatures
    return ("features", EngineeredFeatures(output_columns=cfg.features))
