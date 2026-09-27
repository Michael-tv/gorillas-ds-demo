"""Run config for the skewed-extrapolation demo.

A thin shim over models/regression/common/loader.py, the same shape as
run_raw/train_utils.py and run_eng/train_utils.py. It used to be a
200-line near-duplicate with its own loader, splitter, metrics and a
load-time angle filter; three things moved out of it (AUDIT.md tasks 8, 9, 7):

  * The angle filter is now the `filter_skewed` DVC stage, so the filtered pool
    is a real artifact, the bound is a sweepable param, and the excluded rows
    survive as data/skewed_holdout.parquet instead of being silently dropped.
  * MAX_ELEVATION_DEG is gone -- the bound lives in params.yaml as
    `skew.max_angle_deg`.
  * Feature engineering happens in the model's Pipeline via FEATURE_STEP, not at
    load time, so the saved .joblib carries its own preprocessing.

Defining FEATURE_STEP is also what unbreaks this run: the nine `train_skewed@*`
stages import it from here, and its absence made every one of them die on
`ImportError: cannot import name 'FEATURE_STEP'`.
"""
import os
import sys
from functools import partial

import pandas as pd

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(BASE_DIR, "..", "..", "..")
sys.path.insert(0, REPO_ROOT)

import params
from models.regression.common import loader
from feature_engineering import EngineeredFeatures

# The filtered slice, not the full pool -- produced by the filter_skewed stage.
DATA         = os.path.join(REPO_ROOT, "data", "skewed_training_data.parquet")
HOLDOUT_DATA = os.path.join(REPO_ROOT, "data", "skewed_holdout.parquet")

MODELS_DIR = os.path.join(REPO_ROOT, "experiments", "regression", "run_skewed", "models")
N_SAMPLES  = params.load_params()["n_samples"]

# Engineered feature set, same as run_eng: loader.load_data hands back the raw
# columns and FEATURE_STEP derives these inside the Pipeline.
FEATURES = ["launch_angle_deg", "wind_x_ms", "drag_param", "height_diff_m", "landing_distance_m"]
TARGET   = loader.TARGET

FEATURE_STEP = ("engineer", EngineeredFeatures(output_columns=FEATURES))

# n_samples is deliberately NOT applied here. It sizes the sample-size tiers out
# of the full pool; this run reads an already-filtered subset whose size is set
# by skew.max_angle_deg, and slicing it again would confound the two knobs.
load_data     = partial(loader.load_data, DATA, n_samples=None)
model_path    = partial(loader.model_path, MODELS_DIR)
save_metrics  = partial(loader.save_metrics, MODELS_DIR)
print_metrics = partial(loader.print_metrics, MODELS_DIR)


def load_holdout():
    """The out-of-distribution rows -- everything ABOVE skew.max_angle_deg.

    This is the set the demo's claim is actually about. Without it "extrapolates
    poorly beyond the cap" was asserted against a test split drawn from the same
    filtered slice, which has the same hole (AUDIT.md task 10).
    """
    return loader.load_data(HOLDOUT_DATA, n_samples=None)


def skewed_row_count():
    """How many rows the skewed run trains on.

    The balanced model exists to be compared against the skewed one, so it has
    to be the same size: otherwise part of any difference between them is just
    "one saw more data", which is a different lesson from the one being taught
    (AUDIT.md task 11 -- the two were 8,525 vs a hardcoded 8,000).
    """
    return len(pd.read_parquet(DATA))


def load_full_pool(n_samples=None):
    """The unfiltered pool -- the full angle range, for the balanced comparison."""
    return loader.load_data(params.data_path(), n_samples=n_samples)
