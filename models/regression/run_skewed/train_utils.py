"""Skewed-demo-specific helpers for train_skewed.py and train_balanced.py --
the two standalone concept scripts that live in this folder.

The train_skewed@* matrix stages no longer import this file: they go through
models/regression/train.py --run skewed, which reads the same "skewed"
RunConfig directly from models/regression/runs.py. This file now exists only
for the three things unique to the concept scripts -- the out-of-distribution
holdout, the skewed run's row count (so the balanced control can be sized to
match it), and loading the full unfiltered pool.
"""
import os
import sys

import pandas as pd

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(BASE_DIR, "..", "..", "..")
sys.path.insert(0, REPO_ROOT)

import params
from models.regression.common import loader
from models.regression.runs import RUNS, feature_step

_CFG = RUNS["skewed"]

DATA         = _CFG.data
HOLDOUT_DATA = os.path.join(REPO_ROOT, "data", "skewed_holdout.parquet")
MODELS_DIR   = _CFG.models_dir
FEATURE_STEP = feature_step(_CFG)


def load_data():
    return loader.load_data(DATA, n_samples=None)


def model_path(filename):
    return loader.model_path(MODELS_DIR, filename)


def save_metrics(model_name, **kw):
    return loader.save_metrics(MODELS_DIR, model_name, **kw)


def print_metrics(model_name, y_test, y_pred):
    return loader.print_metrics(MODELS_DIR, model_name, y_test, y_pred)


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
