"""Config shim for the Gorillas EFFORT-mode pool, raw feature set.

Unlike run_raw/train_utils.py, DATA is fixed rather than reading
params.data_path() -- this run-folder exists specifically to keep a
Gorillas-effort model group permanently on disk, side by side with the
velocity group, rather than switching training_data back and forth. See
AUDIT.md's note on the "two permanent parallel model groups" decision.

N_SAMPLES is None: the pool is a fixed ~5,000 rows, not part of the
n_samples size-tier sweep that run_raw/run_eng use against the much larger
standard pool -- see params.take_samples, which treats None as "use the
whole pool" rather than slicing it further.
"""
import os
import sys
from functools import partial

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(BASE_DIR, "..", "..", "..")
sys.path.insert(0, REPO_ROOT)

from models.regression.common import loader

DATA       = os.path.join(REPO_ROOT, "data", "gorillas_effort_regression.parquet")
MODELS_DIR = os.path.join(REPO_ROOT, "experiments", "regression", "run_raw_effort", "models")
N_SAMPLES  = None

FEATURES = loader.FEATURES
TARGET   = loader.TARGET

# Raw run: the model sees FEATURES as-is, no derived columns.
FEATURE_STEP = ("engineer", "passthrough")

load_data     = partial(loader.load_data, DATA, n_samples=N_SAMPLES)
model_path    = partial(loader.model_path, MODELS_DIR)
save_metrics  = partial(loader.save_metrics, MODELS_DIR)
print_metrics = partial(loader.print_metrics, MODELS_DIR)
