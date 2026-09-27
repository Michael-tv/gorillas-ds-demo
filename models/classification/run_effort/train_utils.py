"""Config shim for the Gorillas EFFORT-mode pool, classification.

See models/regression/run_raw_effort/train_utils.py's docstring for why DATA
is fixed here rather than reading params.data_path().
"""
import os
import sys
from functools import partial

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(BASE_DIR, "..", "..", "..")
sys.path.insert(0, REPO_ROOT)

from models.classification.common import loader

DATA       = os.path.join(REPO_ROOT, "data", "gorillas_effort_classification.parquet")
MODELS_DIR = os.path.join(REPO_ROOT, "experiments", "classification", "run_effort", "models")
N_SAMPLES  = None

FEATURES = loader.FEATURES
TARGET   = loader.TARGET

load_data     = partial(loader.load_data, DATA, n_samples=N_SAMPLES)
model_path    = partial(loader.model_path, MODELS_DIR)
save_metrics  = partial(loader.save_metrics, MODELS_DIR)
print_metrics = partial(loader.print_metrics, MODELS_DIR)
