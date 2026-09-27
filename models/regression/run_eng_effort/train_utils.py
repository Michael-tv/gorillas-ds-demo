"""Config shim for the Gorillas EFFORT-mode pool, engineered feature set.

See run_raw_effort/train_utils.py's docstring for why DATA is fixed here
rather than reading params.data_path(). FEATURE_STEP derives wind_x_ms/
drag_param/height_diff_m inside the model's own sklearn Pipeline, same as
run_eng/train_utils.py -- loader.load_data still hands back the raw columns.
"""
import os
import sys
from functools import partial

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(BASE_DIR, "..", "..", "..")
sys.path.insert(0, REPO_ROOT)

from models.regression.common import loader
from feature_engineering import EngineeredFeatures

DATA       = os.path.join(REPO_ROOT, "data", "gorillas_effort_regression.parquet")
MODELS_DIR = os.path.join(REPO_ROOT, "experiments", "regression", "run_eng_effort", "models")
N_SAMPLES  = None

FEATURES = ["launch_angle_deg", "wind_x_ms", "drag_param", "height_diff_m", "landing_distance_m"]
TARGET   = loader.TARGET

FEATURE_STEP = ("engineer", EngineeredFeatures(output_columns=FEATURES))

load_data     = partial(loader.load_data, DATA, n_samples=N_SAMPLES)
model_path    = partial(loader.model_path, MODELS_DIR)
save_metrics  = partial(loader.save_metrics, MODELS_DIR)
print_metrics = partial(loader.print_metrics, MODELS_DIR)
