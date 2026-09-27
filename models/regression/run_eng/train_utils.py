import os
import sys
from functools import partial

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(BASE_DIR, "..", "..", "..")
sys.path.insert(0, REPO_ROOT)

import params
from models.regression.common import loader
from feature_engineering import EngineeredFeatures

DATA       = params.data_path()
MODELS_DIR = os.path.join(REPO_ROOT, "experiments", "regression", "run_eng", "models")
N_SAMPLES  = params.load_params()["n_samples"]

# Engineered feature set: loader.load_data still hands back the raw columns
# (loader.FEATURES) -- FEATURE_STEP derives these from them inside the
# model's own sklearn Pipeline, only for this run.
FEATURES = ["launch_angle_deg", "wind_x_ms", "drag_param", "height_diff_m", "landing_distance_m"]
TARGET   = loader.TARGET

FEATURE_STEP = ("engineer", EngineeredFeatures(output_columns=FEATURES))

load_data     = partial(loader.load_data, DATA, n_samples=N_SAMPLES)
model_path    = partial(loader.model_path, MODELS_DIR)
save_metrics  = partial(loader.save_metrics, MODELS_DIR)
print_metrics = partial(loader.print_metrics, MODELS_DIR)
