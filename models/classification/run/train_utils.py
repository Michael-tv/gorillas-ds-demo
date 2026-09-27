import os
import sys
from functools import partial

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(BASE_DIR, "..", "..", "..")
sys.path.insert(0, REPO_ROOT)

import params
from models.classification.common import loader

DATA       = params.data_path()
MODELS_DIR = os.path.join(REPO_ROOT, "experiments", "classification", "run", "models")
N_SAMPLES  = params.load_params()["n_samples"]

FEATURES = loader.FEATURES
TARGET   = loader.TARGET

load_data     = partial(loader.load_data, DATA, n_samples=N_SAMPLES)
model_path    = partial(loader.model_path, MODELS_DIR)
save_metrics  = partial(loader.save_metrics, MODELS_DIR)
print_metrics = partial(loader.print_metrics, MODELS_DIR)
