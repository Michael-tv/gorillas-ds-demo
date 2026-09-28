import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import pandas as pd
from feature_engineering import add_engineered_columns
from feature_evaluation import evaluate, plot_features, plot_outliers
import params

# Inlined rather than imported from models.regression.runs.RUNS -- that
# registry (and train.py's --run dispatch) is gone, replaced by
# models/regression/standard/experiment_eng/, one standalone script per
# model. This is the same DATA/FEATURES/N_SAMPLES that folder's scripts use.
DATA      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "standard_velocity.parquet")
FEATURES  = ["launch_angle_deg", "wind_x_ms", "drag_param", "height_diff_m", "landing_distance_m"]
N_SAMPLES = params.load_params()["n_samples"]
TARGET    = "initial_velocity_ms"

df = add_engineered_columns(params.take_samples(pd.read_parquet(DATA), N_SAMPLES))  # take_samples raises instead of silently truncating (AUDIT.md task 35)

label = f"run_eng ({N_SAMPLES:,} samples)"
evaluate(df, FEATURES, TARGET, label=label)
plot_features(df, FEATURES, TARGET, label=label)
plot_outliers(df, FEATURES, TARGET, label=label)
plot_features(df, FEATURES, TARGET, label=label, clean_only=True)
