import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import pandas as pd
from feature_engineering import add_engineered_columns
from feature_evaluation import evaluate, plot_features, plot_outliers
import params

# Inlined rather than imported from a train_utils.py shim -- this demo is now
# experiments/regression/standard/experiment_bias_variance/train_underfitting.py/
# train_overfitting.py, self-contained scripts with no shared train_utils.py.
DATA     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "standard_velocity.parquet")
FEATURES = ["launch_angle_deg", "wind_x_ms", "drag_param", "height_diff_m", "landing_distance_m"]
TARGET   = "initial_velocity_ms"
N_SAMPLES = params.load_params()["n_samples"]

# Prefix, not a random sample -- same rows the run trains on (AUDIT.md task 4b).
df = add_engineered_columns(
    params.take_samples(pd.read_parquet(DATA), N_SAMPLES)).reset_index(drop=True)

evaluate(df, FEATURES, TARGET, label="run_bias_variance")
plot_features(df, FEATURES, TARGET, label="run_bias_variance")
plot_outliers(df, FEATURES, TARGET, label="run_bias_variance")
plot_features(df, FEATURES, TARGET, label="run_bias_variance", clean_only=True)
