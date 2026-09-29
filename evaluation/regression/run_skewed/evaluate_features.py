import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import pandas as pd
from feature_engineering import add_engineered_columns
from feature_evaluation import evaluate, plot_features, plot_outliers
import params

DATA     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "skewed_training_data.parquet")
FEATURES = ["launch_angle_deg", "wind_x_ms", "drag_param", "height_diff_m", "landing_distance_m"]
TARGET   = "initial_velocity_ms"

# DATA is already the filtered slice the filter_skewed pipeline stage wrote --
# no need to re-apply the angle cut here (that would risk drifting from the bound in params.yaml).
MAX_ANGLE = params.load_params()["skew"]["max_angle_deg"]

df = add_engineered_columns(pd.read_parquet(DATA))

label = f"run_skewed (angle <= {MAX_ANGLE})"
evaluate(df, FEATURES, TARGET, label=label)
plot_features(df, FEATURES, TARGET, label=label)
plot_outliers(df, FEATURES, TARGET, label=label)
plot_features(df, FEATURES, TARGET, label=label, clean_only=True)
