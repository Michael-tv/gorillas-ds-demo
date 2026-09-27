import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "models", "regression", "run_skewed"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import pandas as pd
from feature_engineering import add_engineered_columns
from feature_evaluation import evaluate, plot_features, plot_outliers
from train_utils import DATA, FEATURES, TARGET

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import params

# DATA is already the filtered slice that the filter_skewed stage wrote, so
# there is nothing to filter here. This script used to re-apply the cut itself
# using train_utils.MAX_ELEVATION_DEG, which no longer exists -- the bound is
# params.yaml's skew.max_angle_deg now, and the filtering is a pipeline stage
# (AUDIT.md tasks 8/9). Re-deriving it here risked drifting from what the
# models actually trained on.
MAX_ANGLE = params.load_params()["skew"]["max_angle_deg"]

df = add_engineered_columns(pd.read_parquet(DATA))

label = f"run_skewed (angle <= {MAX_ANGLE})"
evaluate(df, FEATURES, TARGET, label=label)
plot_features(df, FEATURES, TARGET, label=label)
plot_outliers(df, FEATURES, TARGET, label=label)
plot_features(df, FEATURES, TARGET, label=label, clean_only=True)
