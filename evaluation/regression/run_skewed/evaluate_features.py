import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "models", "regression", "run_skewed"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import pandas as pd
from feature_engineering import add_engineered_columns
from feature_evaluation import evaluate, plot_features, plot_outliers
from train_utils import DATA, FEATURES, TARGET, MAX_ELEVATION_DEG

df = pd.read_parquet(DATA)
df = df[df["launch_angle_deg"] <= MAX_ELEVATION_DEG]
df = add_engineered_columns(df)

label = f"run_skewed (angle <= {MAX_ELEVATION_DEG})"
evaluate(df, FEATURES, TARGET, label=label)
plot_features(df, FEATURES, TARGET, label=label)
plot_outliers(df, FEATURES, TARGET, label=label)
plot_features(df, FEATURES, TARGET, label=label, clean_only=True)
