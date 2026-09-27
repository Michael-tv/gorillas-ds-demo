import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "models", "regression", "run_leakage"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import pandas as pd
from feature_engineering import add_engineered_columns
from feature_evaluation import evaluate, plot_features, plot_outliers
from train_utils import DATA, FEATURES, TARGET, N_SAMPLES

import params  # train_utils put the repo root on sys.path

# Prefix, not a random sample -- same rows the run trains on (AUDIT.md task 4b).
df = add_engineered_columns(
    params.take_samples(pd.read_parquet(DATA), N_SAMPLES)).reset_index(drop=True)

evaluate(df, FEATURES, TARGET, label="run_leakage")
plot_features(df, FEATURES, TARGET, label="run_leakage")
plot_outliers(df, FEATURES, TARGET, label="run_leakage")
plot_features(df, FEATURES, TARGET, label="run_leakage", clean_only=True)
