import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "models", "regression", "run_leakage"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import pandas as pd
from feature_engineering import add_engineered_columns
from feature_evaluation import evaluate, plot_features, plot_outliers
from train_utils import DATA, FEATURES, TARGET, N_SAMPLES

df = add_engineered_columns(pd.read_parquet(DATA)).sample(n=N_SAMPLES, random_state=42).reset_index(drop=True)

evaluate(df, FEATURES, TARGET, label="run_leakage")
plot_features(df, FEATURES, TARGET, label="run_leakage")
plot_outliers(df, FEATURES, TARGET, label="run_leakage")
plot_features(df, FEATURES, TARGET, label="run_leakage", clean_only=True)
