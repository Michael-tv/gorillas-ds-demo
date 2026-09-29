import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import pandas as pd
from feature_engineering import add_engineered_columns
from feature_evaluation import evaluate, plot_features, plot_outliers
import params
from experiments.classification.common import loader

# Inlined (not imported) to match the DATA/N_SAMPLES used by
# experiments/classification/standard/experiment_classification/'s scripts.
DATA      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "standard_velocity.parquet")
N_SAMPLES = params.load_params()["n_samples"]
FEATURES, TARGET = loader.FEATURES, loader.TARGET

df = add_engineered_columns(params.take_samples(pd.read_parquet(DATA), N_SAMPLES))  # take_samples raises instead of silently truncating

label = f"run ({N_SAMPLES:,} samples)"
evaluate(df, FEATURES, TARGET, label=label)
plot_features(df, FEATURES, TARGET, label=label)
plot_outliers(df, FEATURES, TARGET, label=label)
plot_features(df, FEATURES, TARGET, label=label, clean_only=True)
