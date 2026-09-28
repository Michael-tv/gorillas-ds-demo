import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import pandas as pd
from feature_evaluation import evaluate, plot_features, plot_outliers
import params
from models.regression.runs import RUNS

_CFG = RUNS["raw"]
DATA, FEATURES, N_SAMPLES = _CFG.data, _CFG.features, _CFG.n_samples
TARGET = "initial_velocity_ms"

df = params.take_samples(pd.read_parquet(DATA), N_SAMPLES)  # take_samples raises instead of silently truncating (AUDIT.md task 35)

label = f"run_raw ({N_SAMPLES:,} samples)"
evaluate(df, FEATURES, TARGET, label=label)
plot_features(df, FEATURES, TARGET, label=label)
plot_outliers(df, FEATURES, TARGET, label=label)
plot_features(df, FEATURES, TARGET, label=label, clean_only=True)
