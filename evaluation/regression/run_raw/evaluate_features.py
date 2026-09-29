import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import pandas as pd
from feature_evaluation import evaluate, plot_features, plot_outliers
import params
from experiments.regression.common import loader

# Inlined rather than imported from experiments.regression.runs.RUNS -- that
# registry (and train.py's --run dispatch) is gone, replaced by
# experiments/regression/standard/experiment_raw/, one standalone script per
# model. This is the same DATA/FEATURES/N_SAMPLES that folder's scripts use.
DATA      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "standard_velocity.parquet")
FEATURES  = loader.FEATURES  # the 9 raw columns, as-is
N_SAMPLES = params.load_params()["n_samples"]
TARGET    = "initial_velocity_ms"

df = params.take_samples(pd.read_parquet(DATA), N_SAMPLES)  # take_samples raises instead of silently truncating (AUDIT.md task 35)

label = f"run_raw ({N_SAMPLES:,} samples)"
evaluate(df, FEATURES, TARGET, label=label)
plot_features(df, FEATURES, TARGET, label=label)
plot_outliers(df, FEATURES, TARGET, label=label)
plot_features(df, FEATURES, TARGET, label=label, clean_only=True)
