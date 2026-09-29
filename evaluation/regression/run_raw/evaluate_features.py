import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import pandas as pd
from feature_evaluation import evaluate, plot_features, plot_outliers
import params
from experiments.regression.common import loader

# Same DATA/FEATURES/N_SAMPLES as experiments/regression/standard/experiment_raw/.
DATA      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "standard_velocity.parquet")
FEATURES  = loader.FEATURES  # the 9 raw columns, as-is
N_SAMPLES = params.load_params()["n_samples"]
TARGET    = "initial_velocity_ms"

df = params.take_samples(pd.read_parquet(DATA), N_SAMPLES)  # raises instead of silently truncating

label = f"run_raw ({N_SAMPLES:,} samples)"
evaluate(df, FEATURES, TARGET, label=label)
plot_features(df, FEATURES, TARGET, label=label)
plot_outliers(df, FEATURES, TARGET, label=label)
plot_features(df, FEATURES, TARGET, label=label, clean_only=True)
