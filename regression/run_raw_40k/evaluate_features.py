import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_evaluation import evaluate, plot_features, plot_outliers
from train_utils import DATA, FEATURES, TARGET

evaluate(DATA, FEATURES, TARGET, label="run_raw_40k")
plot_features(DATA, FEATURES, TARGET, label="run_raw_40k")
plot_outliers(DATA, FEATURES, TARGET, label="run_raw_40k")
plot_features(DATA, FEATURES, TARGET, label="run_raw_40k", clean_only=True)
