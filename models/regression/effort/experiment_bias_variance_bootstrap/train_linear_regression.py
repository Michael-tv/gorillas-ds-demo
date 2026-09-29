"""Bootstrap-based bias/variance decomposition for Linear Regression on the
effort pool, at the same six sample-size tiers as experiment_row_count -- so
this experiment answers, with real numbers instead of the train/test-gap
proxy used there, the question that sweep's flat error curve raised: is
linear regression's plateaued error a bias problem (more data won't help)
or a variance problem (more data would)?

Method (the same identity used by e.g. mlxtend.evaluate.bias_variance_decomp,
and by the bootstrap implementation at
https://www.geeksforgeeks.org/machine-learning/bias-vs-variance-in-machine-learning/):
for each tier, hold ONE fixed test set aside, then fit N_BOOTSTRAP models on
independent bootstrap resamples of the remaining training rows. Stack their
predictions on that fixed test set into a (N_BOOTSTRAP, n_test) array and
decompose the expected squared error:

    mean_test_mse ~= bias2 + variance + noise_estimate

    bias2          = ((y_test - preds.mean(axis=0)) ** 2).mean()
                     -- how far the *average* prediction sits from the truth
    variance       = preds.var(axis=0).mean()
                     -- how much the N_BOOTSTRAP models disagree with each other
    noise_estimate = mean_test_mse - bias2 - variance
                     -- whatever's left: irreducible label noise, plus any
                     slack from a finite N_BOOTSTRAP/n_test

This only holds for squared-error loss (hence MSE here, not MAE) -- see
../experiment_row_count/train_linear_regression.py's docstring for why the
train/test-gap proxy used there is a heuristic, not this.

Bootstrap resampling is done at the group_id level, not per-row: gorillas
throws are grouped by board (see splitting.py), so a naive row-level
bootstrap would let one board's correlated throws be resampled as if
independent, understating their real correlation and biasing the variance
estimate down. Resampling whole groups with replacement keeps that
correlation intact in every resample, the same way splitting.split() keeps
it intact across the train/test boundary.

One plain (non-bootstrapped) model is also fit and saved per tier, exactly
as experiment_row_count does, purely so each tier has an inspectable model
-- the N_BOOTSTRAP models used for the decomposition itself are transient
and not saved (180 of them across 6 tiers x 30 bootstraps would dwarf the
one-model-per-tier convention for no benefit).
"""
import csv
import os

import joblib
import numpy as np

import params
import splitting
from feature_engineering import EngineeredFeatures
from models.regression.algorithms import linear_regression as algo
from models.regression.common import loader

_REPO_ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DATA           = os.path.join(_REPO_ROOT, "data", "gorillas_effort.parquet")
EXPERIMENT_DIR = os.path.join(_REPO_ROOT, "experiments_results", "regression", "effort", "experiment_bias_variance_bootstrap")
MODELS_DIR     = os.path.join(EXPERIMENT_DIR, "models")
KEY            = "linear_regression"
TIERS_DIR      = os.path.join(EXPERIMENT_DIR, "tiers", KEY)
_EXP_PARAMS    = params.load_experiment_params(__file__)
TIERS          = _EXP_PARAMS["tiers"]
N_BOOTSTRAP    = _EXP_PARAMS["n_bootstrap"]


def _take_rows(data, idx):
    """Positional selection that works for both a DataFrame and an ndarray --
    same need as splitting._take, duplicated locally since that's a private
    helper of a different module."""
    return data.iloc[idx] if hasattr(data, "iloc") else data[idx]


def _bootstrap_resample(X_train, y_train, groups_train, rng):
    """One bootstrap resample of the training rows, at the group level when
    groups exist -- see module docstring for why row-level would understate
    the real correlation between a board's throws."""
    if groups_train is None:
        idx = rng.integers(0, len(y_train), size=len(y_train))
        return _take_rows(X_train, idx), y_train[idx]

    unique_groups, inverse = np.unique(groups_train, return_inverse=True)
    group_rows = [np.flatnonzero(inverse == i) for i in range(len(unique_groups))]
    picks = rng.integers(0, len(unique_groups), size=len(unique_groups))
    idx = np.concatenate([group_rows[p] for p in picks])
    return _take_rows(X_train, idx), y_train[idx]


def main():
    X_full, y_full, groups_full = loader.load_data(DATA, clean="no_outlier")
    test_size = params.load_params()["test_size"]
    rng = np.random.default_rng(42)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(TIERS_DIR, exist_ok=True)
    rows = []
    for n in TIERS:
        X, y, groups = X_full.iloc[:n], y_full[:n], groups_full[:n]
        print(f"\n=== {algo.NAME} -- n_samples={n} ===")
        X_train, X_test, y_train, y_test, groups_train = splitting.split(
            X, y, groups, test_size=test_size, random_state=42)

        preds = np.empty((N_BOOTSTRAP, len(y_test)))
        for b in range(N_BOOTSTRAP):
            Xb, yb = _bootstrap_resample(X_train, y_train, groups_train, rng)
            feature_step = ("features", EngineeredFeatures(output_columns=loader.FEATURES))
            model_b = algo.fit(Xb, yb, feature_step, {})
            preds[b] = model_b.predict(X_test)

        mean_pred = preds.mean(axis=0)
        bias2 = float(((y_test - mean_pred) ** 2).mean())
        variance = float(preds.var(axis=0).mean())
        mean_test_mse = float(((preds - y_test[None, :]) ** 2).mean())
        noise_estimate = mean_test_mse - bias2 - variance
        print(f"  bias2={bias2:.3f}  variance={variance:.3f}  "
              f"mean_test_mse={mean_test_mse:.3f}  noise_estimate={noise_estimate:.3f}")
        rows.append({"n_samples": n, "bias2": bias2, "variance": variance,
                     "mean_test_mse": mean_test_mse, "noise_estimate": noise_estimate})

        feature_step = ("features", EngineeredFeatures(output_columns=loader.FEATURES))
        model = algo.fit(X_train, y_train, feature_step, {})
        joblib.dump(model, os.path.join(TIERS_DIR, f"model_n{n}.joblib"))

    metrics_path = os.path.join(MODELS_DIR, f"metrics_{KEY}.csv")
    with open(metrics_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_samples", "bias2", "variance", "mean_test_mse", "noise_estimate"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved metrics_{KEY}.csv ({len(rows)} tiers) and {len(rows)} models to {TIERS_DIR}")


if __name__ == "__main__":
    main()
