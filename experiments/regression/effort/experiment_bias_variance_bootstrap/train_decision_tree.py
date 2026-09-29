"""Bootstrap-based bias/variance decomposition for Decision Tree on the
effort pool, at the same tiers as train_linear_regression.py (this folder)
-- see that file's docstring for the method and the group-aware bootstrap
resampling rationale.

Decision Tree, unlike Linear Regression, has hyperparameters to tune. The
bootstrap decomposition assumes a *fixed* model -- only the training data
varies across the N_BOOTSTRAP resamples -- so hyperparameters are tuned once
per tier via algo.fit's RandomizedSearchCV on the full training split, then
held fixed for every resample. Re-searching per resample would multiply the
search cost by N_BOOTSTRAP (30x here) for no benefit: the decomposition is
about how one fixed model class behaves, not about re-optimizing on every
resample.
"""
import csv
import os

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

import params
import splitting
from feature_engineering import EngineeredFeatures
from experiments.regression.algorithms import decision_tree as algo
from experiments.regression.common import loader

_REPO_ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DATA           = os.path.join(_REPO_ROOT, "data", "gorillas_effort.parquet")
EXPERIMENT_DIR = os.path.join(_REPO_ROOT, "experiments", "regression", "effort", "experiment_bias_variance_bootstrap", "results")
MODELS_DIR     = os.path.join(EXPERIMENT_DIR, "models")
METRICS_DIR    = os.path.join(EXPERIMENT_DIR, "metrics")
KEY            = "decision_tree"
TIERS_DIR      = os.path.join(MODELS_DIR, KEY)
_EXP_PARAMS    = params.load_experiment_params(__file__)
TIERS          = _EXP_PARAMS["tiers"]
N_BOOTSTRAP    = _EXP_PARAMS["n_bootstrap"]
N_ITER         = _EXP_PARAMS["n_iter"][KEY]


def _take_rows(data, idx):
    """Positional selection that works for both a DataFrame and an ndarray --
    same need as splitting._take, duplicated locally since that's a private
    helper of a different module."""
    return data.iloc[idx] if hasattr(data, "iloc") else data[idx]


def _bootstrap_resample(X_train, y_train, groups_train, rng):
    """One bootstrap resample of the training rows, at the group level when
    groups exist -- see train_linear_regression.py's module docstring for why
    row-level would understate the real correlation between a board's
    throws."""
    if groups_train is None:
        idx = rng.integers(0, len(y_train), size=len(y_train))
        return _take_rows(X_train, idx), y_train[idx]

    unique_groups, inverse = np.unique(groups_train, return_inverse=True)
    group_rows = [np.flatnonzero(inverse == i) for i in range(len(unique_groups))]
    picks = rng.integers(0, len(unique_groups), size=len(unique_groups))
    idx = np.concatenate([group_rows[p] for p in picks])
    return _take_rows(X_train, idx), y_train[idx]


def _fit_fixed(X_train, y_train, feature_step, tree_params):
    """Fit a Decision Tree pipeline with fixed hyperparameters -- no search,
    used for the bootstrap resamples once a tier's hyperparameters have been
    chosen (see module docstring)."""
    pipeline = Pipeline([feature_step, ("model", DecisionTreeRegressor(**tree_params))])
    pipeline.fit(X_train, y_train)
    return pipeline


def main():
    X_full, y_full, groups_full = loader.load_data(DATA, clean="no_outlier")
    test_size = params.load_params()["test_size"]
    rng = np.random.default_rng(42)

    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(TIERS_DIR, exist_ok=True)
    rows = []
    for n in TIERS:
        X, y, groups = X_full.iloc[:n], y_full[:n], groups_full[:n]
        print(f"\n=== {algo.NAME} -- n_samples={n} ===")
        X_train, X_test, y_train, y_test, groups_train = splitting.split(
            X, y, groups, test_size=test_size, random_state=42)
        cv_folds = splitting.single_split_cv()

        feature_step = ("features", EngineeredFeatures(output_columns=loader.FEATURES))
        model = algo.fit(X_train, y_train, feature_step, {"n_iter": N_ITER, "cv": cv_folds})
        tree_params = model.named_steps["model"].get_params()

        preds = np.empty((N_BOOTSTRAP, len(y_test)))
        for b in range(N_BOOTSTRAP):
            Xb, yb = _bootstrap_resample(X_train, y_train, groups_train, rng)
            feature_step = ("features", EngineeredFeatures(output_columns=loader.FEATURES))
            model_b = _fit_fixed(Xb, yb, feature_step, tree_params)
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

        joblib.dump(model, os.path.join(TIERS_DIR, f"model_n{n}.joblib"))

    metrics_path = os.path.join(METRICS_DIR, f"metrics_{KEY}.csv")
    with open(metrics_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_samples", "bias2", "variance", "mean_test_mse", "noise_estimate"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved metrics_{KEY}.csv ({len(rows)} tiers) and {len(rows)} models to {TIERS_DIR}")


if __name__ == "__main__":
    main()
