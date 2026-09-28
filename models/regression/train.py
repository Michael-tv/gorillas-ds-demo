"""The one entrypoint for every regression algorithm, on every regression run.

Replaces sixteen `python scripts/run_with_pythonpath.py <run_dir> <script>
TRAIN_MODEL_NAME=... TRAIN_CLEAN=...` DVC commands with one explicit CLI:

    python -m models.regression.train --run raw --algorithm random_forest --key random_forest

Everything common to every algorithm lives here once: loading the run's pool
(models/regression/runs.py), the group-aware split, computing CV folds,
printing/saving metrics, and saving the model. Everything that genuinely
differs per algorithm -- the estimator, its Pipeline shape, its search space,
whether it searches at all -- lives in models/regression/algorithms/<name>.py,
one small module per algorithm (AUDIT.md C1 / the run_*/train_utils.py
cleanup).

--run and --algorithm are usually the same key in dvc_models_regression.yaml,
except for the deliberate aliases that reuse one algorithm's logic under a
different name (random_forest_no_outlier --key vs. --algorithm random_forest,
--clean no_outlier).
"""
import argparse

import joblib

import params
import splitting
from models.regression.algorithms import ALGORITHMS
from models.regression.common import loader
from models.regression.runs import RUNS, feature_step


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, choices=sorted(RUNS),
                   help="which pool/feature-set to train on")
    p.add_argument("--algorithm", required=True, choices=sorted(ALGORITHMS),
                   help="which model to build, and whose params.yaml search budget to use")
    p.add_argument("--key", required=True,
                   help="output name: experiments/.../model_<key>.joblib, metrics_<key>.csv")
    p.add_argument("--clean", default="", choices=["", "iqr", "range", "no_outlier"])
    args = p.parse_args()

    cfg  = RUNS[args.run]
    algo = ALGORITHMS[args.algorithm]

    X, y, groups = loader.load_data(cfg.data, n_samples=cfg.n_samples, clean=args.clean)
    # Group-aware when the active pool has groups -- a Gorillas pool shares one
    # board's wind and skyline across all 32 of its throws, so a random split
    # would put the same board on both sides and score against rows the model
    # has effectively already seen. Degrades to an ordinary random split when
    # the pool has no group structure (the Python pool's ids are unique per
    # row), so the same call is correct for every run. See splitting.py --
    # AUDIT.md task 36/§5.1.
    X_train, X_test, y_train, y_test, groups_train = splitting.split(
        X, y, groups, test_size=params.load_params()["test_size"], random_state=42)

    search = params.search_params("regression", args.algorithm)
    # Group-aware folds too: cross-validating with a random KFold inside a
    # group-aware split would leak across folds instead of across the test set.
    cv_folds = splitting.cv_for(search["cv"], X_train, y_train, groups_train)

    print(f"{algo.NAME} -- run={args.run}\n")
    model = algo.fit(X_train, y_train, feature_step(cfg),
                     {"n_iter": search["n_iter"], "cv": cv_folds})

    print(f"{algo.NAME} -- test set")
    loader.print_metrics(cfg.models_dir, args.key, y_test, model.predict(X_test))

    out = loader.model_path(cfg.models_dir, f"model_{args.key}.joblib")
    joblib.dump(model, out)
    print(f"Saved model_{args.key}.joblib")


if __name__ == "__main__":
    main()
