"""The one entrypoint for every classification algorithm, on every
classification run. See models/regression/train.py's docstring for the full
rationale -- same pattern, no --run feature-set split since classification
always uses the engineered features.

    python -m models.classification.train --run run --algorithm random_forest --key random_forest
"""
import argparse

import joblib

import params
import splitting
from models.classification.algorithms import ALGORITHMS
from models.classification.common import loader
from models.classification.runs import RUNS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, choices=sorted(RUNS),
                   help="which pool to train on")
    p.add_argument("--algorithm", required=True, choices=sorted(ALGORITHMS),
                   help="which model to build, and whose params.yaml search budget to use")
    p.add_argument("--key", required=True,
                   help="output name: experiments/.../model_<key>.joblib, metrics_<key>.csv")
    args = p.parse_args()

    cfg  = RUNS[args.run]
    algo = ALGORITHMS[args.algorithm]

    X, y, groups = loader.load_data(cfg.data, n_samples=cfg.n_samples)
    # Group-aware and stratified: a Gorillas pool shares one board's wind and
    # skyline across all 32 of its throws (AUDIT.md task 36/§5.1), and the
    # 5-7% hit rate means a plain split can starve a fold of positives
    # (task 37/§5.4). Degrades to an ordinary stratified split when the pool
    # has no group structure. See splitting.py.
    X_train, X_test, y_train, y_test, groups_train = splitting.split(
        X, y, groups, test_size=params.load_params()["test_size"],
        random_state=42, stratify=True)

    search = params.search_params("classification", args.algorithm)
    cv_folds = splitting.cv_for(search["cv"], X_train, y_train, groups_train, stratify=True)

    print(f"{algo.NAME} -- run={args.run}\n")
    model = algo.fit(X_train, y_train, {"n_iter": search["n_iter"], "cv": cv_folds})

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"{algo.NAME} -- test set")
    loader.print_metrics(cfg.models_dir, args.key, y_test, y_pred, y_prob)

    out = loader.model_path(cfg.models_dir, f"model_{args.key}.joblib")
    joblib.dump(model, out)
    print(f"Saved model_{args.key}.joblib")


if __name__ == "__main__":
    main()
