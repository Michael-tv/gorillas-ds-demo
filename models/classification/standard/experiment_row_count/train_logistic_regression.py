"""Train Logistic Regression at increasing sample sizes, to show how test
performance improves with more data.

TIERS are nested prefix slices of one seeded, pre-shuffled pool
(data/standard_velocity.parquet, 50000 rows) -- growing the sample size is
the only thing that changes between tiers. Same principle as
models/regression/standard/experiment_row_count/'s regression version;
mirrors params.yaml's n_samples convergence knob, swept here in one script
instead of requiring `dvc exp run --set-param n_samples=...` once per tier.

Writes one row per tier to metrics_<key>.csv (n_samples, precision, recall,
f1, pr_auc, roc_auc, accuracy, baseline_accuracy) and one
model_<key>_n<size>.joblib per tier, so every tier's model stays inspectable,
not just the metrics curve. Precision/recall/F1/PR-AUC lead over accuracy
for the same reason models/classification/common/loader.py's print_metrics
does -- a low hit rate makes "always predict miss" score deceptively high
accuracy (AUDIT.md task 37 / SS5.4).
"""
import csv
import os

import joblib
from sklearn.metrics import (accuracy_score, average_precision_score,
                              f1_score, precision_score, recall_score,
                              roc_auc_score)

import params
import splitting
from models.classification.algorithms import logistic_regression as algo
from models.classification.common import loader

_REPO_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DATA        = os.path.join(_REPO_ROOT, "data", "standard_velocity.parquet")
MODELS_DIR  = os.path.join(_REPO_ROOT, "experiments_results", "classification", "standard", "experiment_row_count", "models")
KEY         = "logistic_regression"
TIERS       = params.load_experiment_params(__file__)["tiers"]


def main():
    X_full, y_full, groups_full = loader.load_data(DATA)
    search = params.search_params("classification", KEY)
    test_size = params.load_params()["test_size"]

    os.makedirs(MODELS_DIR, exist_ok=True)
    rows = []
    for n in TIERS:
        X, y, groups = X_full[:n], y_full[:n], groups_full[:n]
        print(f"\n=== {algo.NAME} -- n_samples={n} ===")
        X_train, X_test, y_train, y_test, groups_train = splitting.split(
            X, y, groups, test_size=test_size, random_state=42, stratify=True)
        cv_folds = splitting.cv_for(search["cv"], X_train, y_train, groups_train, stratify=True)

        model = algo.fit(X_train, y_train, {"n_iter": search["n_iter"], "cv": cv_folds})
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        prec   = precision_score(y_test, y_pred, zero_division=0)
        rec    = recall_score(y_test, y_pred, zero_division=0)
        f1     = f1_score(y_test, y_pred, zero_division=0)
        acc    = accuracy_score(y_test, y_pred)
        auc    = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        baseline = 1.0 - float(y_test.mean())
        print(f"  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}  PR-AUC={pr_auc:.4f}  "
              f"Accuracy={acc:.4f} (baseline {baseline:.4f})")
        rows.append({"n_samples": n, "precision": prec, "recall": rec, "f1": f1,
                     "pr_auc": pr_auc, "roc_auc": auc, "accuracy": acc, "baseline_accuracy": baseline})

        joblib.dump(model, os.path.join(MODELS_DIR, f"model_{KEY}_n{n}.joblib"))

    metrics_path = os.path.join(MODELS_DIR, f"metrics_{KEY}.csv")
    with open(metrics_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_samples", "precision", "recall", "f1",
                                          "pr_auc", "roc_auc", "accuracy", "baseline_accuracy"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved metrics_{KEY}.csv ({len(rows)} tiers) and {len(rows)} models to {MODELS_DIR}")


if __name__ == "__main__":
    main()
