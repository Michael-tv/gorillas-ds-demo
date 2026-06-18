import importlib.util
import os
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ROOT       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")

_TRAIN_SCRIPTS = {
    "linear_regression": "train_linear_regression.py",
    "ridge":             "train_ridge.py",
    "lasso":             "train_lasso.py",
    "decision_tree":     "train_decision_tree.py",
    "knn":               "train_knn.py",
    "polynomial":        "train_polynomial.py",
    "random_forest":     "train_random_forest.py",
    "mlp":               "train_mlp.py",
    "xgboost":           "train_xgboost.py",
}

# ── Config ────────────────────────────────────────────────────────────────────
PAIRS_TO_RUN = [
    # (run_name, model_name)
    ("run_raw_10k", "mlp"),
    # ("run_eng_500k", "mlp"),
]

compare_only = False   # set True to skip training and only show the comparison table
# ─────────────────────────────────────────────────────────────────────────────


def run(script_path, env=None):
    start   = time.time()
    name    = os.path.basename(script_path)
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    ret     = subprocess.run([sys.executable, script_path], env=env)
    elapsed = time.time() - start
    status  = "OK" if ret.returncode == 0 else "FAILED"
    return name, status, elapsed


# ── Training ──────────────────────────────────────────────────────────────────
if not compare_only:
    summary = []

    # Train each (version, model) pair — warn and skip if training data is missing
    for ver, model_name in PAIRS_TO_RUN:
        data_path = os.path.join(ROOT, ver, "training_data", "training_data.csv")
        if not os.path.exists(data_path):
            print(f"[skip] {ver}/{model_name}: training_data.csv not found — run {ver}/generate_data.py first")
            continue
        script = os.path.join(MODELS_DIR, _TRAIN_SCRIPTS[model_name])
        env    = {**os.environ, "PYTHONPATH": os.path.join(ROOT, ver)}
        summary.append(run(script, env=env))

    print(f"\n{'='*60}\n  Summary\n{'='*60}")
    print(f"  {'Script':<35} {'Status':<8} {'Time':>6}")
    print(f"  {'-'*50}")
    for name, status, elapsed in summary:
        print(f"  {name:<35} {status:<8} {elapsed:>5.1f}s")


# ── Load results ──────────────────────────────────────────────────────────────
def load_version_utils(ver):
    spec = importlib.util.spec_from_file_location(
        f"{ver}_train_utils",
        os.path.join(ROOT, ver, "train_utils.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


results = {}   # results[(ver, model_name)] = (mae, r2) or None

# Cache test sets per version to avoid re-loading CSV multiple times
_test_cache = {}

for ver, model_name in PAIRS_TO_RUN:
    key = (ver, model_name)
    tu  = load_version_utils(ver)

    if not os.path.exists(tu.DATA):
        print(f"  [{ver}] training_data.csv not found — run {ver}/generate_data.py first")
        results[key] = None
        continue

    if ver not in _test_cache:
        df = pd.read_csv(tu.DATA)
        X  = df[tu.FEATURES].values
        y  = df[tu.TARGET].values
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        _test_cache[ver] = (X_test, y_test)

    X_test, y_test = _test_cache[ver]

    path = os.path.join(ROOT, ver, "models", f"model_{model_name}.joblib")
    if not os.path.exists(path):
        results[key] = None
        continue

    data   = joblib.load(path)
    model  = data["model"]
    scaler = data.get("scaler")
    X_eval = scaler.transform(X_test) if scaler else X_test
    y_pred = model.predict(X_eval)
    results[key] = (mean_absolute_error(y_test, y_pred), r2_score(y_test, y_pred))


# ── Comparison table ──────────────────────────────────────────────────────────
unique_versions = list(dict.fromkeys(v for v, _ in PAIRS_TO_RUN))
unique_models   = list(dict.fromkeys(m for _, m in PAIRS_TO_RUN))

col_w   = 20
row_lbl = 25

header  = f"{'Model':<{row_lbl}}" + "".join(f"{v:^{col_w}}" for v in unique_versions)
divider = "-" * len(header)

print(f"\n\n{'='*len(header)}")
print("  Results  (MAE m/s  |  R2)")
print(f"{'='*len(header)}")
print(header)
print(divider)

for model_name in unique_models:
    row = f"{model_name:<{row_lbl}}"
    for ver in unique_versions:
        val = results.get((ver, model_name))
        if val is None:
            cell = f"{'n/a':^{col_w}}"
        else:
            mae, r2 = val
            cell = f"{f'{mae:.2f} | {r2:.3f}':^{col_w}}"
        row += cell
    print(row)

print(divider)
print("\nMAE = mean absolute error (lower is better)")
print("R2  = coefficient of determination (higher is better, max 1.0)")
