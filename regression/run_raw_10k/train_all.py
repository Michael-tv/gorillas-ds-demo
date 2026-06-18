import os, subprocess, sys, time

HERE       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "..", "models")
base_env   = {**os.environ, "PYTHONPATH": HERE}

BASE_MODELS = {
    "linear_regression":       ("train_linear_regression.py",    ""),
    # "ridge":                   ("train_ridge.py",                ""),
    # "lasso":                   ("train_lasso.py",                ""),
    "decision_tree":           ("train_decision_tree.py",         ""),
    "decision_tree_overfit":   ("train_decision_tree_overfit.py", ""),
    "knn":                     ("train_knn.py",                   ""),
    # "polynomial":              ("train_polynomial.py",            ""),
    "random_forest":           ("train_random_forest.py",         ""),
    "mlp":                     ("train_mlp.py",                   ""),
    "xgboost":                 ("train_xgboost.py",               ""),
}
RANGE_MODELS = {
    "linear_regression_range": ("train_linear_regression.py",    "range"),
    "decision_tree_range":     ("train_decision_tree.py",         "range"),
    "knn_range":               ("train_knn.py",                   "range"),
    # "polynomial_range":        ("train_polynomial.py",            "range"),
    "random_forest_range":     ("train_random_forest.py",         "range"),
    # "xgboost_range":           ("train_xgboost.py",               "range"),
}
NO_OUTLIER_MODELS = {
    "mlp_no_outlier":            ("train_mlp.py",                  "no_outlier"),
    "decision_tree":           ("train_decision_tree.py",         "no_outlier"),
    "decision_tree_overfit":   ("train_decision_tree_overfit.py", "no_outlier"),
    "knn":                     ("train_knn.py",                   "no_outlier"),
    "polynomial":              ("train_polynomial.py",            "no_outlier"),
}

TRAIN_BASE       = True
TRAIN_RANGE      = True
TRAIN_NO_OUTLIER = True

ALL_MODELS = {}
if TRAIN_BASE:       ALL_MODELS.update(BASE_MODELS)
if TRAIN_RANGE:      ALL_MODELS.update(RANGE_MODELS)
if TRAIN_NO_OUTLIER: ALL_MODELS.update(NO_OUTLIER_MODELS)

results = []


def run(name, script, clean=""):
    start   = time.time()
    print(f"\n{'='*60}")
    print(f"  {name}  (clean={clean!r})")
    print(f"{'='*60}")
    env     = {**base_env, "TRAIN_MODEL_NAME": name, "TRAIN_CLEAN": clean}
    ret     = subprocess.run([sys.executable, os.path.join(MODELS_DIR, script)], env=env)
    elapsed = time.time() - start
    status  = "OK" if ret.returncode == 0 else "FAILED"
    results.append((name, status, elapsed))


if not os.path.isfile(os.path.join(HERE, "training_data", "training_data.csv")):
    print(); print(f"  [error] No training data found in: {HERE}")
    print(  "  Run:    python generate_data.py"); print(); sys.exit(1)

for name, (script, clean) in ALL_MODELS.items():
    run(name, script, clean)

print(f"\n{'='*60}\n  Summary\n{'='*60}")
print(f"  {'Model':<35} {'Status':<8} {'Time':>6}\n  {'-'*52}")
for name, status, elapsed in results:
    print(f"  {name:<35} {status:<8} {elapsed:>5.1f}s")
