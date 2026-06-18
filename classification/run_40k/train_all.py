import os
import subprocess
import sys
import time

HERE       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "..", "models")

env = {**os.environ, "PYTHONPATH": HERE}

ALL_MODELS = {
    "logistic_regression": "train_logistic_regression.py",
    "decision_tree":       "train_decision_tree.py",
    "random_forest":       "train_random_forest.py",
    "knn":                 "train_knn.py",
    "mlp":                 "train_mlp.py",
    "xgboost":             "train_xgboost.py",
}

# ── Config — comment out what you don't need ─────────────────────────────────
MODELS_TO_TRAIN = list(ALL_MODELS.keys())
# ─────────────────────────────────────────────────────────────────────────────

results = []


def run(script_path, run_env=None):
    start   = time.time()
    name    = os.path.basename(script_path)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    ret     = subprocess.run([sys.executable, script_path], env=run_env)
    elapsed = time.time() - start
    status  = "OK" if ret.returncode == 0 else "FAILED"
    results.append((name, status, elapsed))


if not os.path.isfile(os.path.join(HERE, "training_data", "training_data.csv")):
    print()
    print(f"  [error] No training data found in: {HERE}")
    print(  "  Run:    python generate_data.py")
    print()
    sys.exit(1)

for name in MODELS_TO_TRAIN:
    run(os.path.join(MODELS_DIR, ALL_MODELS[name]), run_env=env)

print(f"\n{'='*60}")
print(f"  Summary")
print(f"{'='*60}")
print(f"  {'Script':<42} {'Status':<8} {'Time':>6}")
print(f"  {'-'*57}")
for script, status, elapsed in results:
    print(f"  {script:<42} {status:<8} {elapsed:>5.1f}s")