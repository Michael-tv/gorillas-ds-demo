import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def _has_data(run_dir):
    return os.path.isfile(os.path.join(HERE, run_dir, "training_data", "training_data.csv"))


# ── Config — comment out what you don't need ─────────────────────────────────

# Standard runs: each has train_all.py that trains all uncommented models.
# Skipped automatically when training_data.csv is missing — run
# generate_all_data.py first.
STANDARD_RUNS = [
    "run_raw_10k",
    "run_raw_20k",
    "run_raw_40k",
    "run_eng_10k",
    "run_eng_20k",
    "run_eng_40k",
    "run_skewed",
]

# Concept-demo scripts: train the named concept models and save metrics CSVs.
# run_leakage and run_bias_variance use run_eng_30k data — no local
# training_data.csv required.
CONCEPT_SCRIPTS = [
    ("run_skewed",        "train_skewed.py"),
    ("run_skewed",        "train_balanced.py"),
    ("run_leakage",       "train_leaky.py"),
    ("run_leakage",       "train_clean.py"),
    ("run_bias_variance", "train_underfitting.py"),
    ("run_bias_variance", "train_overfitting.py"),
]

# ─────────────────────────────────────────────────────────────────────────────

results = []


def run(label, script_path, env=None):
    start = time.time()
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    ret     = subprocess.run([sys.executable, script_path], env=env)
    elapsed = time.time() - start
    status  = "OK" if ret.returncode == 0 else "FAILED"
    results.append((label, status, elapsed))


for run_dir in STANDARD_RUNS:
    if not _has_data(run_dir):
        print(f"\n[skip] {run_dir}/train_all.py — no training_data.csv"
              f" (run generate_all_data.py first)")
        results.append((f"{run_dir}/train_all.py", "skipped", 0.0))
        continue
    run(
        f"{run_dir}/train_all.py",
        os.path.join(HERE, run_dir, "train_all.py"),
    )

for run_dir, script in CONCEPT_SCRIPTS:
    run(
        f"{run_dir}/{script}",
        os.path.join(HERE, run_dir, script),
    )

print(f"\n{'='*60}")
print(f"  Summary")
print(f"{'='*60}")
print(f"  {'Script':<45} {'Status':<8} {'Time':>6}")
print(f"  {'-'*62}")
for label, status, elapsed in results:
    time_str = f"{elapsed:>5.1f}s" if elapsed else "     -"
    print(f"  {label:<45} {status:<8} {time_str}")
