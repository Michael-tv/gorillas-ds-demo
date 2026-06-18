import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# ── Datasets — comment out what you don't need ───────────────────────────────
DATASETS = {
    "run_raw_10k":  "run_raw_10k/generate_data.py",
    "run_raw_20k":  "run_raw_20k/generate_data.py",
    "run_raw_40k":  "run_raw_40k/generate_data.py",
    "run_eng_10k":  "run_eng_10k/generate_data.py",
    "run_eng_20k":  "run_eng_20k/generate_data.py",
    "run_eng_40k":  "run_eng_40k/generate_data.py",
    "run_skewed":   "run_skewed/generate_data.py",
    # Note: run_leakage and run_bias_variance have no separate data generation —
    #       they use run_eng_30k/training_data.csv.  Generate run_eng_30k first.
    "run_eng_30k":  "run_eng_30k/generate_data.py",
}
# ─────────────────────────────────────────────────────────────────────────────

results = []


def run(name, script_path):
    start = time.time()
    print(f"\n{'='*60}")
    print(f"  {name}  ({script_path})")
    print(f"{'='*60}")
    ret     = subprocess.run([sys.executable, os.path.join(HERE, script_path)])
    elapsed = time.time() - start
    status  = "OK" if ret.returncode == 0 else "FAILED"
    results.append((name, status, elapsed))


for name, script in DATASETS.items():
    run(name, script)

print(f"\n{'='*60}")
print(f"  Summary")
print(f"{'='*60}")
print(f"  {'Dataset':<15} {'Status':<8} {'Time':>6}")
print(f"  {'-'*32}")
for name, status, elapsed in results:
    print(f"  {name:<15} {status:<8} {elapsed:>5.1f}s")
