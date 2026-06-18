import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# ── Datasets — comment out what you don't need ───────────────────────────────
DATASETS = {
    "run_10k":  "run_10k/generate_data.py",
    "run_20k":  "run_20k/generate_data.py",
    "run_40k":  "run_40k/generate_data.py",
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
print(f"  {'Dataset':<12} {'Status':<8} {'Time':>6}")
print(f"  {'-'*29}")
for name, status, elapsed in results:
    print(f"  {name:<12} {status:<8} {elapsed:>5.1f}s")
