import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def _has_data(run_dir):
    return os.path.isfile(os.path.join(HERE, run_dir, "training_data", "training_data.csv"))


# ── Config — comment out what you don't need ─────────────────────────────────
RUNS = [
    "run_10k",
    "run_20k",
    "run_40k",
]
# ─────────────────────────────────────────────────────────────────────────────

results = []


def run(label, script_path):
    start = time.time()
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    ret     = subprocess.run([sys.executable, script_path])
    elapsed = time.time() - start
    status  = "OK" if ret.returncode == 0 else "FAILED"
    results.append((label, status, elapsed))


for run_dir in RUNS:
    if not _has_data(run_dir):
        print(f"\n[skip] {run_dir}/train_all.py — no training_data.csv"
              f" (run generate_data.py first)")
        results.append((f"{run_dir}/train_all.py", "skipped", 0.0))
        continue
    run(
        f"{run_dir}/train_all.py",
        os.path.join(HERE, run_dir, "train_all.py"),
    )

print(f"\n{'='*60}")
print(f"  Summary")
print(f"{'='*60}")
print(f"  {'Script':<30} {'Status':<8} {'Time':>6}")
print(f"  {'-'*47}")
for label, status, elapsed in results:
    time_str = f"{elapsed:>5.1f}s" if elapsed else "     -"
    print(f"  {label:<30} {status:<8} {time_str}")
