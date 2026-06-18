import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    ("Leaky DT", "train_leaky.py"),
    ("Clean DT", "train_clean.py"),
]

results = []

for name, script in SCRIPTS:
    print(f"\n{'='*60}")
    print(f"  {script}")
    print(f"{'='*60}")
    start = time.time()
    ret   = subprocess.run([sys.executable, os.path.join(HERE, script)])
    elapsed = time.time() - start
    status  = "OK" if ret.returncode == 0 else "FAILED"
    results.append((script, status, elapsed))

print(f"\n{'='*60}")
print(f"  Summary")
print(f"{'='*60}")
print(f"  {'Script':<25} {'Status':<8} {'Time':>6}")
print(f"  {'-'*42}")
for script, status, elapsed in results:
    print(f"  {script:<25} {status:<8} {elapsed:>5.1f}s")
