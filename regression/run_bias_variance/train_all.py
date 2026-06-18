"""
Trains the underfitting and overfitting models, then sweeps max_depth
across [1..20, None] to plot the full bias-variance U-curve.
"""
import os
import sys
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_utils import load_data

# ── Train underfitting and overfitting models ─────────────────────────────────
scripts = [
    ("train_underfitting.py", "Underfitting (depth=1)"),
    ("train_overfitting.py",  "Overfitting  (depth=None)"),
]
results = []
for script, _ in scripts:
    print(f"\n{'='*60}")
    print(f"  {script}")
    print(f"{'='*60}")
    start = time.time()
    ret   = subprocess.run([sys.executable, os.path.join(HERE, script)])
    elapsed = time.time() - start
    results.append((script, "OK" if ret.returncode == 0 else "FAILED", elapsed))

print(f"\n{'='*60}")
print(f"  Summary")
print(f"{'='*60}")
print(f"  {'Script':<30} {'Status':<8} {'Time':>6}")
print(f"  {'-'*46}")
for script, status, elapsed in results:
    print(f"  {script:<30} {status:<8} {elapsed:>5.1f}s")

# ── Validation curve: sweep depths ───────────────────────────────────────────
print("\nSweeping max_depth for validation curve...")
X_train, X_test, y_train, y_test = load_data()

DEPTHS     = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, None]
train_maes = []
test_maes  = []

for d in DEPTHS:
    m = DecisionTreeRegressor(max_depth=d, random_state=42)
    m.fit(X_train, y_train)
    train_maes.append(mean_absolute_error(y_train, m.predict(X_train)))
    test_maes.append( mean_absolute_error(y_test,  m.predict(X_test)))

x_labels = [str(d) if d is not None else "None" for d in DEPTHS]
xs        = list(range(len(DEPTHS)))
best_idx  = int(np.argmin(test_maes))

print(f"\n{'='*60}")
print(f"  Bias-Variance Curve")
print(f"{'='*60}")
print(f"  {'depth':>6}  {'Train MAE':>10}  {'Test MAE':>10}")
print(f"  {'-'*30}")
for i, (lbl, tr, te) in enumerate(zip(x_labels, train_maes, test_maes)):
    flag = "  << best test" if i == best_idx else ""
    print(f"  {lbl:>6}  {tr:>10.3f}  {te:>10.3f}{flag}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(xs, train_maes, "o-", color="steelblue", linewidth=2, label="Train MAE")
ax.plot(xs, test_maes,  "s-", color="tomato",    linewidth=2, label="Test MAE")
ax.axvline(best_idx, color="green", linestyle="--", linewidth=1.5,
           label=f"Best test depth ({x_labels[best_idx]})")

ax.axvspan(-0.5,              best_idx - 0.5, alpha=0.07, color="blue",
           label="High bias (underfitting)")
ax.axvspan(best_idx + 0.5,   xs[-1] + 0.5,   alpha=0.07, color="red",
           label="High variance (overfitting)")

ax.set_xticks(xs)
ax.set_xticklabels(x_labels)
ax.set_xlabel("max_depth")
ax.set_ylabel("MAE (m/s)")
ax.set_title("Bias-Variance Tradeoff -- Decision Tree max_depth sweep",
             fontweight="bold")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
