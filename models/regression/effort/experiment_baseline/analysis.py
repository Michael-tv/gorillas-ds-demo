"""Read metrics_linear_regression.csv and report the baseline's bias/variance
proxy read. One row, not a sweep -- see ../experiment_row_count/analysis.py
for the tiered version of this same read. Self-contained, run by hand (not
in the DAG), matching evaluation/regression/compare_models.py's style: no
PYTHONPATH wrapper, no repo imports, just pandas + matplotlib on the CSV
this experiment's dvc.yaml already produced.

    python models/regression/effort/experiment_baseline/analysis.py
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(HERE, "..", "..", "..", "..", "experiments_results",
                             "regression", "effort", "experiment_baseline",
                             "models", "metrics_linear_regression.csv")

if not os.path.isfile(METRICS_PATH):
    print(f"[error] {METRICS_PATH} not found -- run `dvc repro` in this folder first.")
    raise SystemExit(1)

row = pd.read_csv(METRICS_PATH).iloc[0]
mae, train_mae, variance_proxy = row["mae"], row["train_mae"], row["variance_proxy"]

# ── Console summary ─────────────────────────────────────────────────────────
print("\nexperiment_baseline -- Linear Regression (n_samples=1000, no outliers)")
print("─" * 60)
print(f"{'Test MAE':<20}{mae:>10.3f} m/s")
print(f"{'Train MAE':<20}{train_mae:>10.3f} m/s   (bias proxy)")
print(f"{'Variance proxy':<20}{variance_proxy:>10.3f} m/s   (= test - train)")
print("─" * 60)

read = ("bias-limited: train and test MAE are close" if abs(variance_proxy) < 0.1 * mae
        else "variance-limited: a real gap between train and test MAE")
print(f"Read: {read}\n")

# ── Plot: train vs test MAE, with the gap called out ────────────────────────
fig, ax = plt.subplots(figsize=(4.5, 4.5))
bars = ax.bar(["Train MAE\n(bias proxy)", "Test MAE"], [train_mae, mae],
              color=["#4C72B0", "#DD8452"], edgecolor="white", linewidth=0.5)
ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
ax.set_ylabel("MAE (m/s)")
ax.set_title(f"experiment_baseline\nvariance proxy = {variance_proxy:+.3f} m/s", fontsize=10, fontweight="bold")
ax.grid(axis="y", alpha=0.3, linewidth=0.5)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()

out = os.path.join(HERE, "analysis.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
