"""Read metrics_linear_regression.csv and plot the real (bootstrap) bias^2 /
variance / noise decomposition across n_samples tiers -- the rigorous
counterpart to experiment_row_count/analysis.py's train/test-gap proxy plot.
See train_linear_regression.py's docstring for the method.

Self-contained, run by hand (not in the DAG), matching
evaluation/regression/compare_models.py's style: no PYTHONPATH wrapper, no
repo imports, just pandas + matplotlib on the CSV this experiment's dvc.yaml
already produced.

    python experiments/regression/effort/experiment_bias_variance_bootstrap/analysis.py
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(HERE, "..", "..", "..", "..", "experiments",
                             "regression", "effort", "experiment_bias_variance_bootstrap",
                             "results", "metrics", "metrics_linear_regression.csv")

if not os.path.isfile(METRICS_PATH):
    print(f"[error] {METRICS_PATH} not found -- run `dvc repro` in this folder first.")
    raise SystemExit(1)

df = pd.read_csv(METRICS_PATH)

# ── Console table ────────────────────────────────────────────────────────────
C = 12
print("\nexperiment_bias_variance_bootstrap -- Linear Regression (bootstrap decomposition)")
print("─" * (10 + C * 4))
print(f"{'n_samples':<10}{'bias2':>{C}}{'variance':>{C}}{'mean_mse':>{C}}{'noise_est':>{C}}")
for _, row in df.iterrows():
    print(f"{int(row['n_samples']):<10}{row['bias2']:>{C}.3f}{row['variance']:>{C}.3f}"
          f"{row['mean_test_mse']:>{C}.3f}{row['noise_estimate']:>{C}.3f}")

bias_share = (df["bias2"] / df["mean_test_mse"]).mean()
read = (f"bias-dominated: bias2 is {bias_share:.0%} of test MSE on average -- "
        f"more data won't move this model's error much" if bias_share > 0.7
        else f"variance plays a real role: bias2 is only {bias_share:.0%} of test MSE on average")
print(f"\nRead: {read}\n")

# ── Plot: stacked bias2/variance/noise vs n_samples ─────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
x = df["n_samples"].astype(str)
ax.bar(x, df["bias2"], label="bias²", color="#DD8452")
ax.bar(x, df["variance"], bottom=df["bias2"], label="variance", color="#55A868")
ax.bar(x, df["noise_estimate"].clip(lower=0), bottom=df["bias2"] + df["variance"],
       label="noise (residual)", color="#8C8C8C")
ax.plot(x, df["mean_test_mse"], "o--", color="black", linewidth=1, markersize=4, label="mean test MSE")

ax.set_title("experiment_bias_variance_bootstrap\nLinear Regression: bias² vs variance across n_samples",
             fontsize=11, fontweight="bold")
ax.set_xlabel("n_samples")
ax.set_ylabel("m²/s² (squared error)")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3, linewidth=0.5)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()

out = os.path.join(HERE, "analysis.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
