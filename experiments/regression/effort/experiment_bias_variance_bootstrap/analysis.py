"""Plot the bootstrap bias^2/variance/noise decomposition across n_samples
tiers -- see train_linear_regression.py's and train_decision_tree.py's
docstrings for the method, and experiment_row_count/analysis.py for the
simpler train/test-gap proxy. Run by hand, not part of the DAG.

    python experiments/regression/effort/experiment_bias_variance_bootstrap/analysis.py
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(HERE, "results", "metrics")

MODELS = [
    ("linear_regression", "Linear Regression"),
    ("decision_tree", "Decision Tree"),
]


def _load(key):
    path = os.path.join(METRICS_DIR, f"metrics_{key}.csv")
    if not os.path.isfile(path):
        print(f"[skip] {path} not found -- run `dvc repro` in this folder first.")
        return None
    return pd.read_csv(path)


dfs = {}
for key, name in MODELS:
    df = _load(key)
    if df is None:
        continue
    dfs[key] = df

    # ── Console table ────────────────────────────────────────────────────────
    C = 12
    print(f"\nexperiment_bias_variance_bootstrap -- {name} (bootstrap decomposition)")
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

if "linear_regression" in dfs:
    # ── Plot: stacked bias2/variance/noise vs n_samples (Linear Regression) ──
    df = dfs["linear_regression"]
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

# ── Plot: classic bias²/variance/total-error curves, one panel per model,
# x-axis = n_samples (number of data points) rather than model complexity --
# see the module docstring for why: this experiment sweeps sample size, not
# model complexity. ──────────────────────────────────────────────────────────
active_models = [(key, name) for key, name in MODELS if key in dfs]
if active_models:
    fig, axes = plt.subplots(1, len(active_models), figsize=(6.5 * len(active_models), 5), squeeze=False)
    for ax, (key, name) in zip(axes[0], active_models):
        df = dfs[key]
        x = df["n_samples"]
        ax.plot(x, df["bias2"], "o-", color="#C44E52", linewidth=2, markersize=6, label="bias²")
        ax.plot(x, df["variance"], "o-", color="#4C9F9F", linewidth=2, markersize=6, label="variance")
        ax.plot(x, df["mean_test_mse"], "o-", color="#333333", linewidth=2, markersize=6, label="total error")

        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("n_samples (number of data points)")
        ax.set_ylabel("m²/s² (squared error)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(v):,}" for v in x])
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("experiment_bias_variance_bootstrap\nbias² / variance / total error vs number of data points",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    out = os.path.join(HERE, "bias_variance_curve.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

plt.show()
