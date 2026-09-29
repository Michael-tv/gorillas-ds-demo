"""Read every metrics_<model>.csv this experiment's dvc.yaml produces and
plot the row-count learning curves: test MAE, train MAE, and the
variance_proxy gap against n_samples, one line per model. Answers "does more
data help" directly from the numbers instead of eyeballing a printed table.

Self-contained, run by hand (not in the DAG), matching
evaluation/regression/compare_models.py's style: no PYTHONPATH wrapper, no
repo imports, just pandas + matplotlib on the CSVs this experiment's
dvc.yaml already produced.

    python models/regression/effort/experiment_row_count/analysis.py
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "..", "..", "..", "..", "experiments_results",
                           "regression", "effort", "experiment_row_count", "models")

# Every metrics_<model>.csv actually present -- stays in sync with whichever
# train_<model>.py stages are uncommented in this folder's dvc.yaml, rather
# than hardcoding a model list that drifts from it.
MODELS = sorted(
    fname[len("metrics_"):-len(".csv")]
    for fname in os.listdir(MODELS_DIR)
    if fname.startswith("metrics_") and fname.endswith(".csv")
) if os.path.isdir(MODELS_DIR) else []

if not MODELS:
    print(f"[error] No metrics_*.csv under {MODELS_DIR} -- run `dvc repro` in this folder first.")
    raise SystemExit(1)

frames = {model: pd.read_csv(os.path.join(MODELS_DIR, f"metrics_{model}.csv")) for model in MODELS}

# ── Console table ────────────────────────────────────────────────────────────
C = 10
print("\nexperiment_row_count -- learning curves (no-outlier effort pool)")
for model, df in frames.items():
    print(f"\n{model}")
    print("─" * (10 + C * 4))
    print(f"{'n_samples':<10}{'test_mae':>{C}}{'train_mae':>{C}}{'variance_proxy':>{C+5}}")
    for _, row in df.iterrows():
        print(f"{int(row['n_samples']):<10}{row['mae']:>{C}.3f}{row['train_mae']:>{C}.3f}{row['variance_proxy']:>{C+5}.3f}")

    # Read bias/variance off variance_proxy's magnitude, not the raw test-MAE
    # trend -- a small, noisy test MAE swing across tiers can look like
    # "moving with n_samples" even for a bias-limited model (linear_regression
    # here is the example: test MAE bounces +/-0.4 m/s tier to tier with no
    # real trend, which a trend-only read would misclassify as
    # variance-limited). variance_proxy's average size relative to the error
    # itself is the actual bias/variance signal.
    vp_share = df["variance_proxy"].abs().mean() / df["mae"].mean()
    read = (f"bias-limited: variance proxy averages {vp_share:.0%} of mean test MAE -- "
            f"train and test track closely, more data is unlikely to help" if vp_share < 0.15
            else f"variance plays a real role: variance proxy averages {vp_share:.0%} of mean test MAE")
    print(f"Read: {read}")
print()

# ── Plot: one row per model -- (test vs train MAE) | (variance proxy) ───────
_COLORS = {"test": "#DD8452", "train": "#4C72B0", "gap": "#55A868"}
fig, axes = plt.subplots(len(MODELS), 2, figsize=(10, 3.5 * len(MODELS)), squeeze=False)
fig.suptitle("experiment_row_count -- test/train MAE and variance proxy vs n_samples",
             fontsize=12, fontweight="bold")

for i, model in enumerate(MODELS):
    df = frames[model]
    ax_mae, ax_gap = axes[i]

    ax_mae.plot(df["n_samples"], df["mae"], "o-", color=_COLORS["test"], label="test MAE")
    ax_mae.plot(df["n_samples"], df["train_mae"], "o--", color=_COLORS["train"], label="train MAE")
    ax_mae.set_title(f"{model} -- MAE", fontsize=10, fontweight="bold")
    ax_mae.set_xlabel("n_samples")
    ax_mae.set_ylabel("MAE (m/s)")
    ax_mae.legend(fontsize=8)
    ax_mae.grid(alpha=0.3, linewidth=0.5)
    ax_mae.spines[["top", "right"]].set_visible(False)

    ax_gap.bar(df["n_samples"].astype(str), df["variance_proxy"], color=_COLORS["gap"], alpha=0.85)
    ax_gap.axhline(0, color="black", linewidth=0.6)
    ax_gap.set_title(f"{model} -- variance proxy (test - train)", fontsize=10, fontweight="bold")
    ax_gap.set_xlabel("n_samples")
    ax_gap.set_ylabel("MAE gap (m/s)")
    ax_gap.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax_gap.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out = os.path.join(HERE, "analysis.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
