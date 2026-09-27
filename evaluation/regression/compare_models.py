import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(HERE, "..", "..")

# ── Config ────────────────────────────────────────────────────────────────────
MODELS_TO_COMPARE = [
    # (run_dir,   model_name)   -- run_dir is "run_raw" or "run_eng"; vary
    # n_samples via `dvc exp run --set-param n_samples=...` for the
    # convergence comparison instead of separate per-size directories.

    # Model names must be keys in dvc_models.yaml's regression_models -- the
    # sweep is the seven algorithms plus decision_tree_overfit and
    # random_forest_no_outlier (see that file's matrix rule). Two of the
    # commented examples here used to name decision_tree_range and
    # decision_tree_no_outlier, which task 3 removed.
    # ("run_raw", "knn"),
    ("run_raw", "decision_tree_overfit"),
    ("run_eng", "decision_tree"),
    # ("run_raw", "random_forest"),
    # ("run_raw", "xgboost"),
    # ("run_raw", "random_forest_no_outlier"),
    # ("run_eng", "random_forest"),
]
# ─────────────────────────────────────────────────────────────────────────────

_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]


def load_metrics():
    records = []
    for run, model in MODELS_TO_COMPARE:
        path = os.path.join(REPO_ROOT, "experiments", "regression", run, "models", f"metrics_{model}.csv")
        if not os.path.isfile(path):
            print(f"  [skip] {run}/{model}: metrics file not found (train first)")
            continue
        try:
            row = pd.read_csv(path).iloc[0].to_dict()
        except Exception as e:
            print(f"  [warn] Could not read {path}: {e}")
            continue
        row["label"] = f"{run}/{model}"
        row["run"]   = run
        row["model"] = model
        records.append(row)
    return pd.DataFrame(records)


df = load_metrics()

if df.empty:
    print("No metrics found. Train models first (dvc repro).")
    raise SystemExit(1)

labels = df["label"].tolist()
colors = [_COLORS[i % len(_COLORS)] for i in range(len(labels))]

# ── Console table ─────────────────────────────────────────────────────────────
C = 10
divider = "─" * (35 + C * 3)
print(f"\n{'Regression Model Comparison':^{len(divider)}}")
print(divider)
print(f"{'Model':<35}{'MAE':>{C}}{'MSE':>{C}}{'RMSE':>{C}}")
print(divider)
for _, row in df.iterrows():
    print(f"{row['label']:<35}{row['mae']:>{C}.3f}{row['mse']:>{C}.3f}{row['rmse']:>{C}.3f}")
print(divider + "\n")

# ── Plot ──────────────────────────────────────────────────────────────────────
METRICS = [
    ("mae",  "MAE (m/s)  ↓ lower is better"),
    ("mse",  "MSE (m²/s²) ↓ lower is better"),
    ("rmse", "RMSE (m/s) ↓ lower is better"),
]

fig, bar_axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Regression — Model Comparison", fontsize=14, fontweight="bold")
fig.subplots_adjust(wspace=0.35, bottom=0.22)

x = np.arange(len(labels))

for ax, (metric, title) in zip(bar_axes, METRICS):
    vals = [float(df.loc[df["label"] == lbl, metric].iloc[0]) for lbl in labels]
    bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [lbl.replace("/", "\n") for lbl in labels],
        fontsize=8, ha="right", rotation=45
    )
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)

out = os.path.join(HERE, "compare_models.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
