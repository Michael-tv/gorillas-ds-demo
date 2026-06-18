import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE      = os.path.dirname(os.path.abspath(__file__))

# ── Config ────────────────────────────────────────────────────────────────────
MODELS_TO_COMPARE = [
    # (run_dir,        model_name)

    # ("run_raw_10k",  "knn"),
    # ("run_raw_10k",  "decision_tree_no"),
    ("run_raw_10k",  "decision_tree_overfit"),
    ("run_eng_10k",  "decision_tree"),
    # ("run_eng_10k",  "decision_tree_range"),

    # ("run_raw_20k",  "knn"),
    # ("run_raw_20k",  "decision_tree"),
    # ("run_eng_20k",  "decision_tree"),
    # ("run_eng_20k",  "decision_tree_range"),

    # ("run_raw_30k",  "knn"),
    # ("run_raw_30k",  "decision_tree"),
    # ("run_eng_10k",  "decision_tree"),
    # ("run_eng_30k",  "decision_tree"),
    # ("run_eng_30k",  "decision_tree_range"),

    # ("run_raw_10k",  "decision_tree_overfit"),
    # ("run_raw_10k",  "random_forest"),
    # ("run_raw_10k",  "xgboost"),
    # ("run_raw_20k",  "decision_tree"),
    # ("run_raw_20k",  "random_forest"),
    # ("run_raw_40k",  "decision_tree_no_outlier"),
    # ("run_raw_50k",  "decision_tree_no_outlier"),

    # ("run_eng_10k",  "random_forest"),
    # ("run_eng_20k",  "random_forest"),
    # ("run_raw_50k",  "decision_tree"),
]
# ─────────────────────────────────────────────────────────────────────────────

_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]


def load_metrics():
    records = []
    for run, model in MODELS_TO_COMPARE:
        path = os.path.join(HERE, run, "models", f"metrics_{model}.csv")
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
    print("No metrics found. Train models first (python train_all.py in a run directory).")
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
