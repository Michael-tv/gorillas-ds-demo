import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(HERE, "..", "..")

# ── Config ────────────────────────────────────────────────────────────────────
MODELS_TO_COMPARE = [
    # (domain, model_name); reads experiments/classification/<domain>/experiment_classification/.
    # For an n_samples sweep, use experiment_row_count/ instead of separate entries here.
    ("standard", "decision_tree"),
    # ("standard", "random_forest"),
    # ("standard", "mlp"),
    # ("effort",   "random_forest"),
    # ("velocity", "random_forest"),
]
# ─────────────────────────────────────────────────────────────────────────────

_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]


def load_metrics():
    records = []
    for domain, model in MODELS_TO_COMPARE:
        path = os.path.join(REPO_ROOT, "experiments", "classification", domain, "experiment_classification", "results", "metrics", f"metrics_{model}.csv")
        label = f"{domain}/{model}"
        if not os.path.isfile(path):
            print(f"  [skip] {label}: metrics file not found (train first)")
            continue
        try:
            row = pd.read_csv(path).iloc[0].to_dict()
        except Exception as e:
            print(f"  [warn] Could not read {path}: {e}")
            continue
        row["label"]  = label
        row["domain"] = domain
        row["model"]  = model
        records.append(row)
    return pd.DataFrame(records)


df = load_metrics()

if df.empty:
    print("No metrics found. Train models first (dvc repro).")
    raise SystemExit(1)

labels = df["label"].tolist()
colors = [_COLORS[i % len(_COLORS)] for i in range(len(labels))]

# ── Console table ─────────────────────────────────────────────────────────────
C = 9
divider = "─" * (35 + C * 5)
print(f"\n{'Classification Model Comparison':^{len(divider)}}")
print(divider)
print(f"{'Model':<35}{'Acc':>{C}}{'Prec':>{C}}{'Rec':>{C}}{'F1':>{C}}{'AUC':>{C}}")
print(divider)
for _, row in df.iterrows():
    auc = row.get("roc_auc", float("nan"))
    auc_str = f"{float(auc):>{C}.4f}" if not np.isnan(float(auc)) else f"{'—':>{C}}"
    print(f"{row['label']:<35}"
          f"{row['accuracy']:>{C}.4f}{row['precision']:>{C}.4f}"
          f"{row['recall']:>{C}.4f}{row['f1']:>{C}.4f}{auc_str}")
print(divider + "\n")

# ── Plot ──────────────────────────────────────────────────────────────────────
METRICS = [
    ("accuracy",  "Accuracy  ↑"),
    ("f1",        "F1 Score  ↑"),
    ("roc_auc",   "ROC-AUC   ↑"),
    ("precision", "Precision ↑"),
    ("recall",    "Recall    ↑"),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle("Classification — Model Comparison", fontsize=14, fontweight="bold")
fig.subplots_adjust(hspace=0.5, wspace=0.4, bottom=0.18)

bar_axes = list(axes[0]) + list(axes[1, :2])
axes[1, 2].set_visible(False)

x = np.arange(len(labels))

for ax, (metric, title) in zip(bar_axes, METRICS):
    vals = []
    for lbl in labels:
        v = df.loc[df["label"] == lbl, metric]
        vals.append(float(v.iloc[0]) if not v.empty and not np.isnan(float(v.iloc[0])) else np.nan)
    bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.12)
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
