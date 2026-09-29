"""
Input-output relationship plots for the training split of run_raw_10k.

Applies the same 80/20 train_test_split used by train_all.py so the plots
reflect exactly what the models are trained on, not the holdout test set.

Each panel: training points (blue) over withheld test points (grey) +
dashed line showing the true physics relationship computed from training-set
means while sweeping that feature across its range.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(BASE_DIR, "..", "..", "..")
sys.path.insert(0, REPO_ROOT)
import params
from physics import simulate

DATA         = params.data_path()
N_SAMPLES    = params.load_params()["n_samples"]
TEST_SIZE    = params.load_params()["test_size"]
RANDOM_STATE = 42
DT           = 0.02   # matches generate_data.py

FEATURES = [
    "launch_angle_deg", "wind_speed_ms", "wind_direction_norm",
    "mass_kg", "radius_m", "drag_coeff", "launch_height_m", "landing_height_m",
]
TARGET = "landing_distance_m"

# ── Load and split the full DataFrame so every column is available ────────────
df = params.take_samples(pd.read_parquet(DATA), N_SAMPLES)  # raises instead of silently truncating
df_train, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

X_train = df_train[FEATURES].values
y_train = df_train[TARGET].values
X_test  = df_test[FEATURES].values
y_test  = df_test[TARGET].values

# Column means from training rows only (includes initial_velocity_ms)
means = df_train.mean()

rng = np.random.default_rng(42)

print(f"Dataset  : {DATA}")
print(f"Total    : {len(df):,}  |  Train: {len(y_train):,}  |  Test: {len(y_test):,}")


def physics_curve(feat_name, x_vals):
    """Landing distance from the physics sim, sweeping feat_name over x_vals."""
    results = []
    for v in x_vals:
        speed  = means["initial_velocity_ms"]
        el     = means["launch_angle_deg"]
        ws            = means["wind_speed_ms"]
        wind_dir_norm = means["wind_direction_norm"]
        mass          = means["mass_kg"]
        radius        = means["radius_m"]
        Cd            = means["drag_coeff"]
        gz            = means["landing_height_m"] - means["launch_height_m"]

        if feat_name == "launch_angle_deg":      el            = v
        elif feat_name == "wind_speed_ms":       ws            = v
        elif feat_name == "wind_direction_norm": wind_dir_norm = v
        elif feat_name == "mass_kg":             mass          = v
        elif feat_name == "radius_m":            radius        = v
        elif feat_name == "drag_coeff":          Cd            = v
        elif feat_name == "launch_height_m":     gz            = means["landing_height_m"] - v
        elif feat_name == "landing_height_m":    gz            = v - means["launch_height_m"]

        wind_x = ws * wind_dir_norm
        traj   = simulate(speed, el, wind_x, mass, radius, Cd, dt=DT, ground_z=gz)
        results.append(traj[-1][1])
    return np.array(results)


# ── Pre-compute physics curves ────────────────────────────────────────────────
N_COLS     = 4
n_features = len(FEATURES)
n_rows     = (n_features + N_COLS - 1) // N_COLS

curves = {}
for feat, x_col in zip(FEATURES, X_train.T):
    x_c = np.linspace(x_col.min(), x_col.max(), 100)
    curves[feat] = (x_c, physics_curve(feat, x_c))

# ── Figure 1: full y-axis  |  Figure 2: y-axis clamped to physics range ──────
train_idx = rng.choice(len(y_train), size=min(1_600, len(y_train)), replace=False)
test_idx  = rng.choice(len(y_test),  size=min(400,   len(y_test)),  replace=False)

for zoomed in [False, True]:
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(N_COLS * 4, n_rows * 3 + 1))
    title_suffix = "physics y-scale" if zoomed else "full y-scale"
    fig.suptitle(
        f"Input → Output Relationships   |   run_raw   TRAINING SPLIT  "
        f"({len(y_train):,} / {len(df):,} samples)   [{title_suffix}]\n"
        f"Target: {TARGET}   |   grey = withheld test set   |   dashed = true physics (train means)",
        fontsize=11, fontweight="bold",
    )
    axes = axes.flatten()

    for i, feat in enumerate(FEATURES):
        ax            = axes[i]
        x_tr          = X_train[:, i]
        x_te          = X_test[:, i]
        x_curve, y_curve = curves[feat]

        ax.scatter(x_te[test_idx], y_test[test_idx],
                   alpha=0.15, s=6, color="dimgray", linewidths=0,
                   rasterized=True, label="test (held out)")
        ax.scatter(x_tr[train_idx], y_train[train_idx],
                   alpha=0.30, s=8, color="steelblue", linewidths=0,
                   rasterized=True, label="train")
        ax.plot(x_curve, y_curve, color="crimson", linewidth=1.8,
                linestyle="--", label="physics")

        if zoomed:
            margin = max((y_curve.max() - y_curve.min()) * 0.10, 0.5)
            ax.set_ylim(y_curve.min() - margin, y_curve.max() + margin)

        ax.set_title(feat, fontsize=9, fontweight="bold")
        ax.set_xlabel(feat, fontsize=8)
        ax.set_ylabel(TARGET if i % N_COLS == 0 else "", fontsize=8)
        ax.legend(fontsize=7, loc="best", handlelength=1.5)
        ax.tick_params(labelsize=7)

    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

plt.show()
