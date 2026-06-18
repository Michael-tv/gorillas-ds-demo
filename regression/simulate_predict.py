import importlib.util
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from physics import simulate

_here     = os.path.dirname(os.path.abspath(__file__))
RUNS_ROOT = _here

# ── Config ────────────────────────────────────────────────────────────────────
elevation      = 80.0   # degrees
wind_x         = 0.0   # m/s  (positive = tailwind, negative = headwind)
mass           = 0.145  # kg
radius         = 0.037  # m
Cd             = 0.47
launch_height  = 0      # m  (absolute height of launch platform)
landing_height = 5      # m  (absolute height of landing platform)
ground_z       = landing_height - launch_height  # derived; passed to physics / predict
target_x       = 60     # m  (desired landing distance — models predict velocity to reach this)

MODELS_TO_RUN = [
    # (run_name, model_name)
    # ("physics", 30.0), # Physics only model
    # ("run_raw_10k",       "decision_tree"),
    # ("run_raw_10k",       "decision_tree_overfit"),
    ("run_raw_10k",       "knn"),
    ("run_raw_20k",       "knn"),
    ("run_raw_30k",       "knn"),
    # ("run_raw_10k",       "knn"),
    # ("run_raw_10k",       "random_forest"),
    # ("run_raw_10k",       "mlp"),
    # ("run_eng_100k",      "random_forest"),
    # ("run_eng_10k",       "random_forest"),

    # ("run_eng_500k",      "random_forest"),
    # ("run_bias_variance", "underfitting"),
    # ("run_bias_variance", "overfitting"),
    # ("run_skewed",        "skewed"),
    # ("run_skewed",        "balanced"),
    # ("run_leakage", "dt_leaky"),
    # ("run_leakage", "dt_clean"),
]
# ─────────────────────────────────────────────────────────────────────────────

_COLORS = [
    "tomato", "limegreen", "deepskyblue", "hotpink",
    "orange", "mediumpurple", "gold", "cyan", "coral", "yellowgreen",
]

_predict_modules = {}


def get_predict_module(ver):
    if ver not in _predict_modules:
        path = os.path.join(RUNS_ROOT, ver, "predict.py")
        spec = importlib.util.spec_from_file_location(f"{ver}_predict", path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _predict_modules[ver] = mod
    return _predict_modules[ver]


# ── Predictions / simulations ─────────────────────────────────────────────────
ml_results = {}   # label -> (velocity, actual_landing, trajectory, color)

for idx, (ver, model_name) in enumerate(MODELS_TO_RUN):
    label = f"{ver}/{model_name}"
    color = _COLORS[idx % len(_COLORS)]

    if ver == "physics":
        # model_name is the initial velocity; skip ML loading entirely
        vel  = float(model_name)
        traj = simulate(vel, elevation, wind_x, mass, radius, Cd, ground_z=ground_z)
        ml_results[label] = (vel, traj[-1][1], traj, color)
        continue

    mod  = get_predict_module(ver)
    path = os.path.join(RUNS_ROOT, ver, f"model_{model_name}.joblib")

    if not os.path.exists(path):
        print(f"[skip] {label}: model file not found")
        continue

    ml_vel = mod.predict_velocity(elevation, wind_x, mass, radius, Cd, launch_height, landing_height, target_x, model_name)
    traj   = simulate(ml_vel, elevation, wind_x, mass, radius, Cd, ground_z=ground_z)
    ml_results[label] = (ml_vel, traj[-1][1], traj, color)

# ── Console summary ───────────────────────────────────────────────────────────
print(f"\ntarget: {target_x:.2f} m")
print(f"{'Model':<25} {'Velocity (m/s)':>14}  {'Lands at (m)':>12}  {'Error (m)':>10}")
print("-" * 67)
for label, (vel, land, _, __) in ml_results.items():
    print(f"{label:<25} {vel:>14.2f}  {land:>12.2f}  {land - target_x:>+10.2f}")

if not ml_results:
    print("(no entries to plot)")
    exit()

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))

all_x  = [target_x] + [v[1] for v in ml_results.values()]
all_zs = [z for _, _, traj, _ in ml_results.values() for _, _, z in traj]

x_min = min(0, min(all_x)) - 2
x_max = max(all_x) * 1.08
z_min = min(ground_z - 1, min(all_zs) - 1)
z_max = max(all_zs) * 1.2

ax.set_xlim(x_min, x_max)
ax.set_ylim(z_min, z_max)
ax.set_xlabel("x (m)")
ax.set_ylabel("z (m)")
ax.set_title("Model comparison")
ax.grid(alpha=0.3)

# Wind indicator — arrow in upper-left, pointing in the direction the wind blows
if wind_x == 0:
    ax.text(0.05, 0.92, "wind: 0 m/s", transform=ax.transAxes,
            fontsize=9, color="steelblue", va="center")
else:
    tail_x = 0.05 if wind_x > 0 else 0.22
    head_x = 0.22 if wind_x > 0 else 0.05
    ax.annotate(
        f"wind: {abs(wind_x):.1f} m/s",
        xy=(head_x, 0.92), xycoords="axes fraction",
        xytext=(tail_x, 0.92), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="steelblue", lw=2),
        fontsize=9, color="steelblue",
        ha="left" if wind_x > 0 else "right", va="center",
    )

ax.axhline(ground_z, color="saddlebrown", linewidth=2, label=f"Ground (z={ground_z} m)")
ax.axvline(target_x, color="black", linestyle="--", linewidth=2, label=f"Target: {target_x:.1f} m")

anim_artists = {}
for label, (ml_vel, ml_land, traj, color) in ml_results.items():
    t_xs   = [p[1] for p in traj]
    t_zs   = [p[2] for p in traj]
    trail, = ax.plot([], [], "-",  color=color, linewidth=1.5, alpha=0.8,
                     label=f"{label}: {ml_land:.1f} m")
    dot,   = ax.plot([], [], "o",  color=color, markersize=7)
    anim_artists[label] = (trail, dot, t_xs, t_zs)

ax.legend(loc="upper right", fontsize=8)

max_len = max(len(v[2]) for v in ml_results.values())
step    = max(1, max_len // 250)
frames  = list(range(0, max_len, step))


def update(frame_idx):
    artists = []
    for trail, dot, t_xs, t_zs in anim_artists.values():
        i = min(frame_idx, len(t_xs) - 1)
        trail.set_data(t_xs[:i + 1], t_zs[:i + 1])
        dot.set_data([t_xs[i]], [t_zs[i]])
        artists.extend([trail, dot])
    return artists


ani = animation.FuncAnimation(fig, update, frames=frames, interval=20, blit=True, repeat=False)
plt.tight_layout()
plt.show()
