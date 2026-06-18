import importlib.util
import math
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from physics import simulate

_here     = os.path.dirname(os.path.abspath(__file__))
RUNS_ROOT = _here

# ── Config ────────────────────────────────────────────────────────────────────
initial_velocity = 45.0   # m/s
elevation        = 45.0   # degrees
wind_x           = 5.0    # m/s  (positive = tailwind, negative = headwind)
mass             = 0.145  # kg
radius           = 0.037  # m
Cd               = 0.47
launch_height    = 0.0    # m
landing_height   = 0.0    # m
ground_z         = landing_height - launch_height
target_distance  = 190.0  # m

MODELS_TO_RUN = [
    # (run_name, model_name)
    ("run_10k", "random_forest"),
    # ("run_10k", "logistic_regression"),
    # ("run_10k", "decision_tree"),
    # ("run_10k", "knn"),
    # ("run_10k", "mlp"),
    # ("run_10k", "xgboost"),
    # ("run_20k",  "random_forest"),
    # ("run_40k",  "random_forest"),
]
# ─────────────────────────────────────────────────────────────────────────────

RHO = 1.225

_predict_modules = {}


def get_predict_module(ver):
    if ver not in _predict_modules:
        path = os.path.join(RUNS_ROOT, ver, "predict.py")
        spec = importlib.util.spec_from_file_location(f"{ver}_predict", path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _predict_modules[ver] = mod
    return _predict_modules[ver]


# ── Run predictions ───────────────────────────────────────────────────────────
results = {}   # label -> (hit, prob)

for ver, model_name in MODELS_TO_RUN:
    label = f"{ver}/{model_name}"
    mod   = get_predict_module(ver)
    path  = os.path.join(RUNS_ROOT, ver, f"model_{model_name}.joblib")

    if not os.path.exists(path):
        print(f"[skip] {label}: model file not found")
        continue

    hit, prob = mod.predict_hit(
        initial_velocity, elevation, wind_x, mass, radius, Cd,
        ground_z, target_distance, model_name,
    )
    results[label] = (hit, prob)

# ── Shared physics trajectory ─────────────────────────────────────────────────
traj = simulate(initial_velocity, elevation, wind_x, mass, radius, Cd, ground_z=ground_z)
ts, xs, zs = zip(*traj)
actual_landing = xs[-1]

# ── Console summary ───────────────────────────────────────────────────────────
drag_param = (0.5 * RHO * Cd * math.pi * radius**2) / mass
print(f"\nInputs   : velocity={initial_velocity} m/s  angle={elevation}°"
      f"  wind_x={wind_x} m/s  drag_param={drag_param:.5f}"
      f"  ground_z={ground_z} m  target={target_distance} m")
print(f"Physics  : lands at {actual_landing:.2f} m\n")
print(f"{'Model':<30} {'Verdict':<8} {'Probability':>11}")
print("-" * 52)
for label, (hit, prob) in results.items():
    print(f"{label:<30} {'HIT' if hit else 'MISS':<8} {prob:>10.3f}")

if not results:
    print("(no entries to predict)")
    exit()

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

ax.set_xlim(0, max(max(xs), target_distance) * 1.08)
ax.set_ylim(min(ground_z - 1, min(zs) - 1), max(zs) * 1.2)
ax.set_xlabel("x (m)")
ax.set_ylabel("z (m)")
ax.set_title(
    f"velocity={initial_velocity} m/s  angle={elevation}°  wind_x={wind_x} m/s  "
    f"target={target_distance} m  |  lands at {actual_landing:.1f} m"
)
ax.grid(alpha=0.3)

# Wind indicator
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
ax.axvline(target_distance, color="black", linestyle="--", linewidth=2,
           label=f"Target: {target_distance:.1f} m")

# Model prediction labels stacked in upper-right
y_pos = 0.95
for label, (hit, prob) in results.items():
    color  = "limegreen" if hit else "tomato"
    verdict = "HIT" if hit else "MISS"
    ax.text(0.62, y_pos, f"{label}: {verdict} ({prob:.3f})",
            transform=ax.transAxes, fontsize=8, color=color, va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    y_pos -= 0.07

# Trajectory animation
trail, = ax.plot([], [], "b-", linewidth=1.5, alpha=0.7,
                 label=f"Trajectory (lands {actual_landing:.1f} m)")
dot,   = ax.plot([], [], "ro", markersize=8)
ax.legend(loc="upper left", fontsize=8)

step   = max(1, len(traj) // 250)
frames = list(range(0, len(traj), step))


def update(i):
    trail.set_data(xs[:i + 1], zs[:i + 1])
    dot.set_data([xs[i]], [zs[i]])
    return trail, dot


ani = animation.FuncAnimation(fig, update, frames=frames, interval=20, blit=True, repeat=False)
plt.tight_layout()
plt.show()
