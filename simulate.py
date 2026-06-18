import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from physics import simulate

# ── Config ────────────────────────────────────────────────────────────────────
speed = 40.0       # m/s
elevation = 45.0   # degrees
wind_x = 5.0       # m/s  (positive = tailwind, negative = headwind)
mass = 0.145       # kg
radius = 0.037     # m
Cd = 0.47
ground_z = 0.0     # m  (landing height relative to launch; positive = above, negative = below)
# ─────────────────────────────────────────────────────────────────────────────

traj = simulate(speed, elevation, wind_x, mass, radius, Cd, ground_z=ground_z)
ts, xs, zs = zip(*traj)

print(f"Landed at x={xs[-1]:.2f} m,  z={ground_z} m,  t={ts[-1]:.3f} s")

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, max(xs) * 1.05)
ax.set_ylim(min(ground_z - 1, min(zs) - 1), max(zs) * 1.15)
ax.set_xlabel("x (m)")
ax.set_ylabel("z (m)")
ax.set_title(f"{speed} m/s @ {elevation}°  |  wind_x={wind_x} m/s  |  landing z={ground_z} m")
ax.axhline(ground_z, color="saddlebrown", linewidth=2)
ax.grid(alpha=0.3)

trail, = ax.plot([], [], "b-", linewidth=1, alpha=0.6)
dot, = ax.plot([], [], "ro", markersize=8)

# Subsample so animation runs in ~5 s regardless of trajectory length
step = max(1, len(traj) // 250)
frames = range(0, len(traj), step)


def update(i):
    trail.set_data(xs[:i + 1], zs[:i + 1])
    dot.set_data([xs[i]], [zs[i]])
    return trail, dot


ani = animation.FuncAnimation(fig, update, frames=list(frames), interval=20, blit=True)
plt.tight_layout()
plt.show()
