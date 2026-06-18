import csv
import math
import os
import random
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from physics import simulate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Number of samples (50/50 hit/miss balance) ───────────────────────────────
N = 20_000

# ── Outliers — unmodeled gravity variation ────────────────────────────────────
N_OUTLIERS    = 400          # appended after normal samples; ~2% of N
N_DATA_ERRORS = 200          # appended after outliers; ~1% of N  (measurement / entry errors)
GRAVITY_LOW  = (1.0,  4.0)  # m/s²  — Moon (1.6) to Mars (3.7)
GRAVITY_HIGH = (15.0, 25.0) # m/s²  — super-Earth to Jupiter (24.8)

# ── A shot is a "hit" if it lands within this many metres of the target ───────
HIT_TOLERANCE = 5.0

# ── Input ranges ─────────────────────────────────────────────────────────────
SPEED_RANGE          = (10.0, 80.0)
ELEVATION_RANGE      = (5.0,  85.0)  # launch angle clip bounds (degrees)
ELEVATION_DIST       = (45.0, 20.0)  # (mean, std) for N(μ,σ); clipped to ELEVATION_RANGE
WIND_SPEED_RANGE     = (0.0,  20.0)  # wind speed clip ceiling (m/s)
WIND_WEIBULL         = (9.0,  2.0)   # (scale λ, shape k); mean ≈ 8 m/s
WIND_DIR_CHOICES     = (-1, 1)       # direction: −1=headwind, 1=tailwind
LAUNCH_HEIGHT_RANGE  = (0.0,  10.0)  # absolute launch platform height (m)
LANDING_HEIGHT_RANGE = (0.0,  10.0)  # absolute landing platform height (m)

dt  = 0.02
RHO = 1.225  # kg/m³

MASS_DIST   = (0.145, 0.050)
MASS_RANGE  = (0.01, 0.5)     # kg            — resample until within [lo, hi]
RADIUS_DIST  = (0.037, 0.010)
RADIUS_RANGE = (0.005, 0.15)   # m             — resample until within [lo, hi]
CD_DIST      = (0.47,  0.10)
CD_RANGE     = (0.05, 1.0)     # dimensionless — resample until within [lo, hi]
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT = os.path.join(BASE_DIR, "training_data", "training_data.csv")
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)


def generate_sample(hit, gravity=9.81):
    while True:
        v             = random.uniform(*SPEED_RANGE)
        while True:
            el = random.gauss(*ELEVATION_DIST)
            if ELEVATION_RANGE[0] <= el <= ELEVATION_RANGE[1]:
                break
        while True:
            ws = random.weibullvariate(*WIND_WEIBULL)
            if ws <= WIND_SPEED_RANGE[1]:
                break
        wind_dir_norm = random.choice(WIND_DIR_CHOICES)
        while True:
            mass = random.gauss(*MASS_DIST)
            if MASS_RANGE[0] <= mass <= MASS_RANGE[1]:
                break
        while True:
            radius = random.gauss(*RADIUS_DIST)
            if RADIUS_RANGE[0] <= radius <= RADIUS_RANGE[1]:
                break
        while True:
            Cd = random.gauss(*CD_DIST)
            if CD_RANGE[0] <= Cd <= CD_RANGE[1]:
                break
        launch_h      = random.uniform(*LAUNCH_HEIGHT_RANGE)
        landing_h     = random.uniform(*LANDING_HEIGHT_RANGE)

        wind_x      = ws * wind_dir_norm
        height_diff = landing_h - launch_h

        drag_param = (0.5 * RHO * Cd * math.pi * radius**2) / mass
        traj       = simulate(v, el, wind_x, mass, radius, Cd, dt, ground_z=height_diff, gravity=gravity)
        landing_x  = traj[-1][1]

        if hit:
            offset = random.uniform(-(HIT_TOLERANCE - 0.1), HIT_TOLERANCE - 0.1)
            target = landing_x + offset
            if target > 0:
                return v, el, wind_x, drag_param, height_diff, round(target, 4), 1
        else:
            sign   = random.choice([-1, 1])
            offset = sign * random.uniform(HIT_TOLERANCE + 5.0, HIT_TOLERANCE + 80.0)
            target = landing_x + offset
            if target > 0:
                return v, el, wind_x, drag_param, height_diff, round(target, 4), 0


with open(OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "initial_velocity_ms", "launch_angle_deg", "wind_x_ms",
        "drag_param", "height_diff_m", "target_distance_m",
        "hit_target", "is_outlier",
    ])

    for i in range(N - N_OUTLIERS - N_DATA_ERRORS):
        hit = (i % 2 == 0)
        v, el, wind_x, drag_param, height_diff, target, label = generate_sample(hit)
        writer.writerow([f"{v:.4f}", f"{el:.4f}", f"{wind_x:.4f}",
                         f"{drag_param:.6f}", f"{height_diff:.4f}",
                         f"{target:.4f}", label, "none"])

        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{N}")

    # ── Outliers: unmodeled gravity variation ─────────────────────────────────
    print(f"\nGenerating {N_OUTLIERS} outliers...")
    for i in range(N_OUTLIERS):
        hit = (i % 2 == 0)
        g = random.uniform(*GRAVITY_LOW) if random.random() < 0.5 else random.uniform(*GRAVITY_HIGH)
        v, el, wind_x, drag_param, height_diff, target, label = generate_sample(hit, gravity=g)
        writer.writerow([f"{v:.4f}", f"{el:.4f}", f"{wind_x:.4f}",
                         f"{drag_param:.6f}", f"{height_diff:.4f}",
                         f"{target:.4f}", label, "gravity"])

    # ── Data errors: measurement / entry errors ───────────────────────────────
    print(f"\nGenerating {N_DATA_ERRORS} data errors...")
    for i in range(N_DATA_ERRORS):
        hit = (i % 2 == 0)
        v, el, wind_x, drag_param, height_diff, target, label = generate_sample(hit)
        row = [v, el, wind_x, drag_param, height_diff, float(target)]
        for idx in random.sample(range(len(row)), random.randint(1, 2)):
            fv = row[idx]
            c  = random.randint(0, 4)
            if c == 0:
                fv *= 10
            elif c == 1:
                fv *= 0.1
            elif c == 2:
                fv = -fv
            elif c == 3:
                fv += abs(fv) * random.uniform(5, 20)
            else:
                fv = 0.0
            row[idx] = fv
        writer.writerow([f"{row[0]:.4f}", f"{row[1]:.4f}", f"{row[2]:.4f}",
                         f"{row[3]:.6f}", f"{row[4]:.4f}", f"{row[5]:.4f}",
                         label, "data_error"])

print(f"\nDone -- saved {N} samples to {OUTPUT}")
print(f"  Hits : {N // 2}   Misses : {N // 2}   Tolerance : ±{HIT_TOLERANCE} m")
print(f"  Outliers [gravity]    : {N_OUTLIERS}")
print(f"  Outliers [data_error] : {N_DATA_ERRORS}")
