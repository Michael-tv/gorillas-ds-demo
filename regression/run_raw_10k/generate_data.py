import csv
import math
import os
import random
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from physics import simulate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

N             = 10_000
N_OUTLIERS    = 200          # ~2% of N — unmodeled gravity variation
N_DATA_ERRORS = 100          # ~1% of N — measurement / entry errors
GRAVITY_LOW   = (1.0,  4.0)  # m/s²  — Moon (1.6) to Mars (3.7)
GRAVITY_HIGH  = (15.0, 25.0) # m/s²  — super-Earth to Jupiter (24.8)

SPEED_RANGE          = (10.0, 80.0)
ELEVATION_RANGE      = (5.0,  85.0)  # degrees (clipped)
ELEVATION_DIST       = (45.0, 20.0)  # (mean, std) for N(μ,σ)
WIND_SPEED_RANGE     = (0.0,  20.0)  # m/s (ceiling)
WIND_WEIBULL         = (9.0,  2.0)   # (scale λ, shape k)
WIND_DIR_CHOICES     = (-1, 1)
LAUNCH_HEIGHT_RANGE  = (0.0,  10.0)  # m
LANDING_HEIGHT_RANGE = (0.0,  10.0)  # m

dt  = 0.02
RHO = 1.225  # kg/m³

MASS_DIST    = (0.145, 0.050)
MASS_RANGE   = (0.01, 0.5)
RADIUS_DIST  = (0.037, 0.010)
RADIUS_RANGE = (0.005, 0.15)
CD_DIST      = (0.47,  0.10)
CD_RANGE     = (0.05, 1.0)

OUTPUT = os.path.join(BASE_DIR, "training_data", "training_data.csv")
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)


def generate_sample(gravity=9.81):
    v = random.uniform(*SPEED_RANGE)
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
    launch_h  = random.uniform(*LAUNCH_HEIGHT_RANGE)
    landing_h = random.uniform(*LANDING_HEIGHT_RANGE)

    wind_x      = ws * wind_dir_norm
    height_diff = landing_h - launch_h
    traj        = simulate(v, el, wind_x, mass, radius, Cd, dt, ground_z=height_diff, gravity=gravity)
    landing_x   = traj[-1][1]
    return v, el, ws, wind_dir_norm, mass, radius, Cd, launch_h, landing_h, landing_x


with open(OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "initial_velocity_ms", "launch_angle_deg", "wind_speed_ms", "wind_direction_norm",
        "mass_kg", "radius_m", "drag_coeff",
        "launch_height_m", "landing_height_m", "landing_distance_m", "is_outlier",
    ])

    for i in range(N - N_OUTLIERS - N_DATA_ERRORS):
        v, el, ws, wd, mass, radius, Cd, lh, ldh, lx = generate_sample()
        writer.writerow([f"{v:.4f}", f"{el:.4f}", f"{ws:.4f}", f"{wd:.1f}",
                         f"{mass:.6f}", f"{radius:.6f}", f"{Cd:.6f}",
                         f"{lh:.4f}", f"{ldh:.4f}", f"{lx:.4f}", "none"])
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{N}")

    print(f"\nGenerating {N_OUTLIERS} outliers (gravity variation)...")
    for i in range(N_OUTLIERS):
        g = random.uniform(*GRAVITY_LOW) if random.random() < 0.5 else random.uniform(*GRAVITY_HIGH)
        v, el, ws, wd, mass, radius, Cd, lh, ldh, lx = generate_sample(gravity=g)
        writer.writerow([f"{v:.4f}", f"{el:.4f}", f"{ws:.4f}", f"{wd:.1f}",
                         f"{mass:.6f}", f"{radius:.6f}", f"{Cd:.6f}",
                         f"{lh:.4f}", f"{ldh:.4f}", f"{lx:.4f}", "gravity"])

    print(f"\nGenerating {N_DATA_ERRORS} data errors...")
    for i in range(N_DATA_ERRORS):
        v, el, ws, wd, mass, radius, Cd, lh, ldh, lx = generate_sample()
        row = [v, el, ws, float(wd), mass, radius, Cd, lh, ldh, lx]
        for idx in random.sample(range(len(row)), random.randint(1, 2)):
            fv = row[idx]
            c  = random.randint(0, 4)
            if   c == 0: fv *= 10
            elif c == 1: fv *= 0.1
            elif c == 2: fv = -fv
            elif c == 3: fv += abs(fv) * random.uniform(5, 20)
            else:        fv = 0.0
            row[idx] = fv
        writer.writerow([f"{row[0]:.4f}", f"{row[1]:.4f}", f"{row[2]:.4f}", f"{row[3]:.1f}",
                         f"{row[4]:.6f}", f"{row[5]:.6f}", f"{row[6]:.6f}",
                         f"{row[7]:.4f}", f"{row[8]:.4f}", f"{row[9]:.4f}", "data_error"])

print(f"\nDone -- saved {N} samples to {OUTPUT}")
print(f"  Normal samples        : {N - N_OUTLIERS - N_DATA_ERRORS}")
print(f"  Outliers [gravity]    : {N_OUTLIERS}")
print(f"  Outliers [data_error] : {N_DATA_ERRORS}")
