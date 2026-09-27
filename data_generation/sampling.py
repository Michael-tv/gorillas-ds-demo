"""Shared projectile-parameter sampling, used by every dataset generator.

Extracted from the ~4x copy-pasted generate_data.py bodies that used to live in
each regression/run_*/ and classification/run_*/ folder.
"""
import math
import random
from dataclasses import dataclass

from physics import simulate
from data_generation.outliers import corrupt_row

SPEED_RANGE          = (10.0, 80.0)   # m/s
ELEVATION_RANGE      = (5.0,  85.0)   # degrees (clip bounds)
WIND_SPEED_RANGE     = (0.0,  20.0)   # m/s (ceiling)
WIND_WEIBULL         = (9.0,  2.0)    # (scale lambda, shape k)
WIND_DIR_CHOICES     = (-1, 1)
LAUNCH_HEIGHT_RANGE  = (0.0,  10.0)   # m
LANDING_HEIGHT_RANGE = (0.0,  10.0)   # m

DT  = 0.02
RHO = 1.225  # kg/m^3

MASS_DIST    = (0.145, 0.050)
MASS_RANGE   = (0.01, 0.5)
RADIUS_DIST  = (0.037, 0.010)
RADIUS_RANGE = (0.005, 0.15)
CD_DIST      = (0.47,  0.10)
CD_RANGE     = (0.05, 1.0)

GRAVITY_LOW  = (1.0,  4.0)   # m/s^2 - Moon (1.6) to Mars (3.7)
GRAVITY_HIGH = (15.0, 25.0)  # m/s^2 - super-Earth to Jupiter (24.8)

OUTLIER_FRAC     = 0.02  # ~2% of N - unmodeled gravity variation
DATA_ERROR_FRAC  = 0.01  # ~1% of N - measurement / entry errors


@dataclass
class Shot:
    v: float
    el: float
    wind_speed: float
    wind_dir_norm: int
    wind_x: float
    mass: float
    radius: float
    Cd: float
    drag_param: float
    launch_h: float
    landing_h: float
    height_diff: float
    landing_x: float


def sample_shot(elevation_dist, gravity=9.81):
    """Draw one random physical sample and simulate its trajectory."""
    v = random.uniform(*SPEED_RANGE)
    while True:
        el = random.gauss(*elevation_dist)
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
    drag_param  = (0.5 * RHO * Cd * math.pi * radius**2) / mass
    traj        = simulate(v, el, wind_x, mass, radius, Cd, DT, ground_z=height_diff, gravity=gravity)
    landing_x   = traj[-1][1]

    return Shot(v=v, el=el, wind_speed=ws, wind_dir_norm=wind_dir_norm, wind_x=wind_x,
                mass=mass, radius=radius, Cd=Cd, drag_param=drag_param,
                launch_h=launch_h, landing_h=landing_h, height_diff=height_diff,
                landing_x=landing_x)


def sample_gravity():
    """Draw an out-of-distribution gravity value (Moon/Mars-like or super-Earth-like)."""
    return random.uniform(*GRAVITY_LOW) if random.random() < 0.5 else random.uniform(*GRAVITY_HIGH)


def generate_rows(n, elevation_dist, build_row, no_zero_indices=frozenset()):
    """Shared dataset-generation driver: samples n shots, injects gravity
    outliers and data-error corruption in the standard proportions, and
    returns the raw output rows. One shot per row -- no resampling/retrying.

    build_row(shot, i) -> (numeric_values, extra_labels)
        numeric_values: list of floats to store (and, for data-error rows,
                         to corrupt); `i` is the 0-based index within the
                         current section (normal/gravity/data_error), useful
                         e.g. for classification's hit/miss alternation.
        extra_labels:   trailing non-numeric columns (e.g. hit_target)
                         appended after numeric_values and before is_outlier;
                         never corrupted.
    """
    n_outliers    = round(n * OUTLIER_FRAC)
    n_data_errors = round(n * DATA_ERROR_FRAC)
    n_normal      = n - n_outliers - n_data_errors
    rows = []

    for i in range(n_normal):
        shot = sample_shot(elevation_dist)
        values, labels = build_row(shot, i)
        rows.append(values + labels + ["none"])
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n}")

    print(f"\nGenerating {n_outliers} outliers (gravity variation)...")
    for i in range(n_outliers):
        shot = sample_shot(elevation_dist, gravity=sample_gravity())
        values, labels = build_row(shot, i)
        rows.append(values + labels + ["gravity"])

    print(f"\nGenerating {n_data_errors} data errors...")
    for i in range(n_data_errors):
        shot = sample_shot(elevation_dist)
        values, labels = build_row(shot, i)
        values = corrupt_row(values, no_zero_indices=no_zero_indices)
        rows.append(values + labels + ["data_error"])

    print(f"\nDone -- generated {n} samples")
    print(f"  Normal samples        : {n_normal}")
    print(f"  Outliers [gravity]    : {n_outliers}")
    print(f"  Outliers [data_error] : {n_data_errors}")
    return rows
