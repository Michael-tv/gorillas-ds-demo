"""Shared projectile-parameter sampling, used by every dataset generator.

Extracted from the ~4x copy-pasted generate_data.py bodies that used to live in
each regression/run_*/ and classification/run_*/ folder.

Every draw takes an explicit `rng: random.Random` rather than using the global
`random` module. The caller owns seeding (generate.generate() creates one
`random.Random(seed)` per call and threads it through everything below), so
determinism does not depend on the caller having remembered to seed a global
singleton before importing this module -- see AUDIT.md task 2.
"""
import math
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

# Cap on rejection-sampling attempts per draw (mass/radius/Cd/elevation/wind
# each retry until they land in their valid range). Without a cap, a
# misconfigured distribution -- e.g. elevation_dist=(100.0, 5.0), whose mean
# sits outside ELEVATION_RANGE -- hangs the stage forever with no diagnostic
# (AUDIT.md task 2). 10,000 attempts is generous for any of this module's
# distributions when configured sanely, and fails in well under a second when
# it is not.
MAX_REJECTION_ATTEMPTS = 10_000


class UnreachableRangeError(RuntimeError):
    pass


def _bounded(rng, draw, lo, hi, name):
    """Call draw() until the result falls in [lo, hi], capped at
    MAX_REJECTION_ATTEMPTS so a distribution that rarely (or never) lands in
    range fails loudly instead of hanging."""
    for _ in range(MAX_REJECTION_ATTEMPTS):
        x = draw()
        if lo <= x <= hi:
            return x
    raise UnreachableRangeError(
        f"{name}: no draw landed in [{lo}, {hi}] after {MAX_REJECTION_ATTEMPTS:,} "
        f"attempts -- the sampling distribution is likely misconfigured to be "
        f"mostly or entirely outside this range"
    )


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


def sample_shot(rng, elevation_dist, gravity=9.81):
    """Draw one random physical sample and simulate its trajectory."""
    v  = rng.uniform(*SPEED_RANGE)
    el = _bounded(rng, lambda: rng.gauss(*elevation_dist), *ELEVATION_RANGE, "launch_angle_deg")
    ws = _bounded(rng, lambda: rng.weibullvariate(*WIND_WEIBULL),
                  WIND_SPEED_RANGE[0], WIND_SPEED_RANGE[1], "wind_speed_ms")
    wind_dir_norm = rng.choice(WIND_DIR_CHOICES)
    mass   = _bounded(rng, lambda: rng.gauss(*MASS_DIST), *MASS_RANGE, "mass_kg")
    radius = _bounded(rng, lambda: rng.gauss(*RADIUS_DIST), *RADIUS_RANGE, "radius_m")
    Cd     = _bounded(rng, lambda: rng.gauss(*CD_DIST), *CD_RANGE, "drag_coeff")
    launch_h  = rng.uniform(*LAUNCH_HEIGHT_RANGE)
    landing_h = rng.uniform(*LANDING_HEIGHT_RANGE)

    wind_x      = ws * wind_dir_norm
    height_diff = landing_h - launch_h
    drag_param  = (0.5 * RHO * Cd * math.pi * radius**2) / mass
    traj        = simulate(v, el, wind_x, mass, radius, Cd, DT, ground_z=height_diff, gravity=gravity)
    landing_x   = traj[-1][1]

    return Shot(v=v, el=el, wind_speed=ws, wind_dir_norm=wind_dir_norm, wind_x=wind_x,
                mass=mass, radius=radius, Cd=Cd, drag_param=drag_param,
                launch_h=launch_h, landing_h=landing_h, height_diff=height_diff,
                landing_x=landing_x)


def sample_gravity(rng):
    """Draw an out-of-distribution gravity value (Moon/Mars-like or super-Earth-like)."""
    return rng.uniform(*GRAVITY_LOW) if rng.random() < 0.5 else rng.uniform(*GRAVITY_HIGH)


def generate_rows(rng, n, elevation_dist, build_row, no_zero_indices=frozenset()):
    """Shared dataset-generation driver: samples n shots, injects gravity
    outliers and data-error corruption in the standard proportions, and
    returns the raw output rows. One shot per row -- no resampling/retrying.

    build_row(rng, shot, i) -> (numeric_values, extra_labels)
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
        shot = sample_shot(rng, elevation_dist)
        values, labels = build_row(rng, shot, i)
        rows.append(values + labels + ["none"])
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n}")

    print(f"\nGenerating {n_outliers} outliers (gravity variation)...")
    for i in range(n_outliers):
        shot = sample_shot(rng, elevation_dist, gravity=sample_gravity(rng))
        values, labels = build_row(rng, shot, i)
        rows.append(values + labels + ["gravity"])

    print(f"\nGenerating {n_data_errors} data errors...")
    for i in range(n_data_errors):
        shot = sample_shot(rng, elevation_dist)
        values, labels = build_row(rng, shot, i)
        values = corrupt_row(rng, values, no_zero_indices=no_zero_indices)
        rows.append(values + labels + ["data_error"])

    print(f"\nDone -- generated {n} samples")
    print(f"  Normal samples        : {n_normal}")
    print(f"  Outliers [gravity]    : {n_outliers}")
    print(f"  Outliers [data_error] : {n_data_errors}")
    return rows
