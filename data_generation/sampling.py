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
import sys
from dataclasses import dataclass

from physics import simulate_landing
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

# EFFORT input mode: launch speed comes from a capped force model instead of
# being drawn directly, mirroring qbasic_gorillas/dosbox-datagen/gorilla.bas's
# own EFFORT mode (see its ThrowStrokeM#/MaxGorillaForce# comments). A gorilla
# (or here, the launcher) applies a force over a fixed stroke length, so
# v = sqrt(2 * force * stroke / mass) (work-energy over that stroke); the
# thrower controls effort as a 0-100% fraction of MAX_FORCE_N, which is
# calibrated -- same approach as gorilla.bas's MaxGorillaForce# -- so 100%
# effort at the mean sampled mass (MASS_DIST[0]) reaches SPEED_RANGE's own
# ceiling, keeping EFFORT and VELOCITY mode comparable at their extremes.
THROW_STROKE_M = 4.0    # m, arm + stroke length the force acts over
EFFORT_RANGE   = (20.0, 100.0)  # % of MAX_FORCE_N
MAX_FORCE_N    = SPEED_RANGE[1] ** 2 * MASS_DIST[0] / (2 * THROW_STROKE_M)  # N

OUTLIER_FRAC     = 0.02  # ~2% of N - unmodeled gravity variation
DATA_ERROR_FRAC  = 0.01  # ~1% of N - measurement / entry errors

# Cap on rejection-sampling attempts per draw (mass/radius/Cd/elevation/wind
# each retry until they land in their valid range). Without a cap a
# distribution that can never land in range hangs the stage forever with no
# diagnostic (AUDIT.md task 2). 10,000 attempts is generous for any of this
# module's distributions when configured sanely, and fails in well under a
# second when it is not.
MAX_REJECTION_ATTEMPTS = 10_000

# The cap alone only catches the *unreachable* case. Between "returns on the
# first try" and "never returns" sits a band that does neither, and it is the
# dangerous one: a range the distribution reaches only rarely completes
# normally while quietly reshaping the marginal into a truncated tail piled
# against the clip bound. Measured against ELEVATION_RANGE (AUDIT.md task 2b /
# finding N1):
#
#   gauss(45, 20) ->     1.1 attempts   the configured default
#   gauss(100, 20) ->    4.7 attempts   accepted draws: mean 73.4 deg, 30% >= 80
#   gauss(100, 5)  ->  694.4 attempts   ~600x the sampling cost, still completes
#   gauss(100, 2)  -> 10,000 attempts   raises
#
# So warn once per column when the observed acceptance rate is poor enough that
# the accepted draws can no longer be read as samples from the configured
# distribution. Warn, not raise: a deliberately truncated distribution is a
# legitimate thing to sample, it just must not be silent.
MIN_ACCEPTANCE_RATE = 0.10   # 1 accepted draw per 10 attempts
_warned_columns = set()


class UnreachableRangeError(RuntimeError):
    pass


class NonTerminatingShotError(RuntimeError):
    pass


def reset_rejection_warnings():
    """Forget which columns have already warned. Only needed by tests that
    assert on the warning; generation itself wants one warning per process."""
    _warned_columns.clear()


def _bounded(rng, draw, lo, hi, name):
    """Call draw() until the result falls in [lo, hi].

    Capped at MAX_REJECTION_ATTEMPTS so a distribution that can never land in
    range raises instead of hanging, and warns once per column when acceptance
    is rare enough that the accepted draws no longer represent the configured
    distribution (see MIN_ACCEPTANCE_RATE).
    """
    for attempts in range(1, MAX_REJECTION_ATTEMPTS + 1):
        x = draw()
        if lo <= x <= hi:
            if 1.0 / attempts < MIN_ACCEPTANCE_RATE and name not in _warned_columns:
                _warned_columns.add(name)
                print(
                    f"  WARNING: {name}: needed {attempts:,} draws to land one value "
                    f"in [{lo}, {hi}] (acceptance ~{100.0 / attempts:.2f}%). The "
                    f"configured distribution mostly misses this range, so the "
                    f"accepted values are a truncated tail piled against the bound, "
                    f"not samples from the distribution you configured.",
                    file=sys.stderr,
                )
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


def sample_shot(rng, elevation_dist, gravity=9.81, input_mode="VELOCITY"):
    """Draw one random physical sample and simulate its trajectory.

    input_mode="VELOCITY" draws launch speed directly (today's original
    behavior, draw order unchanged). input_mode="EFFORT" draws mass first
    (the force model needs it) and derives launch speed from a capped force
    model instead -- see EFFORT_RANGE/MAX_FORCE_N above.
    """
    if input_mode == "EFFORT":
        # mass drawn first -- the force model needs it. v is a DERIVED
        # quantity (force/mass, not an independent draw), so unlike every
        # other column here it is not rejection-sampled against a target
        # range: for a light draw it can naturally run above SPEED_RANGE's
        # ceiling, same as gorilla.bas's EFFORT mode has its own ceiling
        # (MaxThrowVelocity#) rather than matching VELOCITY mode's.
        mass = _bounded(rng, lambda: rng.gauss(*MASS_DIST), *MASS_RANGE, "mass_kg")
        effort_pct = rng.uniform(*EFFORT_RANGE)
        v = math.sqrt(2 * (effort_pct / 100.0 * MAX_FORCE_N) * THROW_STROKE_M / mass)
    else:
        v = rng.uniform(*SPEED_RANGE)
        mass = None
    el = _bounded(rng, lambda: rng.gauss(*elevation_dist), *ELEVATION_RANGE, "launch_angle_deg")
    ws = _bounded(rng, lambda: rng.weibullvariate(*WIND_WEIBULL),
                  WIND_SPEED_RANGE[0], WIND_SPEED_RANGE[1], "wind_speed_ms")
    wind_dir_norm = rng.choice(WIND_DIR_CHOICES)
    if mass is None:
        mass = _bounded(rng, lambda: rng.gauss(*MASS_DIST), *MASS_RANGE, "mass_kg")
    radius = _bounded(rng, lambda: rng.gauss(*RADIUS_DIST), *RADIUS_RANGE, "radius_m")
    Cd     = _bounded(rng, lambda: rng.gauss(*CD_DIST), *CD_RANGE, "drag_coeff")
    launch_h  = rng.uniform(*LAUNCH_HEIGHT_RANGE)
    landing_h = rng.uniform(*LANDING_HEIGHT_RANGE)

    wind_x      = ws * wind_dir_norm
    height_diff = landing_h - launch_h
    drag_param  = (0.5 * RHO * Cd * math.pi * radius**2) / mass
    traj, landed = simulate_landing(v, el, wind_x, mass, radius, Cd, DT,
                                    ground_z=height_diff, gravity=gravity)
    # If the integration never crossed the landing height, traj[-1] is wherever
    # the projectile happened to be when the loop gave up -- not a measurement.
    # Recording it anyway is what AUDIT.md §1.2 believed was happening to 727
    # rows; measuring it says otherwise (0 of 20,000 shots fail to land at the
    # configured ranges, and the negative distances are genuine headwind
    # landings). Raising keeps that true: widening SPEED_RANGE, raising DT, or
    # lowering physics.simulate's max_time could make it false, and this says so
    # instead of quietly writing meaningless rows (AUDIT.md task 13).
    if not landed:
        raise NonTerminatingShotError(
            f"shot did not land within the integrator's max_time: speed={v:.2f} "
            f"angle={el:.2f} wind_x={wind_x:.2f} mass={mass:.4f} radius={radius:.4f} "
            f"Cd={Cd:.3f} ground_z={height_diff:.2f} gravity={gravity} -- its "
            f"endpoint is not a landing point, so it must not become a row"
        )
    landing_x   = traj[-1][1]

    return Shot(v=v, el=el, wind_speed=ws, wind_dir_norm=wind_dir_norm, wind_x=wind_x,
                mass=mass, radius=radius, Cd=Cd, drag_param=drag_param,
                launch_h=launch_h, landing_h=landing_h, height_diff=height_diff,
                landing_x=landing_x)


def sample_gravity(rng):
    """Draw an out-of-distribution gravity value (Moon/Mars-like or super-Earth-like)."""
    return rng.uniform(*GRAVITY_LOW) if rng.random() < 0.5 else rng.uniform(*GRAVITY_HIGH)


def generate_rows(rng, n, elevation_dist, build_row, no_zero_indices=frozenset(), input_mode="VELOCITY"):
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

    Each returned row also gets its own group_id -- a running counter across
    all three sections, unique per row -- appended last, after is_outlier.
    This path's rows are independently drawn (no board-like correlation), so
    "every row is its own group" is the correct group_id, not a placeholder;
    see generate.py's COLUMNS comment.
    """
    n_outliers    = round(n * OUTLIER_FRAC)
    n_data_errors = round(n * DATA_ERROR_FRAC)
    n_normal      = n - n_outliers - n_data_errors
    rows = []
    row_id = 0

    for i in range(n_normal):
        shot = sample_shot(rng, elevation_dist, input_mode=input_mode)
        values, labels = build_row(rng, shot, i)
        rows.append(values + labels + ["none", str(row_id)])
        row_id += 1
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n}")

    print(f"\nGenerating {n_outliers} outliers (gravity variation)...")
    for i in range(n_outliers):
        shot = sample_shot(rng, elevation_dist, gravity=sample_gravity(rng), input_mode=input_mode)
        values, labels = build_row(rng, shot, i)
        rows.append(values + labels + ["gravity", str(row_id)])
        row_id += 1

    print(f"\nGenerating {n_data_errors} data errors...")
    for i in range(n_data_errors):
        shot = sample_shot(rng, elevation_dist, input_mode=input_mode)
        values, labels = build_row(rng, shot, i)
        values = corrupt_row(rng, values, no_zero_indices=no_zero_indices)
        rows.append(values + labels + ["data_error", str(row_id)])
        row_id += 1

    print(f"\nDone -- generated {n} samples")
    print(f"  Normal samples        : {n_normal}")
    print(f"  Outliers [gravity]    : {n_outliers}")
    print(f"  Outliers [data_error] : {n_data_errors}")
    return rows
