"""Generates the projectile-shot dataset
"""

import argparse
import random

import pandas as pd

from data_generation.io import resolve_output, write_parquet
from data_generation.sampling import generate_rows

DEFAULT_ELEVATION_DIST = (45.0, 20.0)
HIT_TOLERANCE = 5.0  # a shot is a "hit" if it lands within this many metres of the target

COLUMNS = [
    "initial_velocity_ms", "launch_angle_deg", "wind_speed_ms", "wind_direction_norm",
    "mass_kg", "radius_m", "drag_coeff", "launch_height_m", "landing_height_m",
    "landing_distance_m", "target_distance_m", "hit_target", "is_outlier",
]

MASS_COLUMN_INDEX = 4  # never corrupt to exactly 0 -- engineered features divide by it


def _make_build_row(hit_tolerance):
    def build_row(shot, i):
        hit = (i % 2 == 0)
        if hit:
            offset = random.uniform(-(hit_tolerance - 0.1), hit_tolerance - 0.1)
        else:
            sign   = random.choice([-1, 1])
            offset = sign * random.uniform(hit_tolerance + 5.0, hit_tolerance + 80.0)
        target = round(shot.landing_x + offset, 4)

        values = [shot.v, shot.el, shot.wind_speed, float(shot.wind_dir_norm), shot.mass, shot.radius,
                  shot.Cd, shot.launch_h, shot.landing_h, shot.landing_x, target]
        return values, [int(hit)]
    return build_row


def generate(n, elevation_dist=DEFAULT_ELEVATION_DIST, hit_tolerance=HIT_TOLERANCE):
    rows = generate_rows(n, elevation_dist, _make_build_row(hit_tolerance), no_zero_indices={MASS_COLUMN_INDEX})
    # generate_rows returns rows grouped by section (normal, then gravity
    # outliers, then data errors) -- shuffle so a prefix slice (see
    # models/regression/common/loader.py's `n_samples` slicing, used to nest
    # smaller sample-size tiers inside this pool for the convergence
    # experiment) keeps the same outlier mix and hit/miss balance as the full
    # pool, instead of slicing out only normal rows.
    random.shuffle(rows)
    df = pd.DataFrame(rows, columns=COLUMNS)
    hits = int(df["hit_target"].sum())
    print(f"  Hits : {hits}   Misses : {len(df) - hits}   Tolerance : +/-{hit_tolerance} m")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--elevation-mean", type=float, default=DEFAULT_ELEVATION_DIST[0])
    parser.add_argument("--elevation-std", type=float, default=DEFAULT_ELEVATION_DIST[1])
    parser.add_argument("--hit-tolerance", type=float, default=HIT_TOLERANCE)
    parser.add_argument("--seed", type=int, default=42, help="seeds the shared `random` module for reproducible datasets")
    args = parser.parse_args()
    random.seed(args.seed)
    df = generate(args.n, elevation_dist=(args.elevation_mean, args.elevation_std), hit_tolerance=args.hit_tolerance)
    write_parquet(df, resolve_output(args.out))
