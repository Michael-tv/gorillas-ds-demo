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
    "landing_distance_m", "target_distance_m", "hit_target", "is_outlier", "group_id",
]
# group_id: rows sharing a value were drawn under correlated conditions and
# must not be split across train/test independently. This path's rows are
# i.i.d. (each row's inputs drawn independently), so generate_rows() gives
# every row its own unique id. Contrast gorillas.py, where 32 throws share a
# board's wind and skyline, so its group_id is genuinely shared.

MASS_COLUMN_INDEX = 4  # never corrupt to exactly 0 -- engineered features divide by it


def _make_build_row(hit_tolerance):
    def build_row(rng, shot, i):
        hit = (i % 2 == 0)
        if hit:
            offset = rng.uniform(-(hit_tolerance - 0.1), hit_tolerance - 0.1)
        else:
            sign   = rng.choice([-1, 1])
            offset = sign * rng.uniform(hit_tolerance + 5.0, hit_tolerance + 80.0)
        target = round(shot.landing_x + offset, 4)

        values = [shot.v, shot.el, shot.wind_speed, float(shot.wind_dir_norm), shot.mass, shot.radius,
                  shot.Cd, shot.launch_h, shot.landing_h, shot.landing_x, target]
        return values, [int(hit)]
    return build_row


def generate(n, seed, elevation_dist=DEFAULT_ELEVATION_DIST, hit_tolerance=HIT_TOLERANCE, input_mode="VELOCITY"):
    """Generate `n` rows deterministically from `seed`.

    Seeds its own random.Random rather than relying on the caller having
    seeded the global `random` module -- two calls with the same arguments
    always produce the same DataFrame.

    input_mode="VELOCITY" (default) draws launch speed directly.
    input_mode="EFFORT" derives it from a capped force model instead --
    see data_generation/sampling.py.
    """
    rng = random.Random(seed)
    rows = generate_rows(rng, n, elevation_dist, _make_build_row(hit_tolerance),
                          no_zero_indices={MASS_COLUMN_INDEX}, input_mode=input_mode)
    # generate_rows returns rows grouped by section (normal, then gravity
    # outliers, then data errors) -- shuffle so a prefix slice (loader.py's
    # n_samples) keeps the same outlier mix instead of slicing out only
    # normal rows.
    rng.shuffle(rows)
    df = pd.DataFrame(rows, columns=COLUMNS)
    hits = int(df["hit_target"].sum())
    print(f"  Hits : {hits}   Misses : {len(df) - hits}   Tolerance : +/-{hit_tolerance} m")
    # Validated here (not just in the CLI) so any in-process caller gets the
    # same guarantee. Imported inside the function because contract.py
    # imports this module for COLUMNS (avoids a circular import).
    from data_generation.contract import check_contract
    return check_contract(df, source="generate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--elevation-mean", type=float, default=DEFAULT_ELEVATION_DIST[0])
    parser.add_argument("--elevation-std", type=float, default=DEFAULT_ELEVATION_DIST[1])
    parser.add_argument("--hit-tolerance", type=float, default=HIT_TOLERANCE)
    parser.add_argument("--seed", type=int, default=42, help="seeds this run's own random.Random for reproducible datasets")
    parser.add_argument("--input-mode", choices=["EFFORT", "VELOCITY"], default="VELOCITY",
                         help="VELOCITY samples launch speed directly; EFFORT derives it "
                              "from a capped force model (see sampling.py)")
    args = parser.parse_args()
    df = generate(args.n, args.seed, elevation_dist=(args.elevation_mean, args.elevation_std),
                   hit_tolerance=args.hit_tolerance, input_mode=args.input_mode)
    write_parquet(df, resolve_output(args.out))
