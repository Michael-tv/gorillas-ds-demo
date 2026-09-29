"""Splits the standard pool into a low-angle slice and its complement.

The "skewed" demo trains on launch angles up to `skew.max_angle_deg` only, to
show a model extrapolating outside its training range. Emits two outputs
because the filter must run before the train/test split: the excluded rows
become the out-of-distribution test set, so "extrapolates poorly beyond the
cap" is measured rather than asserted.

A filter, not a second generated dataset -- only the angle distribution
changes, not every other sampled value or the row count.

The cap is one-sided: tree ensembles flatline at the boundary leaf mean past
the training range while linear models keep going, and a mid-range gap would
make that contrast far weaker.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import params
from data_generation.contract import check_contract
from data_generation.io import resolve_output, write_parquet

ANGLE_COLUMN = "launch_angle_deg"


def filter_skewed(df, max_angle_deg):
    """Return `(kept, holdout)` -- rows at or below the cap, and the rest."""
    mask = df[ANGLE_COLUMN] <= max_angle_deg
    return df[mask].copy(), df[~mask].copy()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_path", default=None,
                   help="pool to filter (default: the params.yaml training_data pool)")
    p.add_argument("--out", required=True, help="parquet for the kept (low-angle) rows")
    p.add_argument("--holdout-out", required=True,
                   help="parquet for the excluded rows -- the OOD test set")
    p.add_argument("--max-angle-deg", type=float, default=None,
                   help="default: params.yaml skew.max_angle_deg (sweepable via --set-param)")
    args = p.parse_args()

    in_path = args.in_path or params.data_path()
    if not os.path.isfile(in_path):
        sys.exit(f"  [error] pool not found: {in_path}\n  Run: dvc repro generate")

    max_angle = args.max_angle_deg
    if max_angle is None:
        max_angle = params.load_params()["skew"]["max_angle_deg"]

    # Validated on read as well as on write: the pool may predate this code.
    df = check_contract(pd.read_parquet(in_path), source=os.path.basename(in_path),
                        verbose=False)
    kept, holdout = filter_skewed(df, max_angle)

    print(f"  Skew filter    : {ANGLE_COLUMN} <= {max_angle}")
    print(f"  Kept (train)   : {len(kept):,} of {len(df):,} rows")
    print(f"  Holdout (OOD)  : {len(holdout):,} rows"
          f"  [{holdout[ANGLE_COLUMN].min():.1f}-{holdout[ANGLE_COLUMN].max():.1f} deg]"
          if len(holdout) else "  Holdout (OOD)  : 0 rows")

    # Catch an empty/single-class split here, not as a confusing failure downstream.
    if kept.empty:
        sys.exit(f"  [error] no rows at or below {max_angle} deg -- nothing to train on. "
                 f"Raise skew.max_angle_deg.")
    if holdout.empty:
        sys.exit(f"  [error] no rows above {max_angle} deg -- the holdout set would be "
                 f"empty, so 'extrapolates poorly beyond the cap' could not be measured "
                 f"at all. Lower skew.max_angle_deg.")
    check_contract(kept, source="skewed_training_data")
    check_contract(holdout, source="skewed_holdout")

    write_parquet(kept, resolve_output(args.out))
    write_parquet(holdout, resolve_output(args.holdout_out))


if __name__ == "__main__":
    main()
