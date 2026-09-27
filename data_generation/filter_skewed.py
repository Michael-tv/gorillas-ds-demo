"""Splits the standard pool into a low-angle slice and its complement.

The "skewed" demo trains on launch angles up to `skew.max_angle_deg` only, to
show what happens when a model is asked to predict outside the range it was
trained on. This used to be a filter inside
`models/regression/run_skewed/train_utils.py`, applied at load time, which had
three problems (AUDIT.md task 8, §"Skew by filter stage, not a second dataset"):

  * The filtered pool was invisible -- not an artifact you could open, plot, or
    show an audience, and not a node in the DAG, so the lineage story
    "generate -> filter -> train" was not there to read.
  * The bound was a Python constant, so it could not be swept with
    `dvc exp run --set-param skew.max_angle_deg=...` or compared in
    `dvc exp show`.
  * **The filter ran before the train/test split, so the test set carried the
    same hole.** The demo asserted the model "extrapolates poorly beyond 30
    degrees" while no stage ever evaluated it beyond 30 degrees.

That last one is why this stage emits TWO outputs. The kept rows are what the
skewed models train on; the excluded rows are the out-of-distribution test set
that makes the claim measurable rather than asserted.

A filter rather than a second generated dataset, deliberately: same draw, same
noise realisation, one variable changed. Generating a second pool with a
different elevation distribution would change the elevation distribution AND
every other sampled value AND the row count, so a worse skewed score could not
be attributed to the distribution.

The cap is one-sided on purpose. Tree ensembles cannot extrapolate -- beyond the
training range a random forest flatlines at its boundary leaf mean while a
linear model keeps going, and that contrast is the lesson. A gap in the middle
of the range is a much weaker demo: a forest interpolates across it adequately
and linear models are indifferent to it.
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
                   help="default: params.yaml skew.max_angle_deg, so it is sweepable "
                        "with --set-param and shows up in dvc exp show")
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

    # Both sides must still satisfy the contract -- a filter that produced an
    # empty or single-class frame would otherwise only surface much later, as a
    # confusing failure inside a training stage.
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
