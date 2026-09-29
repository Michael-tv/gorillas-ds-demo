"""Loads params.yaml for training scripts. See dvc.yaml `params:` sections --
changing a value here (or via `dvc exp run --set-param`) marks the matching
stage(s) as stale for `dvc repro`/`dvc exp run`, and `dvc exp show`/`dvc exp
diff` will pick it up as a compared column.
"""
import os

import yaml

_REPO_ROOT   = os.path.dirname(os.path.abspath(__file__))
_PARAMS_PATH = os.path.join(_REPO_ROOT, "params.yaml")


def load_params():
    with open(_PARAMS_PATH) as f:
        return yaml.safe_load(f)


def load_experiment_params(script_file):
    """Load the params.yaml colocated with a training script's own directory
    -- experiment-specific config (row-count sweep tiers, the RandomizedSearchCV/
    *CV search budget `n_iter`, and the fold count `cv`) that varies by
    pool/experiment rather than being shared globally: single-split on the
    standard pool, grouped k-fold on the Gorillas pools. A caller does e.g.
    `load_experiment_params(__file__)["n_iter"][KEY]`."""
    path = os.path.join(os.path.dirname(os.path.abspath(script_file)), "params.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


class PoolTooSmall(RuntimeError):
    pass


def take_samples(df, n_samples, pool_name=None):
    """Return the first `n_samples` rows of the pool, or raise if there aren't
    that many.

    The sample-size tiers work by prefix-slicing one pre-shuffled pool, so each
    tier is a nested subset of the same draw. `df.iloc[:n]` silently returns
    fewer rows when the pool is smaller than `n` -- and the Gorillas pools hold
    only 5,000 rows, so a mismatched `n_samples` could silently shrink a tier
    instead of erroring. For a knob whose entire purpose is "same data,
    different sample size", not honouring the value has to be an error rather
    than a silent substitution.

    Pass `n_samples=None` to use the whole pool deliberately.
    """
    if n_samples is None:
        return df
    if n_samples > len(df):
        where = f" ({pool_name})" if pool_name else ""
        raise PoolTooSmall(
            f"n_samples={n_samples:,} exceeds the {len(df):,} rows available in "
            f"the configured pool{where}. Lower n_samples in params.yaml (or "
            f"--set-param n_samples=...), or point training_data at a pool with "
            f"at least {n_samples:,} rows -- silently training on {len(df):,} "
            f"rows would make this run incomparable with the other size tiers."
        )
    return df.iloc[:n_samples]


def data_path():
    """Absolute path to the configured training pool (params.yaml
    `training_data`, a name matching a dvc_datasets.yaml or Gorillas dataset
    key), so the active dataset is a single setting rather than something
    edited file by file."""
    return os.path.join(_REPO_ROOT, "data", f"{load_params()['training_data']}.parquet")
