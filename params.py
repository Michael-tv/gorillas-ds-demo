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


def search_params(domain, model):
    """Return the {n_iter, cv} RandomizedSearchCV (or *CV/GridSearchCV) budget
    for one model. Either key may be None if that algorithm doesn't use it."""
    return load_params()["search"][domain][model]


class PoolTooSmall(RuntimeError):
    pass


def take_samples(df, n_samples, pool_name=None):
    """Return the first `n_samples` rows of the pool, or raise if there aren't
    that many.

    The sample-size tiers work by prefix-slicing one pre-shuffled pool, so each
    tier is a nested subset of the same draw. `df.iloc[:n]` silently returns
    fewer rows when the pool is smaller than `n` -- and the Gorillas pools hold
    5,000 rows against `n_samples: 40000`, so switching `training_data` to one
    of them used to change the sample size by 8x with nothing in the output
    saying so, flatlining the convergence experiment (AUDIT.md task 35 / §5.5).
    For a knob whose entire purpose is "same data, different sample size", not
    honouring the value has to be an error rather than a silent substitution.

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
    key). Every consumer that used to hardcode
    data/standard_velocity.parquet should call this instead, so the
    active dataset is a single setting rather than something edited file by
    file -- see the comment on `training_data` in params.yaml."""
    return os.path.join(_REPO_ROOT, "data", f"{load_params()['training_data']}.parquet")
