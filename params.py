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


def data_path():
    """Absolute path to the configured training pool (params.yaml
    `training_data`, a name matching a dvc_datasets.yaml or Gorillas dataset
    key). Every consumer that used to hardcode
    data/standard_training_data.parquet should call this instead, so the
    active dataset is a single setting rather than something edited file by
    file -- see the comment on `training_data` in params.yaml."""
    return os.path.join(_REPO_ROOT, "data", f"{load_params()['training_data']}.parquet")
