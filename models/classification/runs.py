"""Central registry of classification run configs.

Replaces the three run/run_effort/run_velocity train_utils.py shims -- see
models/regression/runs.py's docstring for the full rationale (AUDIT.md C1).
Classification has no raw-vs-engineered split (every run uses the same
engineered feature set), so there are three entries instead of regression's
seven.
"""
import os
from dataclasses import dataclass
from typing import Optional

import params
from models.classification.common import loader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass(frozen=True)
class RunConfig:
    name: str
    data: str
    models_dir: str
    n_samples: Optional[int]


def _exp_dir(run_dir_name):
    return os.path.join(REPO_ROOT, "experiments", "classification", run_dir_name, "models")


def _data(name):
    return os.path.join(REPO_ROOT, "data", f"{name}.parquet")


def _build_runs():
    n_samples = params.load_params()["n_samples"]
    return {
        "run":      RunConfig("run",      params.data_path(),                    _exp_dir("run"),      n_samples),
        "effort":   RunConfig("effort",   _data("gorillas_effort_classification"),   _exp_dir("run_effort"),   None),
        "velocity": RunConfig("velocity", _data("gorillas_velocity_classification"), _exp_dir("run_velocity"), None),
    }


RUNS = _build_runs()
