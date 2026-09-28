"""Cross-platform helper for DVC training stages.

Runs `python <args...>` with PYTHONPATH set to a given directory, so a
`-m package.module` invocation resolves regardless of where `dvc repro` was
invoked from or which directory the calling dvc.yaml lives in -- DVC always
runs `cmd:` with the working directory set to wherever that dvc.yaml file
is, not the repo root. DVC stages also run `cmd:` through the OS shell,
where `set X=Y && cmd` (Windows) and `X=Y cmd` (POSIX) aren't portable --
this keeps every stage's `cmd:` identical on any OS.

Usage:
    python scripts/run_with_pythonpath.py <pythonpath_dir> <python arg>...

Every stage in this repo now uses the same shape:
    ... ../.. -m models.<domain>.<mode>.experiment_<name>.train_<model>

<pythonpath_dir> is always the repo root (`../..` from a pipelines/<name>/
dvc.yaml stage), since `-m` resolves its dotted module path against it
(`models.regression.standard.experiment_raw.train_linear_regression` ->
models/regression/standard/experiment_raw/train_linear_regression.py).
Every `experiment_<name>/train_<model>.py` script is self-contained -- no
folder-local `train_utils.py` to resolve via a different PYTHONPATH, so
there is no second shape to document here anymore (there used to be one,
back when concept/demo scripts imported a sibling `train_utils.py`).
"""
import os
import subprocess
import sys


def main():
    if len(sys.argv) < 3:
        print("usage: run_with_pythonpath.py <pythonpath_dir> <python arg>...")
        sys.exit(2)

    pythonpath_dir, *py_args = sys.argv[1:]
    env = {**os.environ, "PYTHONPATH": os.path.abspath(pythonpath_dir)}
    result = subprocess.run([sys.executable, *py_args], env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
