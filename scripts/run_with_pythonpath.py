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

Every stage in this repo uses the same shape:
    ... ../../../.. -m experiments.<domain>.<mode>.experiment_<name>.train_<model>

<pythonpath_dir> is always the repo root (`../../../..` from an experiments/<domain>/<mode>/
<experiment>/dvc.yaml stage), since `-m` resolves its dotted module path against it
(`experiments.regression.standard.experiment_raw.train_linear_regression` ->
experiments/regression/standard/experiment_raw/train_linear_regression.py).
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
