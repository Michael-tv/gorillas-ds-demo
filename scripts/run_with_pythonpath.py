"""Cross-platform helper for DVC training stages.

Runs `python <args...>` with PYTHONPATH set to a given directory, so a
`-m package.module` invocation (or a script that does absolute imports like
`from models.regression.runs import RUNS`) resolves regardless of where
`dvc repro` was invoked from or which directory the calling dvc.yaml lives
in -- DVC always runs `cmd:` with the working directory set to wherever that
dvc.yaml file is, not the repo root. DVC stages also run `cmd:` through the
OS shell, where `set X=Y && cmd` (Windows) and `X=Y cmd` (POSIX) aren't
portable -- this keeps every stage's `cmd:` identical on any OS.

Usage:
    python scripts/run_with_pythonpath.py <pythonpath_dir> <python arg>...

Two shapes in use across this repo's dvc.yaml files:
    ... <pythonpath_dir> -m models.regression.train --run raw --key ...   (most stages)
    ... <pythonpath_dir> path/to/a/standalone_script.py                   (the concept/demo scripts)

`-m` needs <pythonpath_dir> to be the repo root, since it resolves a DOTTED
module path against it (`models.regression.train` -> models/regression/
train.py); a standalone script needs it to be the script's own run folder, so
its local `from train_utils import ...` resolves -- see the calling dvc.yaml
for which one a given stage passes.
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
