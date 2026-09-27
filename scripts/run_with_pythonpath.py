"""Cross-platform helper for DVC training stages.

Runs a training script with PYTHONPATH set to its run folder (so the script's
`from train_utils import ...` resolves to that run's data/models), and
optionally sets extra environment variables (TRAIN_MODEL_NAME, TRAIN_CLEAN).

DVC stages run `cmd:` through the OS shell, where `set X=Y && cmd` (Windows)
and `X=Y cmd` (POSIX) aren't portable -- this keeps every stage's `cmd:`
identical on any OS.

Usage:
    python scripts/run_with_pythonpath.py <run_dir> <script> [KEY=VALUE ...]
"""
import os
import subprocess
import sys


def main():
    if len(sys.argv) < 3:
        print("usage: run_with_pythonpath.py <run_dir> <script> [KEY=VALUE ...]")
        sys.exit(2)

    run_dir, script, *extra_env = sys.argv[1:]
    env = {**os.environ, "PYTHONPATH": os.path.abspath(run_dir)}
    for pair in extra_env:
        key, _, value = pair.partition("=")
        env[key] = value

    result = subprocess.run([sys.executable, script], env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
