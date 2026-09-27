"""Shared output-path helpers for dataset generators."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"


def resolve_output(path_str):
    """Resolve a --out CLI argument: absolute paths pass through, relative ones
    are resolved against the repo root (so "data/raw_30k.parquet" works)."""
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def write_parquet(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved {len(df):,} rows -> {path}")
    return path
