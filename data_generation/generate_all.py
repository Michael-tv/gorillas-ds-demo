"""Runs every dataset listed in dvc_datasets.yaml -- the same manifest
dvc.yaml's `generate` foreach stage reads, so there is one source of truth
for which datasets exist and their parameters whether you regenerate via
`dvc repro generate` or by running this script directly.

Every generated file carries both the regression target
(landing_distance_m) and the classification target
(target_distance_m/hit_target) -- one dataset, both tasks.

regression/run_leakage and run_bias_variance don't need their own entries --
they draw a random 10k/20k sample straight out of standard_training_data via
their own train_utils.py (`df.sample(n=N_SAMPLES, random_state=seed)`).
"""
import time

import yaml

from data_generation import generate, io

with open(io.REPO_ROOT / "dvc_datasets.yaml") as f:
    DATASETS = yaml.safe_load(f)["datasets"]


def main():
    results = []
    for name, spec in DATASETS.items():
        start = time.time()
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        try:
            df = generate.generate(
                spec["n"], spec["seed"],
                elevation_dist=(spec["elevation_mean"], spec["elevation_std"]),
                hit_tolerance=spec["hit_tolerance"],
            )
            io.write_parquet(df, io.DATA_ROOT / f"{name}.parquet")
            status = "OK"
        except Exception as exc:
            print(f"  [error] {exc}")
            status = "FAILED"
        results.append((name, status, time.time() - start))

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    print(f"  {'Dataset':<32} {'Status':<8} {'Time':>6}")
    print(f"  {'-'*48}")
    for name, status, elapsed in results:
        print(f"  {name:<32} {status:<8} {elapsed:>5.1f}s")


if __name__ == "__main__":
    main()
