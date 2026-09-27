# Session: data/training pipeline refactor (2026-09-13 -> 2026-09-15)

## What this repo is

Teaching project for an "Introduction to ML" talk. `physics.py` simulates
projectile motion; synthetic datasets generated from it feed regression demos
(predict initial velocity) and classification demos (predict hit/miss against
a target) plus teaching-specific pathology demos: `run_skewed` (distribution
shift), `run_leakage` (train/test contamination), `run_bias_variance`
(under/overfitting via `max_depth`).

## Why this refactor happened

Repo originally had ~15 `run_*/` folders under `regression/` and
`classification/`, each with its own copy-pasted `generate_data.py`,
`train_utils.py`, `train_all.py`, `predict.py`, `evaluate_features.py`,
writing to a local `training_data/training_data.csv`. User asked to
centralize data generation, pick a better storage format, add a real
feature-engineering step, and add DAG-based orchestration that tracks what's
run vs. outstanding.

## Decisions made, in order (each one supersedes the last where they conflict)

1. **Storage format: Parquet**, not CSV/SQLite/DuckDB. One file per dataset,
   columnar, typed, no server/connection overhead for what's just flat
   feature tables.
2. **Orchestration: DVC pipelines** (`dvc.yaml`), not Prefect/Snakemake/Luigi.
   Reasoning: every script here already just reads/writes files -- DVC wraps
   that as-is (declarative `cmd`/`deps`/`outs`, hash-based staleness) with no
   new Python task-wrapping layer, no server process, and it's Windows-native
   (Snakemake is Linux/macOS-first). `dvc repro` = "rerun what's stale,
   skip what isn't" -- exactly what was asked for.
3. **Generated data holds ONLY raw physical columns, never engineered
   features.** `wind_x_ms`, `drag_param`, `height_diff_m` are derived at
   *load time* by `feature_engineering.add_engineered_columns()` (a plain
   function) or `feature_engineering.EngineeredFeatures` (an
   sklearn-Pipeline-compatible transformer class), not stored on disk. This
   was a deliberate correction mid-session -- the first version of the plan
   had a separate `run_eng_*` *generated dataset*; the user said no, compute
   it at load time instead.
4. **One dataset serves both regression and classification.** Classification
   doesn't get its own generator/dataset -- every generated row carries both
   `landing_distance_m` (regression target's key feature) and
   `target_distance_m`/`hit_target` (classification target), derived from the
   *same single sample*. The user was explicit: "no special sampling is
   required for the classification case" -- this removed a
   resample-until-target-is-positive retry loop that classification's
   generator used to have. Consequence: `target_distance_m` (like
   `landing_distance_m`) can legitimately be negative now, and **must stay
   that way** -- the user explicitly rejected clipping/guarding it to
   non-negative when a downstream script (`train_linear_regression_sqrt.py`)
   choked on `sqrt()` of a negative value. Fixed with a *signed* sqrt
   (`sign(x) * sqrt(abs(x))`) instead of clipping.
5. **Package split by concern, not by domain**, requested explicitly by the
   user in three follow-up messages:
   - `data_generation/` -- one script (`generate.py`) with CLI flags, no
     per-domain generator files.
   - `models/regression/`, `models/classification/` -- ALL training code:
     shared `train_*.py` scripts, `common/` loaders, per-run `train_utils.py`
     shims, and the bespoke training scripts (`train_skewed.py`,
     `train_leaky.py`, `train_overfitting.py`, etc).
   - `evaluation/regression/`, `evaluation/classification/` -- ALL analysis
     code: `feature_evaluation.py`, `compare_models.py`, `simulate_predict.py`,
     per-run `predict.py`/`evaluate_features.py`/`plot_*.py`.
   - `regression/`, `classification/` -- became **pure output storage**:
     just `run_*/models/*.joblib` + `metrics_*.csv`. No code lives there at
     all except two explicitly-out-of-scope pre-existing files
     (`regression/test.py`, `regression/run_raw_10k/trajectory.csv`).

## Bugs found and fixed during this refactor (not pre-existing intent, actual fixes)

- Corrupting `mass_kg` to exactly `0.0` (one of five random "data_error"
  corruption types) produced `inf` in `drag_param` once engineering ran on
  raw data -- the old eng-only generator never hit this because it corrupted
  the *already-computed* `drag_param` scalar, never a value a later formula
  divides by. Fixed: `corrupt_row(..., no_zero_indices={...})` floors
  protected columns to a small epsilon instead of exactly zero.
- `run_skewed`'s bespoke `train_utils.py` had `print_metrics(y_test, y_pred,
  model_name=None)` that only saved metrics when `model_name` was passed
  explicitly -- fine for its own `train_skewed.py`/`train_balanced.py`, but
  silently dropped metrics when DVC ran the *generic* model scripts
  (`train_random_forest.py` etc.) against it, since those infer the name
  from `TRAIN_MODEL_NAME`/`sys.argv` and never pass it explicitly. Fixed to
  auto-infer like the shared loaders do, falling back to the explicit
  override when given.
- `regression/run_skewed/train_all.py`'s `MODELS_DIR` pointed at the shared
  `regression/models/` instead of `regression/run_skewed/models/` (a
  pre-existing bug) -- moot now since that dispatcher file is deleted
  (superseded by `dvc.yaml`) and its bespoke stages get their own correct
  `outs:` paths.
- `run_leakage`/`run_bias_variance` pointed at `run_eng_30k`/`run_eng_500k`,
  which never existed anywhere in the repo (long-standing dangling
  reference, confirmed via exploration before touching anything). Per the
  user: fixed by making `raw_30k`/`raw_500k` valid on-demand generation
  targets, not by pre-generating 500k rows during this session.

## Verification performed (not just written-and-assumed)

Every one of the 17 model-training scripts, every bespoke training script
(skewed/balanced/leaky/clean/overfitting/underfitting), `predict.py` for
every family, and multiple real `dvc repro` runs (including a back-to-back
same-target rerun that confirmed cache-skip behavior) were actually executed
against real generated data, not just written.

## Feature-engineering-in-sklearn-Pipeline note

The 8 model scripts that used to manually `StandardScaler().fit_transform()`
outside the model and save `{"model":..., "scaler":...}` dict now build
`Pipeline([("scaler", StandardScaler()), ("model", estimator)])` and every
script -- scaled or not -- saves the bare model/pipeline object directly.
`predict.py` everywhere dropped the `scaler = data.get("scaler")` dance
accordingly.

## Repo-specific gotchas worth remembering

- This machine's shell profile auto-activates a **different** project's
  virtualenv (`V-and-V-...`) on every new shell -- `pipenv run`/`pipenv
  install` silently target the wrong environment unless you pass
  `PIPENV_IGNORE_VIRTUALENVS=1`, or just invoke this project's venv
  interpreter directly by absolute path. For anything that spawns
  subprocesses itself (DVC's `cmd:` steps), prepend this project's venv
  `Scripts/` dir to `PATH` for the whole invocation rather than only fixing
  the outer process -- `dvc repro` spawns `python` via PATH lookup
  internally, so fixing only the outer interpreter isn't enough.
- Do not edit the user's global shell profile to fix the above -- their other
  concurrent session may depend on that auto-activation.
