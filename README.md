# Gorillas ML demo

Teaching material for an "Introduction to ML" talk. A projectile-motion dataset is
generated two different ways, a set of regression and classification models are trained on
it through a DVC pipeline, and the whole thing is wired so that each classic failure mode —
leakage, overfitting, distribution shift, class imbalance, dirty data — can be demonstrated
by changing one setting and re-running.

The two data producers are the point of the repo:

| | `data_generation/generate.py` | `data_generation/gorillas.py` |
|---|---|---|
| Source | RK4 numerical integration (`physics.py`) | the real 1990 QBasic *Gorillas* game, driven through DOSBox |
| Nature | clean, independent, fully observed | occluded by buildings, per-board wind, ~7% hit rate |
| Speed | ~875 rows/s, 50,000 rows in ~1 min | ~15–200 rows/s depending on `--workers` |
| Platform | anywhere | Windows (needs DOSBox Staging) |

Both emit the **same 14-column contract**, so either can fill the training slot — and both
now support the same two **input modes**: `VELOCITY` (launch speed drawn/entered directly)
and `EFFORT` (a capped force model derives the speed instead — see "The 14-column contract"
below). The simulator gives a well-behaved dataset; the game gives a realistically awkward
one, where the thing you want to predict genuinely is not a function of the columns you
logged.

---

## Quick start

```bash
pipenv install                      # Python 3.13; see Pipfile
pipenv shell

dvc repro generate                          # build both simulated pools (~2 min)
dvc repro models/regression/standard/experiment_raw/dvc.yaml:train_raw_random_forest   # train one model
dvc repro -P                                 # the whole repo (dvc stage list --all for the current count)
```

The pipeline is split into 20 `dvc.yaml` files — one root file with the shared data producers
(physics + game generation; standard and Gorillas pools are generated and trained on
separately, never merged), and one file per {**domain**, **mode**, **experiment**} that lives
**next to the training scripts it runs**, in `models/<domain>/<mode>/<experiment>/dvc.yaml`
(`regression`|`classification` × `standard`|`effort`|`velocity` × whichever experiments apply).
`dvc repro` from the repo root only touches the root file's stages; `-P` / `--all-pipelines`
is what runs everything, and `-R <dir>` runs every stage found anywhere under one directory
(e.g. `dvc repro -R models/regression/standard` for every regression experiment on the standard
pool). See "Running one pipeline at a time" below.

If `pipenv` picks up the wrong virtualenv, prefix with `PIPENV_IGNORE_VIRTUALENVS=1`.

There is **no DVC remote, by design.** Both producers are deterministic for a fixed seed, so
regeneration — not a stored copy — is the recovery path. A fresh clone therefore starts with
`dvc repro generate`, not `dvc pull`.

---

## What's here

```
params.yaml                    ← the knobs you actually turn
dvc_datasets.yaml               which datasets exist, and their generation parameters
dvc_models_regression.yaml     ← which regression models exist (algorithm + cleaning)
dvc_models_classification.yaml ← which classification models exist
dvc.yaml                       ← ROOT pipeline: the two shared data producers

(no pipelines/ directory -- every experiment's dvc.yaml sits in its own models/ folder, below)

physics.py             RK4 projectile integrator with drag and wind
data_generation/        the two producers, plus the shared contract
  generate.py            physics/RK4 producer -- VELOCITY (direct draw) or EFFORT (capped
                          force model, see sampling.py's EFFORT_RANGE/MAX_FORCE_N) mode
  gorillas.py             the real game, driven through DOSBox -- same two input modes
feature_engineering.py the 3 derived features, as a function and a Pipeline step
splitting.py            group-aware train/test splits and cross-validation
params.py               reads params.yaml; also holds take_samples()

models/regression/
  algorithms/              one small module per algorithm: the estimator, its search space
                           (shared across every experiment below -- the one thing that's DRY)
  common/loader.py         shared load/clean/metrics helpers
  standard/                each folder here is one experiment: a standalone
    experiment_raw/          train_<model>.py per model, no --run/--algorithm dispatch, plus the
                             experiment's own dvc.yaml (the pipeline that runs those scripts)
    experiment_eng/
    experiment_skew/         + train_skewed_concept.py / train_balanced_concept.py
    experiment_leakage/       train_clean.py / train_leaky.py
    experiment_bias_variance/ train_underfitting.py / train_overfitting.py
    experiment_row_count/    sweeps sample-size tiers per model internally (tier list lives in
                             this folder's own params.yaml, not the shared root one)
    experiment_cv_baseline/  experiment_raw, but keeps 5-fold CV in the hyperparameter search
                             instead of the single validation split the other standard-pool
                             experiments use (see "Cross-validation" below)
  effort/                  experiment_raw/, experiment_eng/, experiment_row_count/
                           (no skew/leakage/bias-variance here -- standard-pool-only demos);
                           permanently on data/gorillas_effort.parquet (real game data only --
                           not merged with the standard pool; see "Two producers" below)
  velocity/                same shape as effort/, permanently on data/gorillas_velocity.parquet
models/classification/  the same shape, minus the raw/eng split (one scheme: engineered
                         features only) -- standard|effort|velocity/experiment_classification/,
                         plus experiment_row_count/ in all three; each with its own dvc.yaml
evaluation/              prediction, feature and plotting scripts (run by hand)
experiments_results/    trained models (DVC-cached) and metrics (Git-versioned)
qbasic_gorillas/         the DOSBox game builds, including the datagen fork
```

**No more `train.py`/`runs.py` dispatch.** Every model, in every experiment, is its own
standalone `train_<model>.py` — reading its own pool, building its own split, calling the
shared `algorithms/<model>.py` for the estimator, and saving to its own
`experiments_results/.../<experiment>/models/` directory. There's nothing to trace through a
`--run`/`--algorithm` CLI flag anymore: open the one file for the model and experiment you
care about and read it top to bottom. `algorithms/<model>.py` stays shared on purpose — it's
already one clean file per model with nothing to duplicate.

**Switching a model off**, now that there's no matrix for most experiments: comment out (or
delete) that model's stage block directly in the relevant
`models/<domain>/<mode>/<experiment>/dvc.yaml` — each is independently named
(`train_raw_random_forest`, `train_skewed_ridge`, ...), so removing one touches nothing else,
not even other stages in the same file. The exception: `dvc_models_regression.yaml`/
`dvc_models_classification.yaml` still exist and still matter — they're the source of truth
for the search-budget key each `train_<model>.py` reads from `params.yaml`, and for which
model names are valid. See "The model matrix" below.

### The 14-column contract

Every generated pool has exactly these columns, asserted by
`data_generation/contract.py` both when written and when read:

| Column | Notes |
|---|---|
| `initial_velocity_ms` | the **regression target** |
| `launch_angle_deg`, `wind_speed_ms`, `wind_direction_norm` | raw inputs |
| `mass_kg`, `radius_m`, `drag_coeff` | raw inputs |
| `launch_height_m`, `landing_height_m` | raw inputs |
| `landing_distance_m` | where it landed. **Can be negative** — a high-angle shot into a strong headwind lands behind the launch point |
| `target_distance_m` | where it was aimed |
| `hit_target` | the **classification target** |
| `is_outlier` | `none` / `gravity` / `data_error`. A generation-time oracle — real data has no such column |
| `group_id` | rows sharing a value were drawn under correlated conditions and must not be split apart |

The three engineered features — `wind_x_ms`, `drag_param`, `height_diff_m` — are **never
stored**. They are derived at training time, inside the model's sklearn `Pipeline` (or, for
classification and the standalone concept demos, at load time via
`feature_engineering.add_engineered_columns`), so the saved `.joblib` carries its own
preprocessing and can be fed raw physical inputs.

**`VELOCITY` vs `EFFORT` input mode** (both producers): `VELOCITY` draws/enters launch speed
directly. `EFFORT` derives it from a capped force model instead — a work-energy calculation
over a fixed stroke length, force capped as a percentage of a calibrated maximum (mirrors the
real game's own `gorilla.bas` force model). Same 14 columns either way; only how
`initial_velocity_ms` came to be differs.

`group_id` is what makes honest evaluation possible. The game throws 32 bananas at each
board, and a board's wind and skyline are shared by all of them, so a random split puts the
same board on both sides and scores the model against rows it has effectively already seen.
Measured on a 60-board pool: a random split reports MAE 12.944 where the group-aware split
reports 16.157 — **20% optimistic**. `splitting.py` handles this, and because the simulated
pool gives every row its own `group_id`, the same code degrades to an ordinary random split
there.

---

## Running it end to end

### 1. Generate data

```bash
dvc repro generate                                  # both standard_effort and standard_velocity
dvc repro generate@standard_velocity                # just one
```

For the game producer (Windows, DOSBox Staging installed):

```bash
setx GORILLAS_DOSBOX "C:\Program Files\DOSBox Staging\dosbox.exe"   # once, optional
dvc repro generate_gorillas
```

DOSBox is located via `--dosbox` → `$GORILLAS_DOSBOX` → `dosbox` on `PATH` → known install
locations. `generate_gorillas` expands into two independently-cached variants —
`gorillas_effort`, `gorillas_velocity` — one DOSBox run each. Regression and classification
both train from whichever one file matches their mode; the game has never distinguished
"regression data" from "classification data" (one throw carries both `initial_velocity_ms`
and `hit_target`). Each variant writes both a parquet and a raw throw log
(`data/<name>_throws.parquet`), which keeps the `board` and `outcome` columns the 14-column
contract deliberately omits.

The standard (physics) pool and the Gorillas pool are **not merged** — the `effort`/
`velocity` experiments train on `data/gorillas_{effort,velocity}.parquet` directly, and
`data/standard_{effort,velocity}.parquet` are reached separately, through
the `standard` experiments' switchable pool (see "The experiments" below). Two different sources,
two different experiments, deliberately kept apart — an earlier version of this repo joined
them into one combined pool per mode; reversed, since the point of having two producers is to
compare them, not blend them.

`--workers` only controls how many DOSBox instances run at once and **cannot** change the
resulting data — every chunk seed is drawn single-threaded before the thread pool starts.
`--boards-per-chunk` and `--throws` **do** change the data, which is why they live in
`dvc_datasets.yaml` and `--workers` does not.

You can also run the game interactively from `qbasic_gorillas/dosbox-datagen/`:
`Play Effort.bat` or `Play Velocity.bat`.

### 2. Train

Training stages live in `models/<domain>/<mode>/<experiment>/dvc.yaml`, next to the scripts
they run — not the root file, which holds only data generation. Four ways to run them,
cheapest first:

```bash
dvc repro -P                                                                                # every pipeline in the repo, in order
dvc repro -R models/regression/standard                                                     # every regression experiment on the standard pool
dvc repro models/regression/standard/experiment_raw/dvc.yaml:train_raw_random_forest        # one stage, from the repo root
cd models/regression/standard/experiment_raw && dvc repro && cd ../../../..                 # one whole experiment
```

`-P` (`--all-pipelines`) discovers and reproduces all 20 `dvc.yaml` files, resolving
cross-pipeline dependencies by file path — `models/regression/effort/experiment_raw/dvc.yaml`'s
stages, which read `../../../../data/gorillas_effort.parquet`, are recognized as depending on the
root `generate_gorillas@gorillas_effort` stage automatically, the same way two stages in one
file depend on each other. `-R <dir>` (`--recursive`) does the same discovery scoped to one
directory, which is what makes "run everything for this mode" or "run everything for this
experiment" both single commands despite there being one file per experiment rather than one
per mode. See "Running one pipeline at a time" below for the full pattern and the things that
are NOT obvious about it (relative paths, and where `params:` looks for `params.yaml`).

Root pipeline plus 3 modes, 7/4/4 experiments each (`row_count` has one `dvc.yaml` per domain):

| Pipeline | What it shows |
|---|---|
| `dvc.yaml` (root) | the two producers: `generate` (standard_effort/standard_velocity) and `generate_gorillas` (gorillas_effort/gorillas_velocity) — not merged with each other |
| `models/*/standard/*` (7 experiments) | the raw/eng/classification sweeps, sample-size row-count experiment, and every skew/leakage/bias-variance demo — all on `data/standard_velocity.parquet` (switchable to any generated pool via `params.yaml: training_data`, including `standard_effort` or a Gorillas pool) |
| `models/*/effort/*` (4 experiments) | the same raw/eng/classification sweeps + row-count experiment, permanently on `data/gorillas_effort.parquet` (real game data only) |
| `models/*/velocity/*` (4 experiments) | same, permanently on `data/gorillas_velocity.parquet` |

Models land in `experiments_results/…/models/model_<key>.joblib` (DVC-cached) and metrics in
`metrics_<key>.csv` (Git-versioned, so `dvc metrics diff` works across commits).

### 3. Compare

```bash
dvc exp show                    # params and metrics, side by side
dvc metrics diff                # against HEAD
dvc dag                          # every stage in the repo as one graph
```

Unlike `repro`, `dag` finds stages across **all** `dvc.yaml` files by default — no `-P` needed;
pass a stage name (optionally `path/to/dvc.yaml:stage`) to see just its subgraph.

### 4. Evaluate and plot

The `evaluation/` scripts are run by hand — they are **not** in the DAG. Every script that
used to import a `RUNS` registry or a folder-local `train_utils.py` now has its handful of
constants (`DATA`, `FEATURES`, `N_SAMPLES`, ...) inlined directly — no PYTHONPATH wrapper
needed, every evaluation script is self-contained:

```bash
python evaluation/regression/run_raw/predict.py
python evaluation/regression/run_raw/plot_training_data_relationships.py
python evaluation/regression/compare_models.py
python evaluation/regression/run_raw/evaluate_features.py
```

`predict.py` takes its inputs from a config block at the bottom of the file, including
`model_name` — which must be a key trained in that experiment (`ridge` and `lasso` exist only
under `experiment_skew`). `compare_models.py`'s `MODELS_TO_COMPARE` list now names
`(domain, experiment, model)` triples (regression) or `(domain, model)` pairs
(classification), since `experiments_results/regression/<domain>/<experiment>/` has one more
path segment than the old `experiments/regression/<run_dir>/` did.

### 5. Running one pipeline at a time

So "run everything for effort" (or velocity, or one experiment inside the standard-pool sweep)
is one command instead of naming stages individually. Ways to run a whole pipeline, from the
narrowest to the widest scope:

```bash
cd models/regression/effort/experiment_raw && dvc repro && cd ../../../..   # one experiment, from inside its directory
dvc repro models/regression/effort/experiment_raw/dvc.yaml                    # same, from the repo root, by path
dvc repro -R models/regression/effort                                         # every regression experiment under one mode
dvc repro -P                                                        # every pipeline in the repo, including this one
```

Things about this pattern that are not obvious from the stage files themselves, all found by
testing rather than assumed:

- **Every path inside `models/<domain>/<mode>/<experiment>/dvc.yaml` is `../../../../`-prefixed.**
  DVC resolves a stage's `deps`/`outs`/`cmd` relative to the `dvc.yaml` that declares it, not to
  the repo root or wherever `dvc repro` was invoked from — so `models/regression/standard/
  experiment_raw/train_knn.py` at the repo root becomes `../../../../models/regression/standard/
  experiment_raw/train_knn.py` inside a four-levels-deep pipeline file (the pipeline sits in the
  same folder as the `train_knn.py` it runs, so the script itself is also reachable as a plain
  `train_knn.py`, though this repo spells every path from the root for uniformity).
  Cross-pipeline dependencies (a stage here reading a file the root `generate_gorillas` produces)
  need no special syntax; DVC matches them by the resolved absolute path, the same as any two
  stages in one file.
- **A stage's `params:` list needs the file spelled out.** The `vars:` block some pipeline
  files carry (`- ../../../../params.yaml`, only where `${training_data}` is actually
  interpolated — see "The experiments" below) only controls `${...}` template interpolation at
  parse time. A stage's own `params:` — the list DVC uses for staleness tracking and `dvc exp
  show` — is a separate mechanism that otherwise defaults to a `params.yaml` **next to that
  dvc.yaml**, which is the wrong file here (there's either none, or — in `experiment_row_count/` —
  only that experiment's own local tier list, not the shared root one). Every `params:` block
  in `models/*/*/*/dvc.yaml` therefore names the file explicitly:

  ```yaml
  params:
    - ../../../../params.yaml:
        - test_size
        - search.regression.random_forest.n_iter
  ```

  Leaving this implicit fails loudly and immediately (`dvc repro` errors with "Parameters
  'test_size, ...' are missing from 'params.yaml'"), rather than silently tracking nothing — so
  if you add a `params:` entry and see that error, this is why.

- **`python -m package.module` needs `scripts/run_with_pythonpath.py`, for a different
  reason than a script-path invocation would.** `-m` resolves its dotted module path against
  the *current working directory*, and DVC always runs `cmd:` with the working directory set to
  wherever the calling `dvc.yaml` lives — `models/regression/effort/experiment_raw/`, not the repo root
  — so a bare `python -m models.regression.effort.experiment_raw.train_random_forest` there
  fails immediately with `ModuleNotFoundError: No module named 'models'`. Every `cmd:` in
  `models/*/*/*/dvc.yaml` therefore goes through the wrapper, with `../../../..` (the repo root,
  four levels up) as its PYTHONPATH argument: `python ../../../../scripts/run_with_pythonpath.py
  ../../../.. -m models.regression.effort.experiment_raw.train_random_forest`. Note the two
  different relative spellings in the same line — the wrapper script itself is a `deps`/`cmd`-style
  path (`../../../../`), the PYTHONPATH argument after it is a plain relative directory reference
  (`../../../..`) — both point four levels up, they're just spelled by two different conventions.
  This is the **only** invocation shape in the repo — every experiment script, everywhere, uses it.

- **Not every script resolves its own `--out`-style arguments the same way.** Most `deps`/`outs`
  paths in `models/*/*/*/dvc.yaml` are `../../../../`-prefixed because DVC resolves them relative
  to the dvc.yaml itself (the first bullet above). `data_generation/filter_skewed.py`'s `--out`
  and `--holdout-out` arguments are different: `data_generation/io.py`'s `resolve_output()`
  anchors a relative path to its own hardcoded repo root (`Path(__file__).resolve().parent.
  parent`), not to CWD — so passing it a `../`-prefixed path double-resolves and writes
  outside the repo. `filter_skewed`'s `cmd:` in `models/regression/standard/experiment_skew/dvc.yaml`
  passes bare `data/...` paths for exactly this reason. Confirmed by testing (when these files
  were still three levels deep): the prefixed form ran without error and silently wrote outside
  the repo entirely.

`dvc dag` (no `-P` needed — see "3. Compare") and `dvc stage list` both work the same way run
from any pipeline's own directory, scoped to just that file, or from the repo root against
`models/<domain>/<mode>/<experiment>/dvc.yaml:<stage>`. `dvc stage list --all -R models/<domain>/<mode>`
scopes to one mode's experiments without listing every other mode too.

---

## The experiments — what to change to show what

**Which dataset?** Two different mechanisms, depending on which comparison you want:

- **Switchable pool** (the `standard` experiments only) — `training_data: standard_velocity` →
  any other generated pool name retargets `train_raw_<model>`/`train_eng_<model>`/
  `train_classification_<model>` at it without touching any other file. This is the headline
  comparison: the same models on clean simulated data versus data from a real game with an
  unobserved confounder.

  ```bash
  dvc exp run -R models/regression/standard --set-param training_data=gorillas_effort --set-param n_samples=5000
  ```

  `training_data`/`n_samples` are read across six separate files now
  (`experiment_raw`, `experiment_eng`, `experiment_classification`, `experiment_skew`,
  `experiment_leakage`, `experiment_bias_variance`) rather than one shared `dvc.yaml`, so `-R
  models/regression/standard` (discover every stage under that directory) replaces `cd`-ing
  into a single file's directory — `--set-param` still only needs saying once; it updates
  `params.yaml` and every stage that reads the changed keys goes stale together, regardless of
  which file it's in. `experiment_classification` lives under `models/classification/standard`,
  so run the same command with `-R models/classification/standard` (or `-R models/*/standard`
  in a shell that expands the glob) to cover it too.

  `n_samples` must be set too — a Gorillas pool holds 5,000 rows, and a sample size the pool
  cannot honour is an error rather than a silent truncation.

- **Permanent pipelines** (`models/*/effort`, `models/*/velocity`) — always exist on disk;
  no `--set-param` needed to compare them, just `dvc exp show` or read both
  `experiments_results/.../standard/experiment_raw/` and `.../effort/experiment_raw/` directly.

**Does more data help?** `experiment_row_count/` (all three modes, both `models/regression/`
and `models/classification/`) — one script per model sweeps sample-size tiers internally and
writes one metrics row per tier, so the whole learning curve is in one file, no `--set-param`
sweep needed. The tier list is **not** in the shared root `params.yaml` — each
`experiment_row_count/` folder has its own local `params.yaml` (just `tiers: [...]`), read via
`params.load_experiment_params(__file__)`, since the right tiers depend on the pool's actual
size: `standard`'s are `[1000, 2000, 5000, 10000, 20000, 50000]` (the 50,000-row physics pool);
`effort`'s/`velocity`'s are `[500, 1000, 2000, 3000, 4000, 5000]` (the Gorillas pool is only
5,000 rows — `pandas.iloc[:n]` silently returns fewer rows than asked rather than raising, so a
tier list sized for the bigger pool would silently duplicate its largest tiers instead of
erroring).

Each model's per-tier outputs live in their own `experiments_results/<domain>/<mode>/
experiment_row_count/tiers/<model>/model_n<size>.joblib`, declared in `dvc.yaml` as a single
**directory** `outs:` entry rather than one hand-listed file per tier — DVC then tracks
whatever's actually in that folder. Before training, each script deletes any tier file no
longer in the current `tiers:` list, so editing the tier list and re-running both adds the new
tier's output *and* removes a dropped tier's old one automatically; the combined
`metrics_<model>.csv` (one row per tier) still lands in the plain `models/` folder alongside
every other experiment's metrics.

(`params.yaml: n_samples` still exists too, for shrinking `experiment_raw`/`experiment_eng`/
`experiment_classification`'s single-size runs — a `dvc exp run --set-param n_samples=...`
sweep across those is a second, coarser way to see the same trend.)

**Cross-validation.** Hyperparameter search (`RandomizedSearchCV`/`GridSearchCV`/`RidgeCV`/
`LassoCV`) needs a `cv=` strategy either way, but the standard and Gorillas pools use different
ones, both built in `splitting.py`:

- **Standard pool** (`experiment_raw`, `experiment_eng`, `experiment_skew`,
  `experiment_row_count`, `experiment_classification`): `splitting.single_split_cv()` — one
  `ShuffleSplit(n_splits=1)` train/validation split, not k-fold. The pool is large and i.i.d.
  (50,000 rows, a smooth low-noise physics function), so a single split picks essentially the
  same hyperparameters 5-fold CV would, at roughly a 5x speed-up (each candidate is fit once
  instead of five times). `single_split_cv()` takes no fold-count argument, so these
  experiments read no `cv` knob at all — not from the root `params.yaml`, not from a local one.
  `linear_regression.py`'s old `cross_val_score` diagnostic — a CV-RMSE print that never
  influenced the fitted model — was removed for the same reason.
- **Gorillas pools** (`effort`, `velocity`): still `splitting.cv_for()`, group-aware 5-fold CV
  (`GroupKFold`/`StratifiedGroupKFold`). Here CV isn't just a variance estimate — a board's 32
  throws share wind/skyline, so group-aware folds are what stop rows from the same board landing
  on both sides of a split. **Do not** swap these to `single_split_cv()`.
- **`experiment_cv_baseline`** (`models/regression/standard/experiment_cv_baseline/`) is a
  verbatim copy of `experiment_raw` that deliberately keeps `cv_for()`'s original 5-fold search,
  as a runnable "before" comparison for the CV speed/robustness trade-off above — same data,
  same models, same param grids, only the validation strategy differs.

**Where the fold count lives.** The `cv:` fold count is **not** in the root `params.yaml`
(only `n_iter`, the search budget, still is). It lives in a local `params.yaml` next to the
`dvc.yaml` of every experiment that actually calls
`cv_for()` — `experiment_raw`/`experiment_eng`/`experiment_row_count` under `effort`/`velocity`,
`experiment_classification`/`experiment_row_count` under `classification/effort`/
`classification/velocity`, and `experiment_cv_baseline` — one `cv: 5` value shared by every
model in that folder (mirroring the single value every algorithm already had), read via
`params.load_experiment_params(__file__)["cv"]` the same way `experiment_row_count` already
reads its own `tiers`. Standard-pool experiments other than `experiment_cv_baseline` have no
local `cv` at all, since `single_split_cv()` would never read it. This mirrors why `tiers` lives
per-experiment rather than in the root file (see "Does more data help?" above): the right CV
strategy is a pool/experiment property, not a global one, and a single shared root-level `cv`
value was either ignored by 6 of the 9 non-demo experiments or forced an unrelated fold count on
every one of them.

**Raw or engineered features?** `experiment_raw/` vs `experiment_eng/` (regression only —
classification always uses the engineered feature set). Same data, same models; the only
difference is the first step of the Pipeline.

**Does cleaning help?** `clean` in `dvc_models_regression.yaml` — `""`, `"range"` or `"no_outlier"`.
`random_forest_no_outlier` exists to make this contrast visible in `dvc exp show`.

> **Caveat, and it matters:** `clean: "range"` bounds `landing_distance_m` at 0, which
> discards genuine headwind landings — 727 of the 964 rows it drops. As it stands, the range
> demo mostly removes *real* data. See task 13b in `AUDIT.md`.

**Leakage:** `models/regression/standard/experiment_leakage/train_clean.py` vs `train_leaky.py`.

**Overfitting:** `experiment_bias_variance/train_underfitting.py` vs `train_overfitting.py`, or
the `decision_tree_overfit` model in `experiment_raw`/`experiment_eng` — the same algorithm
with no depth limit and no split at all.

**Distribution shift:** `experiment_skew/` trains 9 models on a low-angle slice only
(`skew.max_angle_deg`), and `filter_skewed` keeps the excluded rows as
`data/skewed_holdout.parquet` — so the claim is measured on angles the model never saw, not on
a test set that shares its blind spot:

| | in-distribution MAE | holdout MAE | |
|---|---|---|---|
| skewed | 4.383 | 8.847 | **2.0× worse** |
| balanced | 7.530 | 6.234 | 0.8× — no penalty |

Note the inversion: the narrow model looks *better* in-distribution precisely because its test set
has the same hole. `train_balanced_concept.py` is the control — same architecture, same split, and
sized by reading the skewed run's row count. Beyond the training range a linear model keeps
extrapolating while a forest flatlines at its boundary leaf mean, which is why `ridge` and `lasso`
appear in `experiment_skew/` and nowhere else.

```bash
cd models/regression/standard/experiment_skew
dvc exp run --set-param skew.max_angle_deg=40    # move the cap; both runs follow
```

**Class imbalance:** run any classification stage against a Gorillas pool — either
`models/classification/standard` with `--set-param training_data=gorillas_effort`, or
`models/classification/effort`/`models/classification/velocity` directly. The hit rate is ~7% on a real game pool, so predicting "miss"
every time scores ~93%. The metrics report leads with precision/recall/PR-AUC and prints the
always-miss baseline beside accuracy:

```
  Precision : 0.3043
  Recall    : 0.1346
  PR-AUC    : 0.2058   (positive rate 0.0650 = a random classifier's PR-AUC)
  Accuracy  : 0.9237   (always-miss baseline 0.9350  <-- accuracy beats the model here)
```

92% accurate and worse than a constant, while PR-AUC 3× random shows it did learn something.
`class_weight="balanced"` is set where the estimator supports it; `KNeighborsClassifier` and
`MLPClassifier` have no such parameter, and both say so in a comment rather than pretending
otherwise.

---

## The model matrix

`dvc_models_regression.yaml` states its rule at the top, so a gap reads as a decision rather
than an oversight. The sweep is seven regression algorithms — `linear_regression`,
`decision_tree`, `knn`, `polynomial`, `random_forest`, `mlp`, `xgboost` — each appearing **once
per experiment** with the default cleaning. There are exactly three deviations, each carrying its
reason in the file: `decision_tree_overfit`, `random_forest_no_outlier`, and `ridge`/`lasso` in
`experiment_skew` only. `dvc_models_classification.yaml` holds the classification sweep — the
same seven minus `polynomial`, for which there is no classifier script.

Both files no longer drive a `foreach` stage the way they used to — `experiment_raw/`,
`experiment_eng/` and `experiment_classification/` each have one standalone script per model
instead. What the matrix files still do: name the valid model keys and each one's cleaning
variant (`clean:`), which `params.yaml`'s `search.<domain>.<model>` block every script reads its
search budget from. **Adding a model to the matrix file alone does nothing anymore** — you also
need the actual `train_<model>.py` script in every experiment folder you want it to appear in
(see "Adding experiments and data" below).

---

## Adding experiments and data to the pipeline

### 1. A new dataset variant, same generation logic

Add an entry to `dvc_datasets.yaml`. For a Python-generated variant, under `datasets:`:

```yaml
datasets:
  my_new_pool:
    n: 50000
    seed: 7
    input_mode: VELOCITY      # or EFFORT
    elevation_mean: 45.0
    elevation_std: 20.0
    hit_tolerance: 5.0
```

For a Gorillas variant, under `gorillas_datasets:`:

```yaml
gorillas_datasets:
  my_gorillas_pool:
    n: 5000
    seed: 7
    input_mode: EFFORT        # or VELOCITY
    throws: 32
    boards_per_chunk: 25
    hit_tolerance: 5.0
```

The root `dvc.yaml`'s `generate` / `generate_gorillas` `foreach` stages pick up the new key
automatically — no `dvc.yaml` edit needed. It becomes `generate@my_new_pool` /
`generate_gorillas@my_gorillas_pool`, independently cached. Train on it by pointing the
switchable pool at it (`--set-param training_data=my_new_pool`, for the `standard`
experiments only) — a new permanent mode pipeline (like `effort`/`velocity`) is more work; see §4.

### 2. A new algorithm in an existing experiment

Write `models/regression/algorithms/<name>.py` (or `models/classification/algorithms/<name>.py`)
exposing `NAME` and a `fit()` function — this part is unchanged and still the one shared,
DRY piece:

```python
# models/regression/algorithms/my_algorithm.py
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import SomeRegressor

NAME = "My Algorithm"
PARAM_DIST = {"model__some_param": [1, 2, 3]}

def fit(X_train, y_train, feature_step, search_cfg):
    pipeline = Pipeline([feature_step, ("model", SomeRegressor(random_state=42))])
    search = RandomizedSearchCV(pipeline, param_distributions=PARAM_DIST,
                                n_iter=search_cfg["n_iter"], cv=search_cfg["cv"],
                                scoring="neg_mean_squared_error", random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")
    return search.best_estimator_
```

(Classification's `fit(X_train, y_train, search_cfg)` has no `feature_step` — every
classification experiment uses the same engineered features.) Not every algorithm searches:
`ridge.py`/`lasso.py` use a `*CV` meta-estimator with no external wrapper, and
`linear_regression.py` doesn't search at all — see those three for the pattern if yours
doesn't either.

Register it in `models/regression/algorithms/__init__.py`'s `ALGORITHMS` dict, add an entry to
`dvc_models_regression.yaml`/`dvc_models_classification.yaml` (for the search-budget key and
cleaning variant), and add its search budget to `params.yaml`'s `search:` block
(`{n_iter: null}` if it doesn't search). If the experiment folder you're adding it to has a
local `cv:` value (see "Cross-validation" above), no per-algorithm entry is needed there — one
`cv:` value already covers every model in that folder.

**Then write the actual script(s)** — unlike the old `train.py`/`runs.py` system, registering a
model in the matrix file no longer makes it appear anywhere by itself. Copy an existing
`train_<model>.py` in the experiment folder you want it in (e.g.
`models/regression/standard/experiment_raw/train_random_forest.py`) and swap the algorithm
import/`KEY`/`NAME` references. Then add its DVC stage to that pipeline's `dvc.yaml`, copied
from a sibling stage block with the model name swapped throughout (`cmd:`, every `deps:`/
`outs:`/`metrics:` path, and the `search.<domain>.<name>` params keys).

**Switching a model off** is simpler than adding one: delete (or comment out) its stage block
in the relevant `models/<domain>/<mode>/<experiment>/dvc.yaml`. Confirm with:

```bash
dvc stage list --all | grep my_algorithm      # --all for every pipeline; should print nothing once it's off
```

This only removes it from *that* experiment/pipeline — a model can exist in `experiment_raw`
but not `experiment_eng`, or in `standard` but not `effort`, with no extra
mechanism needed (there's no shared matrix stage to keep in sync anymore, unlike before). There
is no single switch that turns a model off everywhere at once — that's the deliberate tradeoff
for "no dispatch to trace, one file per model."

Two things this does **not** do:

- **It doesn't delete anything already produced.** `experiments_results/.../model_<name>.joblib`
  and its metrics stay on disk — DVC only manages stages it currently knows about, not history.
  Remove them by hand, or `dvc gc` once you're sure you don't want to switch it back on.

### 3. A new evaluation/plotting script

Not yet part of the DAG (`evaluation/` is run by hand — see "Known state"). Follow an existing
script's pattern for the experiment you're targeting; every evaluation script is self-contained
now (its own `DATA`/`FEATURES`/`N_SAMPLES` constants, no shared registry to import):

```bash
python evaluation/regression/run_raw/my_script.py
```

### 4. A new experiment folder (e.g. a new demo, mirroring `experiment_skew`)

For a new standalone experiment — its own dataset slice or view, its own set of standalone
per-model scripts, its own `experiments_results/.../` output directory. This is more work than
§1–3, but `models/regression/standard/experiment_skew/` and
`models/regression/*/experiment_row_count/` are full worked examples to copy from.

1. **A folder**: `models/<domain>/<mode>/experiment_<name>/`, with an `__init__.py` and one
   `train_<model>.py` per model you want in it. Each script is self-contained: explicit `DATA`
   path, explicit `MODELS_DIR` (under `experiments_results/<domain>/<mode>/experiment_<name>/
   models/`), a `main()` that loads data (`models/<domain>/common/loader.py`), splits
   (`splitting.py`, `stratify=True` for classification), calls the shared
   `algorithms/<model>.py`'s `fit()`, and saves the model + metrics.
2. **A pipeline file**: `models/<domain>/<mode>/experiment_<name>/dvc.yaml` — one file per
   {domain, mode, experiment}, in the same folder as the scripts from step 1. No `foreach`,
   since each model is already its own script; one explicit stage per model. Copy
   `models/regression/standard/experiment_skew/dvc.yaml` for the exact stage shape (`cmd:`
   routes through `../../../../scripts/run_with_pythonpath.py ../../../.. -m
   models.<domain>.<mode>.experiment_<name>.train_<model>` — note both relative depths are 4
   levels, since this file lives four directories under the repo root; `deps:` needs the script itself,
   `algorithms/<model>.py`, `common/loader.py`, `feature_engineering.py`, `splitting.py`, and
   the data file; `outs:`/`metrics:` point at `experiments_results/<domain>/<mode>/
   experiment_<name>/models/`). If the new experiment needs BOTH a regression and a
   classification version (as `experiment_row_count` does), it gets one `dvc.yaml` in each
   domain's folder. Keep the two domains' stage names distinct anyway — regression and
   classification share several algorithm names (`decision_tree`, `knn`, `random_forest`,
   `mlp`, `xgboost`), so identical names would collide in `dvc repro -R models/...` output and
   in `dvc exp show` (see `train_row_count_classification_<domain>_<model>` vs.
   `train_row_count_<domain>_<model>` for the precedent).
3. **Run it**: `dvc repro models/<domain>/<mode>/experiment_<name>/dvc.yaml:train_<name>_<model>`
   for one script, `cd models/<domain>/<mode>/experiment_<name> && dvc repro` for everything in
   that one experiment, or `dvc repro -R models/<domain>/<mode>` for every experiment in that
   mode.

Worth checking `dvc dag --full` after adding stages (a parse error names the exact stage and
line), and a real (non-`--dry`) run of one cheap stage before trusting the rest — `--dry`
cannot distinguish "this cross-pipeline dependency doesn't exist yet" from "this path is
wrong," since it never executes the upstream stage that would create the file.

---

## Things that will stop you

These are deliberate. Each converts a silent substitution into an error, because the silent
version produced results that looked fine and were not.

| Error | Cause | Fix |
|---|---|---|
| `ContractViolation: column set/order does not match` | A pool on disk predates the current schema. `data/` is DVC-cached, not in Git, so it can be older than the code that reads it | `dvc repro generate` / `dvc repro generate_gorillas`. The error prints this. Nothing is lost — generation is deterministic |
| `PoolTooSmall: n_samples=40,000 exceeds the 5,000 rows` | A Gorillas pool with the default `n_samples` | `--set-param n_samples=5000` |
| `GameRunFailed: N worker(s) hit the timeout` | A half-written CSV would make the dataset depend on machine speed rather than `--seed` | Raise `--timeout`, lower `--boards-per-chunk`, or pass `--allow-partial` to accept a non-reproducible run |
| `DosboxNotFound` | DOSBox is not installed or not where it's looked for | Install DOSBox Staging, or set `GORILLAS_DOSBOX` |
| `UnreachableRangeError` | A sampling distribution in `dvc_datasets.yaml` mostly or never lands in its valid range | Fix the distribution. A *rarely*-reachable range warns instead, saying the accepted values are a truncated tail rather than what you configured |
| `NonTerminatingShotError` | A rare combination (very low gravity-outlier draw + high launch angle) exceeds `physics.simulate_landing`'s `max_time` — a real, known-rare simulator edge case, not specific to either input mode | Try a different seed for that dataset entry in `dvc_datasets.yaml` (e.g. `standard_effort` uses seed 47, not 43, for exactly this reason) |

---

## Known state

`AUDIT.md` is a full audit of this repo with a numbered plan; it is the authoritative record
of what is and isn't done, and it carries an implementation log — read tasks 40/41/42/43 there
for the full story behind today's restructuring (real EFFORT mode, `pipelines/` cut from six
files to three then reshaped one-per-mode then split again to one `dvc.yaml` per {mode,
experiment}, and finally moved out of `pipelines/` into each experiment's own
`models/<domain>/<mode>/<experiment>/` folder, every matrix-driven stage converted to one script per model, and a same-day
reversal of a cross-source join that briefly existed between task 40 and task 43 — standard and
Gorillas pools are trained on separately, by design, not merged). Currently open and worth
knowing about:

- **`data/gorillas_velocity.parquet` has not been generated in every environment this repo has
  been touched from** — DOSBox isn't available everywhere. `data/gorillas_effort.parquet` does
  exist in at least one environment this session used, and confirmed two `models/regression/effort/*`
  stages run for real against it; the velocity side is wired and `--dry`-verified only. A real
  `dvc repro -P` on a machine with DOSBox for both modes is still needed to call this fully
  proven end to end.
- **The pipeline has never been fully reproduced on one machine**, and now has 16 separate
  lock files — one per `dvc.yaml`, each tracking only its own stages. Task 12.
- **Pre-split cleaning leaks** in the regression loader — IQR quantiles over the whole pool,
  and row filtering on the target. Task 14.
- **`clean: "range"` discards valid data**, as above. Task 13b.
- **No tests, no CLAUDE.md.** Tasks 26 and 28.
- `evaluation/` (~2,500 LOC) sits outside the DAG and is the least reproducible part of the
  repo, despite producing the figures the talk uses. Tasks 17 and 27.
