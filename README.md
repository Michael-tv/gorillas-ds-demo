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

Both emit the **same 14-column contract**, so either can fill the training slot. The
simulator gives a well-behaved dataset; the game gives a realistically awkward one, where
the thing you want to predict genuinely is not a function of the columns you logged. Swapping
between them is one line in `params.yaml`.

---

## Quick start

```bash
pipenv install                      # Python 3.13; see Pipfile
pipenv shell

dvc repro generate                  # build the simulated pool (~1 min)
dvc repro train_raw@random_forest    # train one model on it
dvc repro                            # the whole DAG: 43 stages
```

If `pipenv` picks up the wrong virtualenv, prefix with `PIPENV_IGNORE_VIRTUALENVS=1`.

There is **no DVC remote, by design.** Both producers are deterministic for a fixed seed, so
regeneration — not a stored copy — is the recovery path. A fresh clone therefore starts with
`dvc repro generate`, not `dvc pull`.

---

## What's here

```
params.yaml            ← the knobs you actually turn
dvc_datasets.yaml      ← which datasets exist, and their generation parameters
dvc_models.yaml        ← which models each training stage expands into
dvc.yaml               ← the pipeline: 8 stage groups, 42 stages

physics.py             RK4 projectile integrator with drag and wind
data_generation/       the two producers, plus the shared contract
feature_engineering.py the 3 derived features, as a function and a Pipeline step
splitting.py           group-aware train/test splits and cross-validation
params.py              reads params.yaml; also holds take_samples()

models/regression/     9 regression algorithms + 3 concept runs
models/classification/ 6 classification algorithms
evaluation/            prediction, feature and plotting scripts (run by hand)
experiments/           trained models (DVC-cached) and metrics (Git-versioned)
qbasic_gorillas/       the DOSBox game builds, including the datagen fork
```

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
stored**. They are derived at training time, inside the model's sklearn `Pipeline`, so the
saved `.joblib` carries its own preprocessing and can be fed raw physical inputs.

`group_id` is what makes honest evaluation possible. The game throws 32 bananas at each
board, and a board's wind and skyline are shared by all of them, so a random split puts the
same board on both sides and scores the model against rows it has effectively already seen.
Measured on a 60-board pool: a random split reports MAE 12.944 where the group-aware split
reports 16.157 — **20% optimistic**. `splitting.py` handles this, and because the simulated
pool gives every row its own `group_id`, the same code degrades to an ordinary random split
there. One path, both producers, no branching in any training script.

---

## Running it end to end

### 1. Generate data

```bash
dvc repro generate                                  # all entries in dvc_datasets.yaml
dvc repro generate@standard_training_data           # just one
```

For the game producer (Windows, DOSBox Staging installed):

```bash
setx GORILLAS_DOSBOX "C:\Program Files\DOSBox Staging\dosbox.exe"   # once, optional
dvc repro generate_gorillas
```

DOSBox is located via `--dosbox` → `$GORILLAS_DOSBOX` → `dosbox` on `PATH` → known install
locations. Each variant writes both a parquet and a raw throw log
(`data/<name>_throws.csv`), which keeps the `board` and `outcome` columns the 14-column
contract deliberately omits.

`--workers` only controls how many DOSBox instances run at once and **cannot** change the
resulting data — every chunk seed is drawn single-threaded before the thread pool starts.
`--boards-per-chunk` and `--throws` **do** change the data, which is why they live in
`dvc_datasets.yaml` and `--workers` does not.

You can also run the game interactively from `qbasic_gorillas/dosbox-datagen/`:
`Play Effort.bat` or `Play Velocity.bat`.

### 2. Train

```bash
dvc repro                                   # everything
dvc repro train_raw@random_forest            # one model
dvc repro train_classification               # one group
```

Eight stage groups:

| Stage | What it shows |
|---|---|
| `generate` / `generate_gorillas` | the two producers |
| `filter_skewed` | splits the pool at `skew.max_angle_deg` into a low-angle training slice and an out-of-distribution holdout |
| `train_raw` | 9 models on the 9 raw columns |
| `train_eng` | the same 9 models, with the 3 derived features — same data, different first Pipeline step |
| `train_skewed` | the same models on a low-angle slice, for extrapolation |
| `train_classification` | 6 models on `hit_target` |
| `train_leakage_{clean,leaky}` | leakage, side by side |
| `train_bias_variance_{under,over}fitting` | the bias/variance pair |
| `train_{skewed,balanced}_concept` | distribution shift |

Models land in `experiments/…/models/model_<key>.joblib` (DVC-cached) and metrics in
`metrics_<key>.csv` (Git-versioned, so `dvc metrics diff` works across commits).

### 3. Compare

```bash
dvc exp show                    # params and metrics, side by side
dvc metrics diff                # against HEAD
dvc dag                          # the pipeline as a graph
```

### 4. Evaluate and plot

The `evaluation/` scripts are run by hand — they are **not** in the DAG yet. Most are
self-contained; the six `evaluate_features.py` scripts import `train_utils`, so they need the
same PYTHONPATH wrapper the training stages use:

```bash
python evaluation/regression/run_raw/predict.py                      # self-contained
python evaluation/regression/run_raw/plot_training_data_relationships.py
python evaluation/regression/compare_models.py

python scripts/run_with_pythonpath.py models/regression/run_raw \
       evaluation/regression/run_raw/evaluate_features.py            # needs the wrapper
```

`predict.py` takes its inputs from a config block at the bottom of the file, including
`model_name` — which must be a key trained in that run (`ridge` and `lasso` exist only under
`run_skewed`).

---

## The experiments — what to change to show what

Everything below is one edit in `params.yaml`, or one `--set-param`, and nothing else.

**Which dataset?** `training_data: standard_training_data` → `gorillas_effort_data`. Every
training and evaluation script follows. This is the headline comparison: the same models on
clean simulated data versus data from a real game with an unobserved confounder.

```bash
dvc exp run --set-param training_data=gorillas_effort_data --set-param n_samples=5000
```

`n_samples` must be set too — a Gorillas pool holds 5,000 rows, and a sample size the pool
cannot honour is an error rather than a silent truncation, because silently training on 5,000
would make the run incomparable with every other size tier.

**Does more data help?** `n_samples: 40000`. Tiers are nested prefixes of one pre-shuffled
pool, so growing it is the only thing that changes.

```bash
dvc exp run --set-param n_samples=10000 && dvc exp run --set-param n_samples=20000
```

**Raw or engineered features?** `train_raw` vs `train_eng`. Same data, same models; the only
difference is the first step of the Pipeline.

**Does cleaning help?** `clean` in `dvc_models.yaml` — `""`, `"range"` or `"no_outlier"`.
`random_forest_no_outlier` exists to make this contrast visible in `dvc exp show`.

> **Caveat, and it matters:** `clean: "range"` bounds `landing_distance_m` at 0, which
> discards genuine headwind landings — 727 of the 964 rows it drops. As it stands, the range
> demo mostly removes *real* data. See task 13b in `AUDIT.md`.

**Leakage:** `train_leakage_clean` vs `train_leakage_leaky`.

**Overfitting:** `train_bias_variance_underfitting` vs `…_overfitting`, or the
`decision_tree_overfit` model — the same algorithm with no depth limit.

**Distribution shift:** `train_skewed` trains on a low-angle slice only (`skew.max_angle_deg`),
and `filter_skewed` keeps the excluded rows as `data/skewed_holdout.parquet` — so the claim is
measured on angles the model never saw, not on a test set that shares its blind spot:

| | in-distribution MAE | holdout MAE | |
|---|---|---|---|
| skewed | 4.383 | 8.847 | **2.0× worse** |
| balanced | 7.530 | 6.234 | 0.8× — no penalty |

Note the inversion: the narrow model looks *better* in-distribution precisely because its test set
has the same hole. `train_balanced_concept` is the control — same architecture, same split, and
sized by reading the skewed run's row count. Beyond the training range a linear model keeps
extrapolating while a forest flatlines at its boundary leaf mean, which is why `ridge` and `lasso`
appear in `skewed_models` and nowhere else.

```bash
dvc exp run --set-param skew.max_angle_deg=40    # move the cap; both runs follow
```

**Class imbalance:** switch to a Gorillas pool and run any classification stage. The hit rate
is ~7%, so predicting "miss" every time scores ~93%. The metrics report leads with
precision/recall/PR-AUC and prints the always-miss baseline beside accuracy:

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

`dvc_models.yaml` states its rule at the top, so a gap reads as a decision rather than an
oversight. The sweep is seven regression algorithms — `linear_regression`, `decision_tree`,
`knn`, `polynomial`, `random_forest`, `mlp`, `xgboost` — each appearing **once per run** with
the default cleaning. There are exactly three deviations, each carrying its reason in the
file: `decision_tree_overfit`, `random_forest_no_outlier`, and `ridge`/`lasso` in
`skewed_models` only.

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

---

## Known state

`AUDIT.md` is a full audit of this repo with a numbered plan; it is the authoritative record
of what is and isn't done, and it carries an implementation log. Currently open and worth
knowing about:

- **The pipeline has never been fully reproduced.** `dvc.lock` covers 4 of 43 stages, and its
  entries predate the current code. Everything in `experiments/` was produced outside DVC.
  Task 12.
- **Pre-split cleaning leaks** in the regression loader — IQR quantiles over the whole pool,
  and row filtering on the target. Task 14.
- **`clean: "range"` discards valid data**, as above. Task 13b.
- **No tests, no CLAUDE.md.** Tasks 26 and 28.
- `evaluation/` (~2,500 LOC) sits outside the DAG and is the least reproducible part of the
  repo, despite producing the figures the talk uses. Tasks 17 and 27.
