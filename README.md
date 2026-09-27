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

dvc repro generate                          # build the simulated pool (~1 min)
cd pipelines/standard_regression && dvc repro train_raw@random_forest && cd ../..   # train one model on it
dvc repro -P                                 # the whole repo: 7 pipelines, 93 stages
```

The pipeline is split into seven `dvc.yaml` files — one root file with the two shared data
producers, and six standalone pipelines under `pipelines/`, one per {source} × {domain}
combination. `dvc repro` from the repo root only touches the root file's 5 stages; `-P` /
`--all-pipelines` is what runs everything. See "Running one pipeline at a time" below.

If `pipenv` picks up the wrong virtualenv, prefix with `PIPENV_IGNORE_VIRTUALENVS=1`.

There is **no DVC remote, by design.** Both producers are deterministic for a fixed seed, so
regeneration — not a stored copy — is the recovery path. A fresh clone therefore starts with
`dvc repro generate`, not `dvc pull`.

---

## What's here

```
params.yaml                    ← the knobs you actually turn
dvc_datasets.yaml              ← which datasets exist, and their generation parameters
dvc_models_regression.yaml     ← which regression models each training stage expands into
dvc_models_classification.yaml ← which classification models each training stage expands into
dvc.yaml                       ← ROOT pipeline: the two shared data producers, 5 stages

pipelines/                     ← six standalone pipelines, one per {source} x {domain}
  standard_regression/dvc.yaml   train_raw, train_eng, filter_skewed, train_skewed,
                                  leakage and bias/variance demos — 34 stages
  standard_classification/dvc.yaml  train_classification — 6 stages
  effort_regression/dvc.yaml     train_raw_effort, train_eng_effort — 18 stages
  effort_classification/dvc.yaml train_classification_effort — 6 stages
  velocity_regression/dvc.yaml   train_raw_velocity, train_eng_velocity — 18 stages
  velocity_classification/dvc.yaml  train_classification_velocity — 6 stages

physics.py             RK4 projectile integrator with drag and wind
data_generation/       the two producers, plus the shared contract
feature_engineering.py the 3 derived features, as a function and a Pipeline step
splitting.py           group-aware train/test splits and cross-validation
params.py              reads params.yaml; also holds take_samples()

models/regression/     9 regression algorithms, on 3 pools: switchable (run_raw/run_eng)
                        plus 2 permanent Gorillas groups (run_raw_effort/velocity,
                        run_eng_effort/velocity) + 3 concept runs
models/classification/ 6 classification algorithms, on the switchable pool (run/) plus
                        2 permanent Gorillas groups (run_effort, run_velocity)
evaluation/             prediction, feature and plotting scripts (run by hand)
experiments/            trained models (DVC-cached) and metrics (Git-versioned)
qbasic_gorillas/        the DOSBox game builds, including the datagen fork
```

**Switchable pool vs. permanent pipelines.** `train_raw`/`train_eng`/`train_classification`
(in `pipelines/standard_*/`) read whichever pool `params.yaml: training_data` names — one
`--set-param` retargets them at any pool, including a Gorillas one, without touching another
file (see "Which dataset?" below). The Gorillas effort/velocity split is the exception: those
are **four fixed pipelines** (`effort_regression`, `effort_classification`,
`velocity_regression`, `velocity_classification`) that always exist on disk side by side, each
reading its own separately-generated file, at the cost of running DOSBox four times (once per
mode × domain) instead of twice. Compare them with `dvc exp show` without re-running anything.
See "Running one pipeline at a time" and "Adding a new standalone pipeline" below for how this
is built, and the design note at the top of `dvc.yaml` (the root one) for why regression and
classification each get their own generated Gorillas file even though one throw carries both
targets.

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
locations. `generate_gorillas` expands into four independently-cached variants —
`gorillas_effort_regression`, `gorillas_effort_classification`, `gorillas_velocity_regression`,
`gorillas_velocity_classification` — one DOSBox run each, at a different seed, so regression
and classification are fully separate pipeline branches sharing no stage node with each other
(the repo owner's choice; the game itself doesn't distinguish "regression data" from
"classification data" — one throw carries both `initial_velocity_ms` and `hit_target`). Each
variant writes both a parquet and a raw throw log (`data/<name>_throws.parquet`), which keeps
the `board` and `outcome` columns the 14-column contract deliberately omits.

`--workers` only controls how many DOSBox instances run at once and **cannot** change the
resulting data — every chunk seed is drawn single-threaded before the thread pool starts.
`--boards-per-chunk` and `--throws` **do** change the data, which is why they live in
`dvc_datasets.yaml` and `--workers` does not.

You can also run the game interactively from `qbasic_gorillas/dosbox-datagen/`:
`Play Effort.bat` or `Play Velocity.bat`.

### 2. Train

Training stages live in `pipelines/<name>/dvc.yaml`, not the root one — the root file holds
only the two data producers from step 1. Three ways to run them, cheapest first:

```bash
dvc repro -P                                          # every pipeline in the repo, in order
dvc repro pipelines/standard_regression/dvc.yaml:train_raw@random_forest   # one stage, from the repo root
cd pipelines/standard_regression && dvc repro && cd ../..                  # one whole pipeline
```

`-P` (`--all-pipelines`) discovers and reproduces all seven `dvc.yaml` files, resolving
cross-pipeline dependencies by file path — a `pipelines/*/dvc.yaml` stage that reads
`../../data/gorillas_effort_regression.parquet` is recognized as depending on the root
`generate_gorillas@gorillas_effort_regression` stage automatically, the same way two stages in
one file depend on each other. See "Running one pipeline at a time" below for the full pattern
and the two things that are NOT obvious about it (relative paths, and where `params:` looks for
`params.yaml`).

**All four Gorillas variants, generation through training** — the four fixed pipelines,
without the two switchable ones:

```bash
dvc repro generate_gorillas   # all 4 variants, from the root
for p in effort_regression effort_classification velocity_regression velocity_classification; do
  (cd "pipelines/$p" && dvc repro)
done
```

This runs 4 DOSBox sessions and 48 training stages (4 regression pipelines × 9 models = 36, 2
classification pipelines × 6 models = 12).

Seven pipelines:

| Pipeline | Stages | What it shows |
|---|---|---|
| `dvc.yaml` (root) | `generate`, `generate_gorillas` | the two producers |
| `pipelines/standard_regression` | `train_raw`, `train_eng`, `filter_skewed`, `train_skewed`, `train_{skewed,balanced}_concept`, `train_leakage_{clean,leaky}`, `train_bias_variance_{under,over}fitting` | the 9-model sweep (raw and engineered features), skew/leakage/bias-variance demos — all on the switchable `training_data` pool |
| `pipelines/standard_classification` | `train_classification` | 6 models on `hit_target`, switchable pool |
| `pipelines/effort_regression` | `train_raw_effort`, `train_eng_effort` | the 9-model sweep, permanently on the Gorillas effort pool |
| `pipelines/effort_classification` | `train_classification_effort` | 6 models, permanently on the Gorillas effort pool |
| `pipelines/velocity_regression` | `train_raw_velocity`, `train_eng_velocity` | the 9-model sweep, permanently on the Gorillas velocity pool |
| `pipelines/velocity_classification` | `train_classification_velocity` | 6 models, permanently on the Gorillas velocity pool |

Models land in `experiments/…/models/model_<key>.joblib` (DVC-cached) and metrics in
`metrics_<key>.csv` (Git-versioned, so `dvc metrics diff` works across commits) — the same
`experiments/` layout regardless of which pipeline produced them.

### 3. Compare

```bash
dvc exp show                    # params and metrics, side by side
dvc metrics diff                # against HEAD
dvc dag                          # every stage in the repo as one graph
```

Unlike `repro`, `dag` finds stages across **all** `dvc.yaml` files by default — no `-P` needed;
pass a stage name (optionally `path/to/dvc.yaml:stage`) to see just its subgraph.

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

### 5. Running one pipeline at a time

The repo owner's choice, so "run everything for effort" (or velocity, or the standard-pool
regression sweep) is one command instead of naming stages individually. Three equivalent ways
to run a whole pipeline:

```bash
cd pipelines/effort_regression && dvc repro && cd ../..     # from inside its directory
dvc repro pipelines/effort_regression/dvc.yaml               # from the repo root, by path
dvc repro -P                                                  # every pipeline, including this one
```

Two things about this pattern that are not obvious from the stage files themselves, both found
by testing rather than assumed:

- **Every path inside `pipelines/<name>/dvc.yaml` is `../../`-prefixed.** DVC resolves a
  stage's `deps`/`outs`/`cmd` relative to the `dvc.yaml` that declares it, not to the repo root
  or wherever `dvc repro` was invoked from — so `models/regression/train_knn.py` at the repo
  root becomes `../../models/regression/train_knn.py` inside a two-levels-deep pipeline file.
  Cross-pipeline dependencies (a stage here reading a file the root `generate_gorillas`
  produces) need no special syntax; DVC matches them by the resolved absolute path, the same as
  any two stages in one file.
- **A stage's `params:` list needs the file spelled out.** The `vars:` block at the top of each
  pipeline file (`- ../../params.yaml`) only controls `${...}` template interpolation at parse
  time. A stage's own `params:` — the list DVC uses for staleness tracking and `dvc exp show` —
  is a separate mechanism that otherwise defaults to a `params.yaml` **next to that dvc.yaml**,
  which doesn't exist here. Every `params:` block in `pipelines/*/dvc.yaml` therefore names the
  file explicitly:

  ```yaml
  params:
    - ../../params.yaml:
        - test_size
        - search.regression.${item.model}.n_iter
  ```

  Leaving this implicit fails loudly and immediately (`dvc repro` errors with "Parameters
  'test_size, ...' are missing from 'params.yaml'"), rather than silently tracking nothing — so
  if you add a `params:` entry and see that error, this is why.

`dvc dag` (no `-P` needed — see "3. Compare") and `dvc stage list` both work the same way run
from any pipeline's own directory, scoped to just that file, or from the repo root against
`pipelines/<name>/dvc.yaml:<stage>`.

---

## The experiments — what to change to show what

Everything below is one edit in `params.yaml`, or one `--set-param`, and nothing else. Every
`dvc exp run`/`dvc repro` in this section targets a stage that now lives under `pipelines/`
(see "5. Running one pipeline at a time" above) — run it from inside that pipeline's directory,
or prefix the target with its path from the repo root
(`pipelines/standard_regression/dvc.yaml:train_raw`). Most examples below are
`pipelines/standard_regression` or `pipelines/standard_classification`; noted where it differs.

**Which dataset?** Two different mechanisms, depending on which comparison you want:

- **Switchable pool** (`pipelines/standard_regression` or `pipelines/standard_classification`)
  — `training_data: standard_training_data` → `gorillas_effort_regression` (or
  `gorillas_velocity_regression`) retargets `train_raw`/`train_eng`/`train_classification` at a
  different pool without touching any other file. This is the headline comparison: the same
  models on clean simulated data versus data from a real game with an unobserved confounder.

  ```bash
  cd pipelines/standard_regression
  dvc exp run --set-param training_data=gorillas_effort_regression --set-param n_samples=5000
  ```

  `n_samples` must be set too — a Gorillas pool holds 5,000 rows, and a sample size the pool
  cannot honour is an error rather than a silent truncation, because silently training on 5,000
  would make the run incomparable with every other size tier. Note `training_data` doesn't
  distinguish regression from classification purpose — a classification stage reading
  `gorillas_effort_regression` still trains on `hit_target` fine, since both targets live in
  every pool. The `_regression`/`_classification` split only matters for the four **permanent**
  Gorillas pools below.

- **Permanent pipelines** (`pipelines/effort_regression`, `pipelines/velocity_regression`, and
  their classification counterparts) — always exist on disk; no `--set-param` needed to compare
  them, just `dvc exp show` or read both `experiments/.../run_raw_effort/` and
  `.../run_raw_velocity/` directly.

**Does more data help?** `n_samples: 40000`. Tiers are nested prefixes of one pre-shuffled
pool, so growing it is the only thing that changes. (`pipelines/standard_regression`.)

```bash
cd pipelines/standard_regression
dvc exp run --set-param n_samples=10000 && dvc exp run --set-param n_samples=20000
```

**Raw or engineered features?** `train_raw` vs `train_eng`. Same data, same models; the only
difference is the first step of the Pipeline.

**Does cleaning help?** `clean` in `dvc_models_regression.yaml` — `""`, `"range"` or `"no_outlier"`.
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
cd pipelines/standard_regression
dvc exp run --set-param skew.max_angle_deg=40    # move the cap; both runs follow
```

**Class imbalance:** run any classification stage against a Gorillas pool — either
`pipelines/standard_classification` with `--set-param training_data=gorillas_effort_classification`,
or one of the two permanent Gorillas classification pipelines directly. The hit rate is ~7%, so
predicting "miss" every time scores ~93%. The metrics report leads with precision/recall/PR-AUC
and prints the always-miss baseline beside accuracy:

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
per run** with the default cleaning. There are exactly three deviations, each carrying its
reason in the file: `decision_tree_overfit`, `random_forest_no_outlier`, and `ridge`/`lasso` in
`skewed_models` only. `dvc_models_classification.yaml` holds the classification sweep — the
same seven minus `polynomial`, for which there is no classifier script.

Both files drive every run of their domain — `regression_models` feeds `train_raw`, `train_eng`
**and** the four permanent Gorillas regression groups; `classification_models` feeds
`train_classification` and its two permanent Gorillas groups. Add a model once, it appears
everywhere that domain trains.

---

## Adding experiments and data to the pipeline

Four things you might want to add, roughly cheapest to most involved.

### 1. A new dataset variant, same generation logic

Add an entry to `dvc_datasets.yaml`. For a Python-generated variant, under `datasets:`:

```yaml
datasets:
  my_new_pool:
    n: 50000
    seed: 7
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
`generate_gorillas@my_gorillas_pool`, independently cached. Train on it either by pointing the
switchable pool at it (`--set-param training_data=my_new_pool`, from
`pipelines/standard_regression` or `standard_classification`) or, if it needs its own permanent
pipeline, see §4.

### 2. A new algorithm in an existing sweep

Write `models/regression/train_<name>.py` (or `models/classification/train_<name>.py`),
matching the shape every existing script uses:

```python
from train_utils import load_data, print_metrics, model_path, FEATURE_STEP  # FEATURE_STEP: regression only
import params, splitting

X, y, groups = load_data()
X_train, X_test, y_train, y_test, groups_train = splitting.split(
    X, y, groups, test_size=params.load_params()["test_size"], random_state=42)
CV_FOLDS = splitting.cv_for(CV, X_train, y_train, groups_train)   # only if it searches
# ... fit, print_metrics(y_test, preds), joblib.dump(..., model_path("model_<name>.joblib"))
```

`splitting.split`/`cv_for` are mandatory, not optional — skip them and the model trains on a
plain `train_test_split`, silently reintroducing the group leakage `group_id` exists to prevent
whenever a Gorillas pool is active.

Add an entry to `dvc_models_regression.yaml` or `dvc_models_classification.yaml`:

```yaml
regression_models:
  my_algorithm: {script: train_my_algorithm.py, clean: "", model: my_algorithm}
```

Add its search budget to `params.yaml`'s `search:` block (`{n_iter: null, cv: null}` if it
doesn't search). It now appears in **every stage, in every pipeline,** that expands
`${regression_models}` — `train_raw`/`train_eng` in `pipelines/standard_regression`, and the
four permanent Gorillas regression pipelines — with no further edits. That fan-out is the point
of keeping one matrix file per domain, loaded by every pipeline of that domain's `vars:` block
(§ "The model matrix").

**Switching a model off** is the same file, the other direction: comment out (or delete) its
entry in `dvc_models_regression.yaml` / `dvc_models_classification.yaml`.

```yaml
regression_models:
  linear_regression:  {script: train_linear_regression.py, clean: "", model: linear_regression}
  # decision_tree:     {script: train_decision_tree.py,     clean: "", model: decision_tree}
  knn:                {script: train_knn.py,               clean: "", model: knn}
```

No other file changes — there's no separate `enabled: false` flag, presence in the matrix *is*
the switch. This drops the model from every pipeline of that domain at once (the same fan-out
that added it), so a commented-out `decision_tree` disappears from `train_raw`, `train_eng` and
all four Gorillas regression pipelines together, not one at a time. Confirm with:

```bash
dvc stage list --all | grep decision_tree      # --all for every pipeline; should print nothing once it's off
```

Two things this does **not** do:

- **It doesn't delete anything already produced.** `experiments/.../model_decision_tree.joblib`
  and its metrics stay on disk — DVC only manages stages it currently knows about, not history.
  Remove them by hand, or `dvc gc` once you're sure you don't want to switch it back on.
- **It doesn't disable one model in one group only.** To keep an algorithm in `train_raw` but
  drop it from `train_eng` (or vice versa), that's no longer "the same model, different run" —
  give it a second matrix entry with a different key (`decision_tree_engonly`, say) and only
  reference that key from the stage you want, the same way `random_forest_no_outlier` exists
  alongside plain `random_forest` today.

### 3. A new evaluation/plotting script

Not yet part of the DAG (`evaluation/` is run by hand — see "Known state"). Follow an existing
script's pattern for the run you're targeting and, if it imports `train_utils`, invoke it
through the same wrapper the training stages use:

```bash
python scripts/run_with_pythonpath.py models/regression/run_raw evaluation/regression/run_raw/my_script.py
```

### 4. A new standalone pipeline

For when a dataset should always have its own dedicated, always-on-disk model group, reached by
`cd`-ing into its own directory or `dvc repro -P`, rather than through the switchable
`training_data` pool. This is more work than §1–3, but every piece already has a template to
copy — it's exactly how `pipelines/effort_regression` was built. Using it as the worked
example:

1. **Dataset** — an entry in `dvc_datasets.yaml` (§1). Give it its own name; don't reuse a name
   another pipeline already owns, since these files are meant to be permanent, not overwritten.
2. **Config shim(s)** — `models/<domain>/<new_run_folder>/train_utils.py`, copied from
   `models/regression/run_raw_effort/train_utils.py` (or the `_eng`/classification equivalent)
   with `DATA` repointed at the new file and `N_SAMPLES` set — `None` trains on the whole pool,
   a number prefix-slices it (and must not exceed the pool's row count, or `params.take_samples`
   raises rather than silently truncating). One shim per stage the new pipeline needs
   (`run_raw_*`/`run_eng_*` if it's a regression pipeline with both feature sets).
3. **Pipeline file** — `pipelines/<new_name>/dvc.yaml`, copied from
   `pipelines/effort_regression/dvc.yaml`. Three things to get right, all found by testing this
   pattern rather than assumed (see "5. Running one pipeline at a time" for the reasoning):
   - `vars:` loads `../../dvc_models_regression.yaml` (or `_classification.yaml`) and
     `../../params.yaml` — two levels up, since the new file lives two directories under the
     repo root.
   - Every `deps`/`outs`/`cmd` path gets the same `../../` prefix.
   - Every `params:` block names `../../params.yaml` explicitly (`- ../../params.yaml:` then
     the key list) — the bare-key form silently looks for a `params.yaml` next to the new
     `dvc.yaml`, which doesn't exist, and fails with "Parameters '...' are missing from
     'params.yaml'".
4. **Run it**: `cd pipelines/<new_name> && dvc repro`, or `dvc repro pipelines/<new_name>/dvc.yaml`
   from the repo root.

That's the whole pattern — six pipeline files this way (four regression, two classification) is
what took the DAG from 43 to 93 stages for the effort/velocity split. Worth checking
`dvc stage list` from inside the new directory after adding one, and a real (non-`--dry`) run of
a single cheap stage before trusting the rest — `--dry` cannot distinguish "this cross-pipeline
dependency doesn't exist yet" from "this path is wrong," since it never executes the upstream
stage that would create the file.

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

- **The pipeline has never been fully reproduced, and now has seven separate lock files** — one
  per `dvc.yaml` (root + six under `pipelines/`), each tracking only its own stages. Currently:
  `pipelines/effort_regression/dvc.lock` covers 1 of 93 stages (`train_raw_effort@linear_regression`,
  verified while building the `pipelines/` split); the root `dvc.lock` and the other five are
  absent. Task 12.
- **Pre-split cleaning leaks** in the regression loader — IQR quantiles over the whole pool,
  and row filtering on the target. Task 14.
- **`clean: "range"` discards valid data**, as above. Task 13b.
- **No tests, no CLAUDE.md.** Tasks 26 and 28.
- `evaluation/` (~2,500 LOC) sits outside the DAG and is the least reproducible part of the
  repo, despite producing the figures the talk uses. Tasks 17 and 27.
