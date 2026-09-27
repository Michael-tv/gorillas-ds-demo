# Repository audit — shortcomings & refactor plan

Audit date: 2026-09-27 · Audited at `5ce27f3` · DVC 3.67.1 · 76 stages (→ 77 with
`filter_skewed`, then far fewer once the model matrices are cut)
Last updated: 2026-09-27 on branch `audit/phase-1` (§0)

**Rescanned 2026-09-27 after the QBasic Gorillas integration** — a second data producer,
`params.yaml: training_data`, and `params.data_path()`. See §5 and Phase 1b. Two earlier
recommendations were wrong and are corrected: purging `qbasic_gorillas/` (§3, C8, task 30)
and the no-remote decision.

**Health score: 4 / 10 at audit time; 5 / 10 as of the 2026-09-27 update.** The architecture is
well-conceived; it has never been executed end-to-end, and the reproducibility layer is
load-bearing in name only. Two of the seven critical risks (#3, #4) are now closed, which is
what moves the score — but risk #2 (the pipeline has never been reproduced) is untouched, and
that is the one gating everything in phases 6–9.

**Implementation status — updated 2026-09-27, branch `audit/phase-1`.** Three tasks are done
and verified: **1** (metrics tracking), **2** (deterministic generation) and **32** (Gorillas
group key). That covers all of Phase 1 as originally written, plus the first blocking task of
Phase 1b; tasks 33–39 and everything from Phase 2 onward are untouched. **§0** records what was
verified and how, three claims in this audit that the work proved wrong, and four new findings
the work exposed. Each completed task's checkbox in the Plan carries its own note.

---

## 0. Implementation log

Updated **2026-09-27**. Commits on `audit/phase-1`, over baseline `5ce27f3` (the reorg commit
this audit was written against).

### Done and verified

**Task 1 — metrics tracking** (`846fed2`). The blanket `model_*.joblib` / `metrics_*.csv` globs
in `.gitignore` are replaced by a single scoped `experiments/**/model_*.joblib`. Verified:
`git ls-files experiments/` now returns **10** metrics files (was 0);
`git check-ignore` still catches `model_linear_regression.joblib` and no longer catches any
`metrics_*.csv`. **Risk #3 closed** — `dvc metrics diff` and `dvc exp show` can now compare
across commits.

**Task 2 — deterministic generation** (`6f8960d`). `sampling.py`, `outliers.py` and the
`build_row` callback contract all take an explicit `rng: random.Random`; `generate(n, seed, …)`
builds its own `random.Random(seed)` and threads it through. No module in the repo draws from
the global `random` singleton any more (grep-verified). The five rejection loops are capped at
`MAX_REJECTION_ATTEMPTS = 10_000` behind a `_bounded()` helper raising `UnreachableRangeError`.
Verified by running it: `generate(300, 42)` three times — once cleanly, once after
`random.seed(999)` plus 50 draws from the singleton, once mid-stream — gives three
`DataFrame.equals()`-identical frames, and seed 43 differs. `gauss(100.0, 1.0)` and
`gauss(1000.0, 1.0)` against `ELEVATION_RANGE` both raise `UnreachableRangeError` instead of
hanging. **Risk #4 closed** for the Python pool (the Gorillas pool is task 33, still open).
The commit also fixed a latent bug it surfaced: `gorillas.py` was calling `corrupt_row` without
an rng.

**Task 32 — Gorillas group key** (`eb839fd`). `generate.COLUMNS` gains a 14th column,
`group_id`, appended after `is_outlier` so every positional index (`MASS_COLUMN_INDEX`, the
`row[:11]` / `row[12]` slicing in both producers) is unaffected. The Python path assigns each
row its own id — correct, not a stub, since `sample_shot` draws every input independently;
verified `nunique() == len(df)`. The Gorillas path builds `f"{chunk_seed}_{board}"` in
`_to_contract`, and `_dump_raw` now emits `chunk_seed` and `group_id` columns in the raw log
too, recovered from a `CHUNKSEED.TXT` the worker writes. `_check_contract` asserts `group_id`
is present and non-empty and warns below 2 distinct groups. The commit also fixed a real
scoping bug on the way: the `ThreadPoolExecutor` futures list had lost its association with
`chunk_seeds`, so the completion loop could not tell which chunk a finished future's rows came
from — futures are now `(chunk_seed, future)` pairs.

The Python side of task 32 is run-verified; **the Gorillas side is code-verified only** —
DOSBox is Windows-only here (§5.3), so `generate_gorillas` cannot execute in a Linux checkout.
Re-verify on the Windows machine that every `group_id` in `data/gorillas_*.parquet` has a single
`wind_speed_ms` — the check that read "25 of 25 broken" in §5.1.

### Corrections to this audit's own claims

**Risk #2's "1 stage of 76" was already wrong when written.** `dvc.lock` records **4** stages:
`generate@standard_training_data`, both `generate_gorillas@*`, and `train_raw@linear_regression`
— all four locked by the 2026-09-27 Gorillas session, i.e. present in the baseline commit and
not produced by any task above. The substance of the risk stands: 4 of 76 is not a reproduced
pipeline, and everything in `experiments/` still predates DVC.

**`dvc.lock` does not match the committed tree, independently of any task here.** Its
`generate@standard_training_data` entry records `physics.py` at md5 `925f0308…`, size 1988;
the committed `physics.py` is `56b7376f…`, size 1935, and has not changed since the initial
commit. So the lock was written from a local tree that differed from what was pushed, and
`dvc status` reports that stage stale on a fresh clone — before considering code changes.
Task 12 should treat all four entries as untrustworthy rather than incrementally valid.

**`elevation_mean: 100.0` never hung the stage.** Task 2's stated evidence — "legal YAML,
unreachable against `ELEVATION_RANGE = (5.0, 85.0)` — hangs the stage forever" — does not
reproduce. Measured against the default `elevation_std: 20.0`, an accepted draw takes **4.7
attempts on average** (max 29 over 200 trials). The cap was still worth adding, but the failure
mode it guards is narrower than claimed, and the real hazard is the one in the new findings
below.

### New findings

**N1 — the rejection cap has a silent middle band** (new task **2b**). `_bounded` either
returns quickly or raises after 10,000 attempts; between those lies a wide band that does
neither. Measured against `ELEVATION_RANGE`:

```
gauss(45.0, 20.0)  →     1.1 attempts   (the configured default)
gauss(100.0, 20.0) →     4.7 attempts   — the config task 2 called "unreachable"
gauss(100.0,  5.0) →   694.4 attempts   — ~600x the sampling cost, completes silently
gauss(100.0,  2.0) → 10,000 attempts    — raises
```

At `(100.0, 20.0)` the accepted draws have mean **73.4°** and **30%** sit at ≥80°, i.e. piled
against the clip bound: the marginal is silently a truncated tail rather than the Gaussian the
config names, and nothing in the output says so. This is the §1.1 realism problem arriving
through the config rather than the code.

**N2 — `group_id` reaches the parquet but not the split** (folded into task **36**).
`loader.load_data` ends `X = df[FEATURES]; y = df[TARGET].values`, and `FEATURES` is the 9 raw
columns — so `group_id` is dropped before it reaches any caller. Task 32 made the key exist;
task 36 additionally needs the loader to return it (or return the frame and let the caller
project) — a scope the original task text omitted, now recorded there.

**N3 — chunk seeds can collide across gravity sessions** (folded into task **32**).
`gorillas.generate` draws chunk seeds for both plan entries — the normal session and the
out-of-distribution-gravity session — from the same `rng.randint(1, 30000)` stream. A repeat
within one session is benign (same seed ⇒ same boards ⇒ genuinely one group), but a repeat
*across* the two merges rows thrown under **different gravity** into one `group_id`. Roughly
0.6% at ~20 chunks total, and the fix is one string: include the plan tag or a chunk counter
in the key.

**N4 — task 32 made "13 columns" stale in the docs.** The contract is 14 columns now.
Corrected in this file; `data_generation/gorillas.py:3` still says "the same 13 columns" and
belongs in task 31's doc-drift list.

---

## Design decisions

The settled design these tasks implement.

### One dataset, engineered features derived on the fly

`generate` produces a single raw pool. The three engineered columns (`wind_x_ms`,
`drag_param`, `height_diff_m`) are derived at runtime inside each model's sklearn
`Pipeline` and never written to disk. Every run reads the same file; the raw-vs-engineered
contrast is only the Pipeline's output column list.

```
generate@standard_training_data          # 13 raw columns — the only training file
   ├─> train_raw@*                       #   output_columns = the 9 raw columns
   ├─> train_eng@*                       #   output_columns = the 5 engineered columns
   ├─> train_leakage_*
   ├─> train_bias_variance_*
   ├─> train_classification@*
   └─> filter_skewed                     # filters on launch_angle_deg, a raw column
         ├─> data/skewed_training_data.parquet   # what the skewed models train on
         └─> data/skewed_holdout.parquet         # the excluded rows = OOD test set
               └─> train_skewed@*
```

Keeping the transform in the Pipeline means the `.joblib` carries its own preprocessing and
can be fed raw physical inputs — no train/serve skew. Two distinct reasons a transform
belongs there, worth stating in `CLAUDE.md` because the repo currently blurs them:

1. **It fits parameters from data** (`StandardScaler`, `_clean_iqr`, any imputer or target
   encoder) — outside the Pipeline it *leaks*. Non-negotiable.
2. **It is stateless but must travel with the model** (`add_engineered_columns`) — no
   leakage either way, but keeping it inside means the artifact is self-contained.

Conflating these is why five runs drifted to load-time derivation unnoticed: for category 2
there is no correctness problem, only a deployment one.

### Skew by filter stage, not a second dataset

The skewed models train on a filtered subset of the same pool, produced by a
`filter_skewed` stage rather than a load-time filter or a separate draw.

- **A filter is a controlled experiment.** Same draw, same noise realisation, one variable
  changed. A second generated dataset with different `elevation_mean`/`elevation_std` would
  differ in the elevation distribution *and* every other sampled value *and* row count, so a
  worse skewed score could not be attributed to the distribution.
- **The filtered pool becomes a visible artifact** — openable, plottable, you can show the
  audience the hole. The DAG reads `generate → filter → train`, which is the lineage story a
  DVC lesson wants, and the bound lives in `params.yaml` so `dvc exp show` can sweep it.
- **The stage emits the complement too.** `skewed_holdout.parquet` is the OOD test set. The
  demo currently claims *"extrapolates poorly beyond 30°"* while no stage evaluates beyond
  30°, because the filter runs before the split so the test set carries the same hole.
- **Keep a one-sided cap, not a missing middle band.** Tree ensembles cannot extrapolate —
  beyond the training range a random forest flatlines at the boundary leaf mean while linear
  regression keeps going. That contrast is the lesson. A gap in the middle is much weaker:
  RF interpolates across it adequately and linear models are indifferent.

### Two producers, one contract, selected by name

`data_generation/generate.py` (RK4 in `physics.py`) and `data_generation/gorillas.py`
(the real 1990 game, driven through DOSBox) both emit the same 14 columns (13 until task 32
appended `group_id`), so either can fill the training slot. `params.yaml: training_data` names
which pool every consumer reads, resolved
by `params.data_path()`; the training stages take it as a `params:` entry and interpolate it
into `deps: data/${training_data}.parquet`.

This is the right shape and it is already adopted in 11 files. It makes the pool a single
setting and a compared column in `dvc exp show`, replacing a filename hardcoded across ~11
scripts and 10 stage `deps:`.

### No DVC remote — **for the Python-generated pool only**

The Python pool is 4.9 MB and regenerates in ~57 s; models regenerate from data + code + seed.
A remote exists to share large artifacts and back up what cannot be recreated — neither applied.
Local cache only, and that costs nothing in the experiment workflow: `dvc exp run` / `show` /
`diff` use local Git refs plus the local cache. Regeneration being the sole recovery path is
what makes deterministic seeding (task 2) critical rather than merely tidy.

**The Gorillas datasets break that reasoning and need a remote.** They are not reproducible
anywhere else: `find_dosbox` searches four hardcoded Windows paths, and DOSBox cannot be a DVC
dependency, so `generate_gorillas` fails on a fresh clone, on Linux/macOS, or on any machine
without DOSBox Staging. Their determinism is also timing-dependent (§5.2). If the local cache
is lost, `data/gorillas_*.parquet` is gone — there is no command that recreates it. Push at
least these four outputs somewhere (task 34).

---

## Critical risks

Status as of the 2026-09-27 update (§0).

| # | Risk | Evidence | Status |
|---|---|---|---|
| 1 | **19 of 76 stages cannot run.** All 17 `train_skewed@*` die on `ImportError: cannot import name 'FEATURE_STEP'`; `train_raw@decision_tree_overfit_no_outlier` and its `train_eng` twin write the wrong metrics filename, so DVC errors on a missing output. | reproduced | **open** |
| 2 | **The pipeline has never been reproduced.** [dvc.lock](dvc.lock) records **4** stages of 76 — corrected from "1" in §0; all four were locked before this audit's tasks began, and the lock does not even match the committed `physics.py`. Every model in `experiments/` was produced outside DVC and is unattributable. | [dvc.lock](dvc.lock) | **open** — gates phases 6–9 (task 12) |
| 3 | **Every metrics file is tracked by neither Git nor DVC.** `metrics_*.csv` is declared `cache: false` — "Git owns this" — but gitignored at [.gitignore:220](.gitignore#L220). `git ls-files experiments/` returns 0. This disables `dvc metrics diff` and `dvc exp show` across commits. | `git check-ignore` | **closed** — task 1, `846fed2`; 10 metrics files now tracked |
| 4 | **Generation is not deterministic, and is the only recovery path.** `generate()` does not self-seed: two seeded calls match, a third unseeded call diverges. With no remote, nothing ties the committed models to the pool that produced them. | [generate.py:64](data_generation/generate.py#L64) | **closed for the Python pool** — task 2, `6f8960d`, verified by re-running. Still open for the Gorillas pools (task 33) |
| 5 | **Test-set leakage in the shared regression loader.** Cleaning — including IQR quantiles over the full pool and target-based row filtering — runs *before* `train_test_split`. | [loader.py:86-98](models/regression/common/loader.py#L86) | **open** |
| 6 | **No tests, no type hints, no logging.** 82 function defs, 0 return annotations, 320 `print()`, 0 `import logging`, no pytest/ruff config, empty `[dev-packages]`. | measured | **open** — re-verified: no `pyproject.toml`, no `tests/`, `[dev-packages]` still empty |
| 7 | **No CLAUDE.md.** Nothing records the venv path, the DVC commands, that DOSBox is required for `generate_gorillas`, or which stages are known-broken. | absent | **open** — re-verified absent |

---

## 1. Data generation rigour & realism

### 1.1 Statistical fidelity

**All marginals are sampled independently — no copula, no correlation structure.** Maximum
off-diagonal correlation among the 8 inputs is **0.033**. Because mass and radius are drawn
independently, implied density is nonsense:

```
implied density (clean rows, kg/m³):
  min 13.3   p1 93.2   median 658.8   p99 13,582   max 290,957
  266 rows denser than osmium (22,590 kg/m³ — the densest element)
```

**No measurement noise on any column.** Every feature is the exact simulator input;
`landing_distance_m` is the exact RK4 output. Irreducible error is zero, so model ranking
reflects optimiser capacity only — not signal-vs-noise, which is the most important thing an
intro-ML audience needs to see.

**The classification task is artificially separable by construction.**
[generate.py:24-37](data_generation/generate.py#L24) picks the label *first*, then draws the
offset from a disjoint interval per class: hits `|offset| ≤ tolerance−0.1` (4.9 m), misses
`|offset| ∈ [tolerance+5, tolerance+80]` (≥ 10 m).

```
rows with |offset| in the 4.9–10 m gap:  3 / 50,000
```

A hard 5 m margin ⇒ Bayes error ≈ 0 ⇒ every classifier reports ~100% and the model
comparison is uninformative.

**Label balance is exactly 50/50 by parity**, not by sampling — `hit = (i % 2 == 0)`.
Harmless post-shuffle, but `stratify=y` is decorative and the demo can never show class
imbalance.

### 1.2 Data leakage

**Pre-split cleaning — the serious one.**
[loader.load_data](models/regression/common/loader.py#L69) cleans the whole pool and returns
`(X, y)`; each script then splits. Three distinct problems:

- `_clean_iqr` ([line 42](models/regression/common/loader.py#L42)) computes Q1/Q3 over
  **train + test combined** — a preprocessing decision fitted on held-out data.
- Both cleaners include `initial_velocity_ms` in their column loop, so rows are dropped
  **based on the target**. The test set loses exactly the rows the model scores worst on ⇒
  optimistically biased MAE.
- `_clean_no_outlier` ([line 65](models/regression/common/loader.py#L65)) filters on
  `is_outlier`, a **generation-time oracle column that cannot exist in real data**, yet it is
  presented alongside `range`/`iqr` as a comparable strategy.

`run_skewed`'s loader has the same shape
([train_utils.py:44-56](models/regression/run_skewed/train_utils.py#L44)). `_clean_iqr` is
dead code — no `dvc_models.yaml` entry ever sets `clean: "iqr"`.

**A silent simulator failure is mislabelled as clean data.**
[physics.simulate](physics.py#L39) returns `traj` unconditionally; if the landing condition
never fires within `max_time=60`, `traj[-1][1]` is wherever the projectile happened to be.
`sampling.py` takes that as `landing_x` with no did-it-land check.

```
clean rows (is_outlier == 'none') with negative landing_distance_m:  727 / 48,500
```

Worse than the *injected* `data_error` rows — and the two cleaners disagree about them:
`clean: "range"` drops all 727 (75% of the 964 rows it drops), while `clean: "no_outlier"`
— the oracle — keeps every one. The pathology demos currently teach the opposite of what
they claim.

**Label/feature contradiction after corruption.** `corrupt_row` runs *after* `hit_target` is
computed and can corrupt `landing_distance_m` / `target_distance_m`. Measured offsets: 0.20 m
on a labelled *miss*, 3,520 m on a labelled *hit*. Also present: `launch_angle_deg` to
1,478°, `initial_velocity_ms` from −76 to 981, 12 rows with negative mass, 55 rows with
`wind_direction_norm` outside {−1, 1}. With the default `clean: ""`, every baseline model
trains on these.

### 1.3 Determinism & reproducibility

- ~~**`generate()` does not seed itself.**~~ **Fixed — task 2 (`6f8960d`).** Was: only
  `__main__` ([generate.py:64](data_generation/generate.py#L64)) and `generate_all.py` called
  `random.seed()`, and a third unseeded call diverged. `generate(n, seed, …)` now builds its
  own `random.Random(seed)`. Re-verified by running three same-seed calls with the global
  singleton deliberately disturbed between them: all three identical.
- ~~**Seeding is global-module action-at-a-distance.**~~ **Fixed — task 2.** Every draw in
  `sampling.py`, `outliers.py` and the `build_row` callback now takes an explicit
  `rng: random.Random`; no module draws from the `random` singleton (grep-verified). This is
  also what unblocks task 24 (parallel generation). NumPy is still never seeded — harmless
  *today* only because generation remains stdlib-only.
- **Environment unpinned and untracked.** [Pipfile](Pipfile) pins every package to `"*"`, and
  no stage depends on `Pipfile.lock`. A `pipenv update` changes every model while
  `dvc status` reports clean. `params.py` is absent from all `deps` too.
- ~~**Unbounded rejection sampling.**~~ **Fixed — task 2**, with a caveat. The five loops are
  now one `_bounded()` helper capped at 10,000 attempts, raising `UnreachableRangeError` naming
  the column and range. Two corrections from re-measuring it (§0): `elevation_mean: 100.0`
  never hung — at the default `elevation_std: 20.0` an accepted draw takes 4.7 attempts — and
  the cap leaves a silent middle band where a rarely-reachable range completes with a distorted
  marginal instead of either running fast or failing. See finding **N1** and task **2b**.
- **Throughput: 875 rows/s, single-core, serial.** One `simulate()` per row in pure Python at
  `dt=0.02`; 50,000 rows ≈ 57 s. Embarrassingly parallel, but blocked by the global `random`
  singleton — so fixing seeding and fixing throughput are one task.

---

## 2. DVC pipelines & data versioning

1. **The metrics contract is self-defeating.** `cache: false` means Git owns the file;
   [.gitignore:219-220](.gitignore#L219) makes Git ignore `model_*.joblib` and
   `metrics_*.csv` globally. Every metric is unrecoverable from any commit or experiment ref.
2. **Outputs are gitignored by a hand-written glob.** DVC normally writes per-directory
   `.gitignore` files for cached outs; the root globs pre-empt that, so the only `.gitignore`
   files are `./.gitignore` and `./.dvc/.gitignore`. It works, but DVC's own bookkeeping
   never ran — consistent with the near-empty lock.
3. **`plots:` is used zero times, and 26 evaluation scripts sit outside the DAG.** All of
   `evaluation/` is invoked by hand, and `MODELS_TO_COMPARE` at
   [compare_models.py:11-24](evaluation/regression/compare_models.py#L11) is a
   hand-maintained, half-commented-out Python list. The comparison figures — the actual
   deliverable of the talk — are the least reproducible thing in the repo.
4. **Generation config is still mostly invisible to `dvc exp run`.** *Which* pool is read is
   now a proper param (`training_data`), but *how it was made* is not: `n`, `seed`,
   `elevation_mean`, `elevation_std`, `hit_tolerance`, and the Gorillas `input_mode` /
   `throws` / `boards_per_chunk` all live in [dvc_datasets.yaml](dvc_datasets.yaml), consumed
   via `vars:` and interpolated into `cmd:`. Hash-tracked, but not reachable by `--set-param`
   and not columns in `dvc exp show`. You cannot sweep seeds to measure run-to-run variance,
   or sweep `hit_tolerance` to vary task difficulty.
5. **Undeclared outputs on all six hand-written stages.** `train_skewed_concept`,
   `train_balanced_concept`, `train_leakage_clean`, `train_leakage_leaky`,
   `train_bias_variance_underfitting`, `train_bias_variance_overfitting` declare only their
   `.joblib`. Each also writes a `metrics_*.csv` that DVC neither tracks nor cleans and Git
   ignores. The leakage demo's headline number is not a tracked artifact of anything.
6. **Those same six declare no `params:`** and hardcode their sample sizes, so
   `dvc exp run --set-param n_samples=20000` silently leaves them at 10k/20k. A
   `dvc exp show` table then mixes responding and non-responding stages in one row.
7. **Asymmetric `foreach` matrix.** `regression_models` has `*_range` for 6 of 10 models (no
   `mlp_range`, no `decision_tree_overfit_range`) and `*_no_outlier` for 8. Nothing enforces
   or documents the intended matrix, so gaps read as deliberate.
8. **Stage-key ↔ script-name coupling is undeclared.** Classification's `model_path` and
   `print_metrics` ignore `TRAIN_MODEL_NAME` — which `dvc.yaml` dutifully passes — and derive
   the name from `sys.argv[0]`. It works *only* because every classification key equals its
   script's `train_` suffix. The first aliased key silently writes the wrong filename.
9. **~20% of generation is wasted.** The pool is 50,000 rows; `n_samples: 40000` caps
   consumption. At 875 rows/s that is ~11 s of the ~57 s producing rows nothing reads.
   [dvc.yaml](dvc.yaml)'s header also claims the pool is "40000 rows".

---

## 3. Code quality & MLOps hygiene

- **Every training script is a module-level side-effect script** — no `main()`, no functions,
  work at import. Nothing in `models/` or `evaluation/` is importable or unit-testable.
- **`save_metrics` / `print_metrics` / `model_path` / `load_data` are reimplemented 6×**
  across `common/loader.py` and four `run_*/train_utils.py`, with **four different
  signatures** (`save_metrics(dir, name, **kw)` vs `save_metrics(name, **kw)`) and three
  model-name resolution strategies (`TRAIN_MODEL_NAME`, `sys.argv[0]` basename, hardcoded
  literal). That inconsistency is the direct cause of risk #1.
- **`train_decision_tree_overfit.py` bypasses the loader.** It reads the parquet directly at
  [line 11](models/regression/train_decision_tree_overfit.py#L11), so it never reads
  `TRAIN_CLEAN` — `decision_tree_overfit` and `decision_tree_overfit_no_outlier` would
  produce byte-identical models from two separately-cached stages. It also hardcodes
  `save_metrics("decision_tree_overfit")` at
  [line 51](models/regression/train_decision_tree_overfit.py#L51), which is risk #1.
- **`train_balanced.py` duplicates the loader** — reads the parquet and slices
  `.iloc[:10_000]` rather than calling `load_data()`, and hardcodes `test_size=0.2` at
  [line 38](models/regression/run_skewed/train_balanced.py#L38) instead of reading
  `params.yaml`. The comparison is also not size-controlled: skewed trains on 8,525 rows
  (10,657 rows ≤ 30°), balanced on 8,000.
- **Sample size configured in four places:** `params.yaml n_samples: 40000`,
  `run_leakage N_SAMPLES = 10_000`, `run_bias_variance N_SAMPLES = 20_000`,
  `train_balanced.py N_SAMPLES = 10_000`.
- **No validation anywhere.** Nothing asserts the 13 expected columns, dtypes or physical
  bounds. `add_engineered_columns` divides by `mass_kg` relying on an invariant enforced by a
  magic constant (`MASS_COLUMN_INDEX = 4`) in a different module, verified by nothing.
- **Doc drift.** `dvc.yaml` says "40000 rows" vs `n: 50000`. Live docstrings still reference
  deleted paths — `run_10k`, `run_raw_10k`, `train_all.py`, `generate_data.py` — in
  [classification/common/loader.py:1](models/classification/common/loader.py#L1),
  [regression/common/loader.py:1](models/regression/common/loader.py#L1), and 6 files under
  `evaluation/`.
- **Repo hygiene.** `qbasic_gorillas/dosbox-datagen/` is now a pipeline dependency and stays
  (§5). The other five sibling builds — `dosbox`, `dosbox-modified`,
  `dosbox-modified-physics`, `dosbox-modified-physics-metrics`,
  `dosbox-modified-physics-metrics-var` — are historical and carry a `QBASIC.EXE` each.
  `machine learning 101.docx` (186 KB) and `ml_101_rev2.pptx` (1.27 MB) are committed
  binaries; `claude sessions/` has a space in the path; `README.md` reads
  `need to add some stuff here`.

---

## 4. Unnecessary complexity (teaching-clarity lens)

Target: **a reader should be able to answer "where does my training data come from?" by
reading one file.** Today it takes four. Ranked by cost to a reader, not lines saved.

### C1 — PYTHONPATH dependency injection (worst offender)

A reader opens [train_random_forest.py](models/regression/train_random_forest.py#L5) and sees
`from train_utils import load_data, FEATURE_STEP`. **There are six different `train_utils.py`
files and nothing in the file says which one this is.**

```
dvc.yaml  cmd: python scripts/run_with_pythonpath.py models/regression/run_raw  models/regression/train_random_forest.py  TRAIN_MODEL_NAME=…  TRAIN_CLEAN=…
   └─> run_with_pythonpath.py sets PYTHONPATH=<run_dir>        (35 LOC of wrapper)
         └─> child's sys.path[0] = models/regression/ ; falls through to PYTHONPATH
               └─> resolves to models/regression/run_raw/train_utils.py
                     └─> which does sys.path.insert(0, "../../..") to find `params`
                           └─> and reads TRAIN_CLEAN out of os.environ at load time
```

Four levels of indirection, two invisible from any source file, plus behaviour smuggled in
through environment variables. Also fragile: adding any `models/regression/train_utils.py`
would silently shadow all six.

### C2 — 44 regression stages for a talk

`regression_models` holds 22 entries, expanded across `train_raw` and `train_eng` = **44
stages**, covering `polynomial_range`, `knn_no_outlier`, `xgboost_range` and similar. No talk
shows 44 models, and the asymmetry (§2.7) means a reader cannot infer the rule.

### C3 — Two ways to engineer features, and the intended one is the minority

| API | Files | Derives features… |
|---|---|---|
| `add_engineered_columns` (plain function) | **16** | at load time, *outside* the Pipeline |
| `EngineeredFeatures` (sklearn transformer) | **3** | on the fly, *inside* the Pipeline |

Five runs — `run_skewed`, `run_leakage`, `run_bias_variance`, `classification`,
`train_decision_tree_overfit` — derive at load time. Only `run_eng` does it on the fly. Two
APIs for one job means a reader must learn both and work out when each applies.

### C4 — `FEATURE_STEP = ("engineer", "passthrough")`

A do-nothing sklearn step ([run_raw/train_utils.py:20](models/regression/run_raw/train_utils.py#L20))
whose only purpose is tuple-shape parity with the engineered run. A reader will lose minutes
deciding whether it matters.

### C5 — `evaluation/` is the biggest thing in the repo and the least governed

**2,541 LOC** — larger than `models/` (1,402) and `data_generation/` (316) combined — across
8 directories, entirely outside the DAG, with near-duplication that has already drifted (all
copies differ by md5): `predict.py` × 6, `evaluate_features.py` × 6,
`plot_training_data_relationships.py` × 5.

### C6 — 5 config files and 332 lines to configure one dataset

`dvc.yaml` (202) + `dvc_models.yaml` (50) + `params.yaml` (41) + `params.py` (21) +
`dvc_datasets.yaml` (18), plus `vars:` / `foreach:` / `${item.script}` templating.
[dvc_datasets.yaml](dvc_datasets.yaml) contains **one dataset entry and 12 lines of comment
explaining how to add more**. The templating is the specific clarity problem: someone
learning DVC here reads `foreach: ${regression_models}` with `${item.clean}` interpolated
into `cmd:` and learns none of what a stage actually is.

### C7 — Callback injection and a magic column index in generation

[generate.py](data_generation/generate.py#L24) defines `_make_build_row(hit_tolerance)`, a
closure factory returning a `build_row(shot, i) -> (values, labels)` callback, passed into
`generate_rows`, which calls it and concatenates results positionally. **One implementation,
one call site.** Worse, `corrupt_row` is told which column not to zero by
`no_zero_indices={MASS_COLUMN_INDEX}` where `MASS_COLUMN_INDEX = 4` is a positional index
into a list built in a different function, coupled to `COLUMNS` order by nothing but a
comment.

### C8 — Orphaned and unreachable files

| File | Status |
|---|---|
| [train_linear_regression_sqrt.py](models/regression/train_linear_regression_sqrt.py) | **0 references** in `dvc_models.yaml` — unreachable by DVC |
| [simulate.py](simulate.py) (root) | nothing imports it |
| `flowchart TD.mmd` | stale, unreferenced |
| `qbasic_gorillas/dosbox{,-modified,-modified-physics,-modified-physics-metrics,-modified-physics-metrics-var}/` | superseded by `dosbox-datagen/`, which the pipeline depends on; one `QBASIC.EXE` each |

**⚠️ `train_linear_regression_sqrt.py` contains a bug rationalised as physics.** Its comment
reads *"landing_distance_m can be negative (e.g. a strong headwind can land a shot behind the
launch point) — preserve sign and magnitude"*. Those negatives are **not** headwind: they are
the 727 non-terminating trajectories from the missing did-it-land check (§1.2). A feature
transform was designed around a simulator defect.

### C9 — Search budget is expensive and has a special-case convention

`RandomizedSearchCV(n_iter=30, cv=5)` = 150 fits per model, on 40,000 rows, across dozens of
stages. `params.yaml` also uses `null` to mean "this algorithm has no such knob", a convention
the reader must infer.

### C10 — `experiments/` collides with `dvc exp`

The directory holds model outputs, not DVC experiments — confusing in a repo whose purpose
includes teaching `dvc exp run`.

---

## 5. QBasic Gorillas — the second data producer

[data_generation/gorillas.py](data_generation/gorillas.py) (572 LOC) drives
`qbasic_gorillas/dosbox-datagen/` — a modified build of the 1990 game that plays itself and
logs every throw — and maps the result onto `generate.COLUMNS`. Two datasets at n=5000,
`gorillas_effort_data` and `gorillas_velocity_data`, differing only in `input_mode`.

**This is good work, and it changes priorities elsewhere in this audit.** The game pool has
exactly the structure §1.1 says the Python generator lacks: the banana stops at the first
non-background pixel, so buildings truncate flights and `landing_distance_m` stops being a
function of the logged columns; wind is a per-board integer shared across a board's throws;
the hit rate is whatever the game produces rather than designed. Measured against the Python
pool:

| | Python pool | Gorillas (EFFORT) |
|---|---|---|
| `corr(initial_velocity_ms, landing_distance_m)` | 0.471 | **0.147** |
| hit rate | exactly 0.500 (by parity) | **0.071** |
| `wind_speed_ms` | continuous Weibull | **15 integer levels, 0–14** |
| `drag_coeff` (clean rows) | Gaussian, truncated | **constant 0.600** |

The weaker correlation is the *point* — it is an unobserved confounder doing real work — but
the demo has to expect and explain it rather than read as a regression.

Three things are right and worth keeping: the `workers` (performance) vs `boards_per_chunk`
(dataset identity) distinction, with chunk seeds drawn single-threaded before the pool so
`--workers` cannot alter the RNG stream; `DESTRUCT=0`, so craters cannot make throw N depend
on throws 1..N−1 through a variable in no column; and `_check_contract`, which is the schema
assertion task 28 asks for, already written with the right "clean rows only" nuance.

### 5.1 Group leakage — the key now exists (task 32), the split still ignores it

Wind and the skyline are **per board**, shared by all 32 throws on it. A random
`train_test_split` therefore puts throws from the same board on both sides, letting the model
memorise board-specific structure — textbook group leakage, and a much bigger effect here
than anything in §1.2.

**The key now exists — task 32 (`eb839fd`).** As audited, `_to_contract` emitted
`generate.COLUMNS` only, so the parquet had no `board` column, and the raw log could not rescue
it either: `_dump_raw` concatenated every worker's CSV without a worker or chunk identifier, and
board numbering restarts per session. Measured then on `gorillas_effort_data_throws.csv`:

```
8,000 rows · 25 distinct board values · 320 throws per "board"
boards whose wind_ms is NOT single-valued: 25 of 25
```

`generate.COLUMNS` now carries a 14th column, `group_id`, set to `f"{chunk_seed}_{board}"` by
`_to_contract`; `_dump_raw` writes `chunk_seed` and `group_id` into the raw log as well, read
back from a `CHUNKSEED.TXT` each worker writes. Code-verified only — DOSBox is Windows-only
(§5.3), so re-run the 25-of-25 wind check on the Windows machine to confirm it now reads 0.

**Two things remain before the leakage itself is fixed:**

- **The split still cannot see the key** (finding N2). `loader.load_data` returns
  `X = df[FEATURES]`, and `FEATURES` is the 9 raw columns, so `group_id` is dropped before any
  caller can group on it. Task 36 needs the loader changed as well as the splitter.
- **Chunk seeds can collide across the two gravity sessions** (finding N3), merging rows thrown
  under different gravity into one `group_id`. ~0.6% at ~20 chunks; fixed by putting the plan
  tag or a chunk counter in the key.

### 5.2 Determinism is timing-dependent — **still open (task 33)**

Re-verified unchanged after task 2: `keep = len(got) / thrown` still feeds `need` → `n_chunks`
→ how many chunk seeds get drawn. Task 2 fixed the Python producer's determinism; this hole is
in the Gorillas producer and is untouched.

`_run_worker` catches `subprocess.TimeoutExpired` and proceeds with whatever partial CSV
exists. The retry loop then computes `keep = len(got) / thrown` from rows actually returned,
and `keep` feeds `need` → `n_chunks` → **how many chunk seeds get drawn**. So a slow or loaded
machine takes a different path through the RNG stream and produces a different dataset for the
same `--seed`. The module is scrupulous about `--workers` not affecting reproducibility and
documents it well; a timeout defeats that guarantee through a different door.

### 5.3 Not reproducible off this machine

`find_dosbox` searches four hardcoded Windows paths and DOSBox cannot be a DVC dep, so
`generate_gorillas` fails on a fresh clone, on Linux/macOS, or without DOSBox Staging
installed. Combined with the no-remote decision, these four outputs have no recovery path.

### 5.4 Severe class imbalance, unhandled

Hit rate is **7.1%** (EFFORT) and **4.9%** (VELOCITY), against the Python path's exact 50/50.
Predicting all-miss scores 93–95% accuracy, and `print_metrics` prints accuracy first. No
`class_weight` or `scale_pos_weight` appears anywhere. This is excellent teaching material —
it is the canonical "accuracy is the wrong metric" lesson, arriving for free — but as wired
today it will simply look like the models work.

### 5.5 `n_samples` silently truncates

`params.yaml` has `n_samples: 40000`; the Gorillas pools hold 5,000 rows. Verified:
`df.iloc[:40000]` returns 5,000 with no warning. Switching `training_data` to a Gorillas pool
silently changes sample size by 8× and flatlines the convergence experiment with no signal
that anything happened.

### 5.7 Sample-size tiers: right design, half-finished execution

One pool read at several sizes via `params.yaml: n_samples` is the correct shape and is what
`run_raw` / `run_eng` / `classification` already do (`loader.load_data` → `df.iloc[:n_samples]`
on the pre-shuffled pool, so each tier is a nested prefix of one draw). Three things stop it
being true across the repo:

**The old per-size directories were never deleted.** 18 directories, 30 files, referenced by
`dvc.yaml` zero times, still tracked in Git:

```
models/{regression/run_raw_10k,run_raw_20k,run_raw_40k,
        regression/run_eng_10k,run_eng_20k,run_eng_40k,
        classification/run_10k,run_20k,run_40k}
evaluation/…  the same nine again
```

**Three runs ignore `n_samples` and use a different sampling method.** `run_leakage`
(`N_SAMPLES = 10_000`), `run_bias_variance` (`20_000`) and `train_balanced.py` (`10_000`)
hardcode their size *and* draw with `df.sample(n=…, random_state=seed)` rather than
`.iloc[:n]`. A random sample is not a nested prefix, so those runs are not comparable with the
tiers and do not respond to `--set-param n_samples=…`. Two evaluation plot scripts also
hardcode `N_SAMPLES = 20_000` / `10_000` with the comment *"matches train_utils.py"* — a manual
sync guaranteed to drift.

**`n_samples` is never validated against pool size**, which is §5.5: 40,000 against a
5,000-row Gorillas pool silently yields 5,000. For a design whose whole point is "same data,
different sampling value", the knob must fail loudly when it cannot be honoured.

### 5.6 Smaller items

- **`drag_coeff` is constant at 0.600 on clean rows** — a zero-variance feature carried into
  every model. `_clean_iqr` skips it correctly (`iqr == 0`), but it is dead weight and will
  look broken in feature-importance plots. `drag_param` still varies through radius and mass.
- **The raw throw logs are uncompressed CSV** — 1.9 MB and 2.4 MB, cached by DVC, and now two
  columns wider after task 32. Parquet would be several times smaller and `pyarrow` is already
  a dependency.
- **`landing_height_m` is silently clipped at 0** when `land_dy_m` puts it below ground
  ([gorillas.py:286](data_generation/gorillas.py#L286)). Currently 0 rows are affected, and
  the code warns above 3 m — but the clip is a data modification that no column records.

---

## 6. Project architecture (CLAUDE.md check)

**There is no `CLAUDE.md`.** `.claude/settings.json` holds two permission entries and nothing
else. Measured cost during this audit: the shell picked up an unrelated project's virtualenv
(`V-and-V-13OSdAjM`, no `pyarrow`) before the correct one
(`talk_Introduction_to_ml-9WAJNeEj`) was located. A developer or agent landing here cannot
know the venv path, that `pipenv` needs `PIPENV_IGNORE_VIRTUALENVS=1` here, that there are no
tests, which stages are broken, or that `generate_gorillas` needs DOSBox Staging installed.

[claude sessions/2026-09-15-ml-pipeline-refactor.md](claude%20sessions/2026-09-15-ml-pipeline-refactor.md)
already holds most of the needed rationale — right content, in a file nothing reads by
default. `CLAUDE.md` should carry the durable subset: environment activation, the exact
`dvc repro` / `dvc exp run --set-param` / `dvc exp show` invocations, the run-config contract,
the two Pipeline rules from the design decisions, and the fresh-clone bootstrap (no remote,
so `dvc pull` does nothing — a new clone runs `dvc repro generate`).

---

## Plan

Tasks are numbered in execution order. Two principles drive the order: **delete before you
repair** (don't unbreak stages you are about to cut, or fix files you are about to remove),
and **change the data before you invest in models** (task 2 alters the draw, so anything
trained before it is thrown away).

Checkboxes are live: `[x]` means done and verified (see §0 for how), `[ ]` means untouched.
**Done: 1, 2, 32.** Everything else below is open.

### Phase 1 — Make results recordable and the data recoverable — **1 and 2 done**

(Task **2b** below is a new, non-blocking follow-up that task 2's own verification exposed; it
does not hold up Phase 1b.)

- [x] **1.** ~~Fix metrics tracking.~~ **Done — `846fed2`.** Replace the blanket
      `model_*.joblib` / `metrics_*.csv` globs in `.gitignore` with `experiments/**/model_*.joblib`
      only, so the `cache: false` metrics are versioned by Git as intended. Verified:
      `git ls-files experiments/` returns 10 metrics files (was 0), the `.joblib` outputs are
      still ignored, and nothing else was caught by accident. Closes risk #3.
- [x] **2.** ~~Make generation deterministic.~~ **Done — `6f8960d`.** An explicit
      `rng: random.Random` is threaded through `sample_shot` / `generate_rows` / `corrupt_row`;
      `generate(n, seed, ...)` seeds its own generator; every rejection loop is capped at 10,000
      attempts behind `_bounded()`, raising `UnreachableRangeError`. Verified by running it —
      three same-seed calls with the global singleton disturbed between them are identical, and
      a genuinely unreachable range raises instead of hanging. Closes risk #4 for the Python
      pool; the Gorillas pools are still task 33. Also unblocks task 24.
- [ ] **2b.** Make a *rarely*-reachable range as loud as an unreachable one (finding N1, new).
      `_bounded` returns fast or raises at 10,000 attempts, and between those sits a band that
      does neither: `gauss(100.0, 5.0)` against `ELEVATION_RANGE` averages 694 attempts per draw
      (~600× the cost) and completes, and `gauss(100.0, 20.0)` completes in 4.7 with accepted
      draws piled against the bound (mean 73.4°, 30% at ≥80°) — a silently truncated marginal,
      not the Gaussian the config names. Warn when the observed acceptance rate falls below some
      threshold, and record the realised distribution rather than only the configured one.
      Cheap, and it is §1.1's realism problem arriving through the config instead of the code.

### Phase 1b — Make the Gorillas producer trustworthy (§5)

Numbered 32–39 to keep tasks 1–31 stable; they belong here, not at the end. Tasks 32–35 are
blocking-class: each one can silently produce wrong results or unrecoverable data.

- [x] **32.** ~~Emit a real group key.~~ **Done — `eb839fd`.** `generate.COLUMNS` gains a 14th
      column, `group_id`, appended after `is_outlier` so no positional index moves;
      `_to_contract` sets `f"{chunk_seed}_{board}"`, `_dump_raw` writes `chunk_seed` and
      `group_id` into the raw log, and `_check_contract` asserts the column is present and
      non-empty. The Python path gives every row its own id, which is correct rather than a stub
      because `sample_shot` draws independently per row. Python side run-verified; **Gorillas
      side code-verified only** — DOSBox is Windows-only (§5.3), so re-run the 25-of-25
      single-valued-`wind_ms` check there. Two follow-ups this left open:
      - Finding **N3**: chunk seeds for the normal and gravity sessions come from one
        `rng.randint(1, 30000)` stream, so a collision across the two merges rows thrown under
        different gravity into one `group_id` (~0.6% at ~20 chunks). Put the plan tag or a chunk
        counter in the key.
      - Finding **N4**: `data_generation/gorillas.py:3` still says "the same 13 columns" — add it
        to task 31's doc-drift list.
- [ ] **33.** Close the timing-dependent determinism hole (§5.2). **Now the only open
      blocking-class determinism gap** — task 2 closed the Python producer's, not this one. A
      worker timeout changes the observed keep rate, which changes how many chunk seeds get
      drawn, which changes the dataset for a fixed `--seed`. Either fail the run on timeout
      rather than accepting partial output, or decouple the chunk plan from observed yield —
      draw a fixed chunk schedule from `seed` and over-generate to a fixed margin instead of
      adapting.
- [ ] **34.** Add a DVC remote and push the four Gorillas outputs. They cannot be regenerated
      off this machine (§5.3), so the local cache is currently their only copy. This narrows
      the no-remote decision rather than reversing it: the Python pool still needs no remote.
- [ ] **35.** Make `n_samples` fail loudly when it exceeds the pool (§5.5). `df.iloc[:40000]`
      on 5,000 rows silently returns 5,000, so switching `training_data` to a Gorillas pool
      changes the sample size 8× with no signal.
- [ ] **36.** Use a group-aware split wherever the active pool has groups — `GroupShuffleSplit`
      / `GroupKFold` on the task-32 key instead of `train_test_split`, and `cv=GroupKFold` in
      the searches. **Task 32 supplied the key; this task also has to get it to the splitter**
      (finding N2): `loader.load_data` returns `X = df[FEATURES]` over the 9 raw columns, so
      `group_id` is dropped before any caller sees it. Either return groups as a third value or
      return the frame and let the caller project. The Python pool's ids are unique per row, so
      a group-aware split degrades to an ordinary random split there — correct, and it means the
      same code path works for both producers. This is the largest leakage in the repo when a
      Gorillas pool is active, and strictly larger than anything in §1.2.
- [ ] **37.** Handle the 5–7% hit rate (§5.4). Add `class_weight="balanced"` /
      `scale_pos_weight`, and lead the classification metrics with precision/recall/PR-AUC
      rather than accuracy. Worth building the talk around: predicting all-miss scores 93–95%,
      which is the canonical "accuracy is the wrong metric" lesson arriving for free.
- [ ] **38.** Wire `_check_contract` into the pipeline. It is already written, with the right
      "clean rows only" nuance, but only runs under `__main__` in `gorillas.py`. Move it beside
      the loader so **both** producers are validated on read — this is most of task 28 already
      done.
- [ ] **39.** Smaller items from §5.6: drop or document the constant `drag_coeff` (zero variance
      on clean rows, dead weight in every model); write the raw throw logs as parquet rather
      than 1.9 MB + 2.4 MB of CSV; record the `landing_height_m` clip in a column instead of
      only a stderr warning.

### Phase 2 — Delete

- [ ] **3.** Cut `regression_models` from 22 entries to the ~4 models that will appear on a
      slide, plus one `_no_outlier` variant to make the cleaning point. Drop `*_range`
      entirely or keep exactly one. Apply the same cut to `skewed_models` and
      `classification_models`. 44 regression stages → ~10.
- [ ] **4.** Delete orphans: `train_linear_regression_sqrt.py` (unreachable, and its
      sign-preserving-sqrt rationale is built on the task-13 bug), `simulate.py`,
      `flowchart TD.mmd`, and any `train_*.py` left unreferenced by task 3.
- [ ] **4b.** Delete the 18 stale size-tier directories (§5.7) — `run_raw_10k/20k/40k`,
      `run_eng_10k/20k/40k`, `classification/run_10k/20k/40k`, under both `models/` and
      `evaluation/`. 30 files, referenced by `dvc.yaml` zero times, superseded by
      `params.yaml: n_samples`. Then make the three runs that still hardcode their own size
      read `n_samples` and slice with `.iloc[:n]` rather than `df.sample(n=…)`, so every tier
      is a nested prefix of one draw and responds to
      `dvc exp run --set-param n_samples=…`: `run_leakage` (10,000), `run_bias_variance`
      (20,000), `train_balanced.py` (10,000), plus the two evaluation plot scripts that
      hardcode `N_SAMPLES` with a *"matches train_utils.py"* comment.
- [ ] **4c.** Delete the five superseded DOSBox builds — `dosbox`, `dosbox-modified`,
      `dosbox-modified-physics`, `dosbox-modified-physics-metrics`,
      `dosbox-modified-physics-metrics-var`. Keep `dosbox-datagen/`, which `generate_gorillas`
      depends on.

### Phase 3 — Collapse the indirection

- [ ] **5.** Replace the PYTHONPATH injection with one `train.py` taking explicit arguments —
      `python -m models.regression.train --run raw --model random_forest --clean none` —
      with each run's config in a plain dict. Deletes
      [scripts/run_with_pythonpath.py](scripts/run_with_pythonpath.py), all six
      `run_*/train_utils.py`, the `TRAIN_MODEL_NAME` / `TRAIN_CLEAN` env vars and every
      `sys.path.insert`. Safe to attempt now: `train_raw@*` and `train_classification@*` work
      today and serve as the reference to validate against.
- [ ] **6.** Fold `train_decision_tree_overfit.py` into that contract: route it through the
      shared loader so `--clean` applies, and let the run config name its output instead of
      hardcoding `save_metrics("decision_tree_overfit")`. **Fixes the 2 broken
      `*_overfit_no_outlier` stages and the duplicate-model problem.**
- [ ] **7.** One feature-engineering API. Keep `EngineeredFeatures`; make
      `add_engineered_columns` private to it. Convert the five load-time call sites —
      `run_skewed`, `run_leakage`, `run_bias_variance`, `classification`,
      `train_decision_tree_overfit` — to the Pipeline step. Then make the step uniform:

      ```python
      FEATURE_STEP = ("features", EngineeredFeatures(output_columns=FEATURES))
      ```

      with `run_raw` passing the 9 raw columns instead of the `"passthrough"` sentinel.
      `transform` is already `add_engineered_columns(X)[output_columns]`, so the raw list
      computes three columns and discards them — free, and it replaces an opaque sentinel with
      one step type and one config difference.

### Phase 4 — Restructure the skew demo

- [ ] **8.** Add a `filter_skewed` stage: reads `data/standard_training_data.parquet`, takes
      `skew.max_angle_deg` from `params.yaml`, writes **two** outs —
      `data/skewed_training_data.parquet` (kept rows) and `data/skewed_holdout.parquet`
      (excluded rows). Point the skewed run's config at the filtered file. **This is what
      unbreaks the 17 `train_skewed@*` stages** — the filter leaving the loader is the fix.
      Declare the `metrics_*.csv` these stages write; they are currently undeclared.
- [ ] **9.** Delete `MAX_ELEVATION_DEG` from
      [train_utils.py:18](models/regression/run_skewed/train_utils.py#L18) — the bound now
      lives in `params.yaml`, so it is sweepable and appears in `dvc exp show`. Pool coverage
      for choosing it: 5–30° = 10,624 rows, 30–50° = 19,520, 50–85° = 19,786.
- [ ] **10.** Report MAE on **both** test sets — the in-distribution split and
      `skewed_holdout.parquet`. Without this the demo asserts something it never measures
      ([train_skewed.py:30](models/regression/run_skewed/train_skewed.py#L30)).
- [ ] **11.** Fix `train_balanced.py`: use the shared loader instead of re-reading the
      parquet, read `test_size` from `params.yaml` rather than hardcoding `0.2`
      ([line 38](models/regression/run_skewed/train_balanced.py#L38)), and size-control the
      comparison against the skewed run (8,525 vs 8,000 rows) or document the difference.

### Phase 5 — Reproduce — milestone

- [ ] **12.** `dvc repro`, then commit `dvc.lock` so it covers the whole DAG. **Nothing below
      is measurable before this point.** Verify `dvc exp show` produces a populated table and
      `dvc metrics diff` works against HEAD.

### Phase 6 — Correctness

- [ ] **13.** Add a `did_land` flag to [physics.simulate](physics.py#L39); drop or mark
      non-terminating trajectories in `sampling.py`. Reclassifies the 727 mislabelled-clean
      rows and makes the cleaning demos truthful.
- [ ] **14.** Move cleaning behind the split. Convert `_clean_iqr` / `_clean_range` into
      sklearn transformers so their statistics fit on training folds only; have the loader
      return the raw frame and let the caller split first. Stop filtering on the target.
- [ ] **15.** Retire `_clean_no_outlier` or rename it honestly — it is an oracle using a
      column real data cannot have. Delete the dead `iqr` branch.
- [ ] **16.** Add `Pipfile.lock` and `params.py` to every stage's `deps`, so a dependency
      upgrade or a config-loader change invalidates the models it affects.

### Phase 7 — Remaining simplification

- [ ] **17.** Collapse `evaluation/` from 2,541 LOC to one `predict.py`, one
      `evaluate_features.py` and one plotting script, each taking `--run` / `--model`. Do this
      before task 27, so you wire up 3 stages rather than 17.
- [ ] **18.** Move `n`, `seed`, `elevation_mean`, `elevation_std`, `hit_tolerance` out of
      `dvc_datasets.yaml` into `params.yaml` under a `generate:` block, declared in the
      stage's `params:`. Then `dvc exp run --set-param generate.seed=7` is a real variance
      experiment. Shrink `dvc_models.yaml` to whatever survived task 3, and consider writing
      the surviving stages out explicitly instead of `foreach`.
- [ ] **19.** Build generated rows as named dicts and corrupt by column *name*. Removes
      `_make_build_row`, the `build_row` callback, `no_zero_indices` and `MASS_COLUMN_INDEX`,
      and makes the mass-never-zero invariant self-evident instead of a comment.
- [ ] **20.** Reduce the hyperparameter search to a small explicit grid, except on the one or
      two models where tuning is the lesson. Rename `experiments/` → `artifacts/`.

### Phase 8 — Data realism

- [ ] **21.** Derive `hit_target` from a continuous offset instead of choosing the label
      first. Removes the 5 m margin and gives classification a non-zero Bayes error.
- [ ] **22.** Sample `radius` and `density` independently and derive
      `mass = density · (4/3)πr³`. Three lines, no new dependency, and it kills the
      290,957 kg/m³ tail while giving `drag_param` real structure. (A Gaussian copula would be
      more faithful but adds `scipy` and a concept the audience never sees.)
- [ ] **23.** Add per-column measurement noise to the recorded features and to
      `landing_distance_m`, with the scale as a `params.yaml` knob. This is *simplifying*: it
      removes the need to explain why every model scores suspiciously well.
- [ ] **24.** Parallelise generation with `multiprocessing` over seeded per-worker RNGs
      (unblocked by task 2). 875 rows/s → ~8× on a typical laptop.
- [ ] **25.** Align pool size with consumption — either `n: 40000` or `n_samples: 50000` — and
      fix the "40000 rows" comment in `dvc.yaml`.

### Phase 9 — Hygiene

- [ ] **26.** Write `CLAUDE.md` (contents listed in §6).
- [ ] **27.** Bring `evaluation/` into the DAG with `plots:` declarations; replace
      `MODELS_TO_COMPARE` with a `dvc_comparisons.yaml` read via `vars:`.
- [ ] **28.** Add `pytest` + `ruff` to `[dev-packages]` and a `pyproject.toml`. Two tests,
      each guarding a bug found above: generation determinism under a fixed seed, and an
      `assert_schema(df)` check on column set, dtypes and physical bounds. Plain asserts, not
      a schema DSL.
- [ ] **29.** Refactor training scripts to `main()` + `if __name__ == "__main__"` so they are
      importable. Keep `print()` — the audience watches console output; the duplication behind
      the 320 calls is resolved by tasks 5 and 17, not by a logging framework.
- [ ] **30.** Purge unrelated content: move the `.docx`/`.pptx` out of Git, rename
      `claude sessions/` → `claude_sessions/`, write a real `README.md`. **`qbasic_gorillas/`
      stays** — it is a pipeline dependency now, not an unrelated project. Only its five
      superseded builds go, in task 4c.
- [ ] **31.** Fix doc drift: the `run_10k` / `run_raw_10k` / `train_all.py` /
      `generate_data.py` references in 8 live docstrings, and the intended `foreach` matrix
      after task 3.

---

Phases 1 through 5 — tasks 1, 2, 32–39, 3, 4, 4b, 4c, then 5–12 in order — take the repo from
"the pipeline is aspirational" to "`dvc repro` reproduces the DAG and `dvc exp show` compares
it." Tasks 5 and 36 are the only substantial refactors in that stretch; the rest are small or
pure deletion. Nothing in phases 6–9 is worth starting before task 12, because until then there
is no reliable way to observe whether a change helped.

**Where that leaves the run as of 2026-09-27:** tasks 1, 2 and 32 are done, so results are
recordable, the Python pool is genuinely regenerable, and the Gorillas group key exists. Next in
order are **33** (the last blocking-class determinism gap) and **34–35**, then the Phase 2
deletions before any of the Phase 3 refactors. Note that task 2 changed the draw exactly as this
plan predicted: all four `dvc.lock` entries are now stale — every generation dep's md5 differs
from what the lock records — so the pre-existing models are throwaway and task 12 starts from
scratch rather than topping up a partial lock.

If the talk is going to use a Gorillas pool at all, tasks 32 and 36 come before any model
result is worth quoting: with 32 throws sharing a board's wind and skyline, a random split
leaks across groups, and the grouping key is currently discarded before the data reaches the
pipeline.
