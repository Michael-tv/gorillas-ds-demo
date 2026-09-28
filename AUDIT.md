# Repository audit — shortcomings & refactor plan

Audit date: 2026-09-27 · Audited at `5ce27f3` · DVC 3.67.1 · **93 stages across 7 `dvc.yaml`
files** (root + six under `pipelines/`) as of 2026-09-28 — see below; 76 at audit time.
Last updated: 2026-09-28, on `main` (§0)

**Rescanned 2026-09-27 after the QBasic Gorillas integration**, and again 2026-09-28 after the
repo owner's own restructuring pass (§0's tasks 5/6/7 entry, and the two new design decisions on
`pipelines/` and the effort/velocity split). Three earlier recommendations were wrong and are
corrected: purging `qbasic_gorillas/` (§3, C8, task 30), the no-remote decision, and — as of
2026-09-28 — C1/C3/C4's "still open" status.

**Health score: 4 / 10 at audit time; 6 / 10 as of 2026-09-28.** The architecture is
well-conceived and, as of this update, no longer just aspirational in its own code organisation:
the worst clarity offender this audit found (C1) is fixed, verified with real `dvc repro` runs,
not just parsed. What still gates the score: risk #2 (the pipeline has never been fully
reproduced — `dvc.lock` coverage is partial, not zero, but nowhere near complete) and risk #5
(pre-split cleaning still leaks in the regression loader, task 14, untouched).

**Implementation status — updated 2026-09-28.** Done and verified: **1** (metrics tracking),
**2** + **2b** (deterministic generation, rare-range warning), **3** (the model matrix cut to a
documented seven-algorithm sweep), **32** (Gorillas group key), **33** (timing-independent chunk
plan), **34** (regeneration as the recovery path, reframed by the repo owner — no remote needed),
**35** (`n_samples` fails loudly), **36** (group-aware splits and CV, with the leakage measured at
20%), **37** (the 5–7% hit rate, with accuracy demoted below the always-miss baseline), **38**
(one dataset contract, checked on write and on read), **8–11** (the `filter_skewed` stage, which
unbroke the `train_skewed@*` stages and made the extrapolation claim measurable), **4b**'s
deletion half, **39**, and **30**'s README half. **Still open from Phase 2: 4b's second half**
(three runs still hardcode their own sample size). **5, 6, 7** (§0) — the entire
`run_*/train_utils.py` PYTHONPATH-injection pattern (C1), gone; **C3/C4 fixed for the matrix
path** (§4). **Task 13 is withdrawn**: measurement disproved its premise, and **13b** replaces
it.

**Two architectural decisions since the last full pass, both the repo owner's, neither
originating from this audit's plan:** the DAG is now **seven `dvc.yaml` files** (root + one per
{source} × {domain} under `pipelines/`), not one, and the Gorillas effort/velocity split now
generates **four** datasets (regression and classification each get their own file per mode,
not two shared ones). Both are recorded as new entries under "Design decisions" and neither
appears anywhere else in the numbered task list below — they were not proposed by this audit,
they were requested directly and implemented in response.

**Risk #1 is closed: no stage in the DAG is known-broken.** Task 12 (the full `dvc repro`
milestone) is still open — **§0** records what was verified and how, the claims in this audit
the work proved wrong, the findings it exposed, and — read this before the next run on an
existing checkout — the silent substitutions that are now hard failures.

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

**Task 33 — timing-independent chunk plan** (`a0bdb76`). The chunk plan came from the keep rate
observed so far, which fed `need` → `n_chunks` → how many chunk seeds were drawn, so a timed-out
worker changed the dataset for a fixed `--seed`. Now `chunk_schedule(want, throws,
boards_per_chunk)` is a pure function that never sees the yield, **all** rounds' seeds are drawn
up front, and a timeout fails the run unless `--allow-partial`. Verified with `_run_worker`
replaced by a fake whose yield and timeouts are controllable (DOSBox cannot run in a Linux
checkout): `chunk_schedule` is pure with equal-sized rounds, `--workers 2` and `--workers 8`
still agree exactly, a simulated timeout raises, and every `group_id` has a single wind value
(75 of 75). Also fixes finding **N3** — the key is built by one `_chunk_key(tag, chunk_seed)`
helper and carries the session tag, so a normal-session and gravity-session seed collision can
no longer merge rows thrown under different gravity.

**Task 35 — `n_samples` fails loudly** (`259bea7`). New `params.take_samples(df, n_samples,
pool_name)` returns the prefix slice or raises `PoolTooSmall` naming both numbers. Both shared
loaders and the six evaluation/model scripts that sliced the pool themselves now go through it;
the `.sample(n=…)` sites already raised from pandas. Verified at, below and above the pool size
and on `None`, and through the real `loader.load_data`.

**Task 2b — rare-range warning** (`259bea7`). `_bounded` now warns once per column when observed
acceptance falls below `MIN_ACCEPTANCE_RATE` (10%), naming the attempt count and what it means
for the marginal. A warning rather than an error, because deliberately sampling a truncated
distribution is legitimate — it just must not be silent. Verified: fires for `gauss(100, 5)`,
silent for the configured default, once per column rather than once per draw.

**Task 38 — one contract, checked on write and on read** (`eb923f5`). `_check_contract` moved
out of `gorillas.py` into `data_generation/contract.py`; both producers write through it and both
shared loaders read through it. The read side is the half that was missing and it matters:
`data/` is DVC-cached rather than in Git, so the pool on disk can predate the code. Verified that
a simulated pre-task-32 13-column pool is now **rejected on read** instead of silently training a
model on the wrong schema. Structural and physical violations raise `ContractViolation`;
degenerate-but-legitimate states (one group, one class) warn.

**Task 4b, deletion half — 18 stale size-tier directories** (`790cbfe`). 30 files, 1,963 lines,
referenced by `dvc.yaml` zero times and self-contained rather than users of the shared loaders, so
nothing live imported them. Deleted before task 36 rather than after, per the plan's own "delete
before you repair": each carried its own `train_test_split`, and converting splits that are about
to be deleted is wasted work that would also leave nine more places a future group-aware split
could silently not apply. 4b's second half — making `run_leakage`, `run_bias_variance` and
`train_balanced.py` read `n_samples` and slice `.iloc[:n]` instead of `df.sample(n=…)` — is still
open.

**Task 36 — group-aware splits and CV** (`f08ad2c`). New `splitting.py` is the single place that
knows what `group_id` means. **The leakage is now measured, not just argued.** On a pool with the
Gorillas structure (60 boards × 32 throws, a per-board effect in no column), with a
RandomForest able to memorise the board:

```
random split      : groups on BOTH sides = 60, test MAE 12.944
group-aware split : groups on BOTH sides =  0, test MAE 16.157
```

The random split reports a number **20% better than the honest one** — a usable slide.

`split()` uses `GroupShuffleSplit`, or `StratifiedGroupKFold` when the labels must also be
balanced (which §5.4's 5–7% hit rate requires, and `GroupShuffleSplit` cannot do). `cv_for()`
returns a **materialised** list of `(train_idx, test_idx)` pairs rather than a splitter object,
because a splitter needs `groups=` again at fit time — `RandomizedSearchCV.fit` accepts that,
`RidgeCV` and `LassoCV` do not. Verified against all three. When the pool has no group structure
everything degrades to an ordinary random split, verified byte-identical, which is why one code
path serves both producers with no branching in any script. Also fixes finding **N2**: both shared
loaders now return `(X, y, groups)`.

Converted 15 top-level training scripts and the six concept scripts; no `train_test_split` remains
under `models/` except in `train_linear_regression_sqrt.py`, the orphan task 4 deletes. Verified
by running real training stages against both pool shapes — the ungrouped pool behaves exactly as
before, and the grouped pool reports "125 groups → 100 train / 25 test, no group on both sides"
with group-aware folds, for a stratified classification run as well as a regression one.

**A side effect of task 1 worth knowing:** metrics CSVs are versioned by Git now, so any test
training run dirties the working tree. One had to be reverted while verifying this task.

**Task 3 — the model matrix, cut to a documented sweep** (`1f08340`). `dvc_models.yaml` now
states its rule at the top: the sweep is seven regression algorithms (`linear_regression`,
`decision_tree`, `knn`, `polynomial`, `random_forest`, `mlp`, `xgboost`), each once per run at the
default cleaning, with exactly three deviations, each carrying its reason — `decision_tree_overfit`
(the overfitting slide), `random_forest_no_outlier` (one cleaning contrast), and `ridge`/`lasso` in
`skewed_models` only (regularised-linear vs tree *is* that run's extrapolation contrast). All
`*_range` variants are gone. **76 stages → 42.** This also closes §2.7: the matrix is now stated
rather than inferred, so a gap reads as a decision.

**Task 34 — reframed by the repo owner, and improved by it** (`1f08340`). The requirement is that
results be regenerable from the Windows machine, not that a remote hold a backup. That is the
better shape — it makes regeneration a real recovery path rather than a claim, and applies the same
reasoning the Python pool already got. The actual blocker was §5.3: `find_dosbox` searched four
hardcoded install paths, so a machine with DOSBox elsewhere could not regenerate at all. It now
resolves `--dosbox` → `$GORILLAS_DOSBOX` → `dosbox` on `PATH` → the known locations, so the
location is stated once per machine instead of edited into the file. **No DVC remote is needed
under this framing**, which is why task 34 is now closed rather than blocked.

**Task 37 — the 5–7% hit rate, handled rather than hidden** (`fd4f61b`). `print_metrics` led
with accuracy, so on a Gorillas pool a model that had learned nothing looked like it worked.
Precision, recall, F1 and PR-AUC now lead; accuracy comes last, printed beside the always-miss
baseline, with an explicit NOTE when the baseline wins. `class_weight="balanced"` where the
estimator supports it, `scale_pos_weight` (from the *training* labels only) for XGBoost, and an
honest note on `KNeighborsClassifier` and `MLPClassifier`, which have no such parameter at all —
"not every model exposes the knob" is worth saying out loud. Every search now tunes on
`average_precision` rather than `roc_auc`. Verified by running it against a pool with the Gorillas
shape (6.8% hit rate, 32 rows per group):

```
Precision : 0.3043      PR-AUC   : 0.2058  (positive rate 0.0650 = random)
Recall    : 0.1346      ROC-AUC  : 0.7293
Accuracy  : 0.9237   (always-miss baseline 0.9350  <-- accuracy beats the model here)
```

That is §5.4's lesson landing unassisted: 92% accurate, worse than a constant, and PR-AUC 3×
random showing it did learn something real.

**Task 30, README half — an end-to-end guide** (`28bffe7`). `README.md` was
`need to add some stuff here`. It now covers the two producers and why there are two, the
14-column contract, what `group_id` is for (with the measured 20%), how to generate/train/compare/
evaluate, the one edit that demonstrates each lesson, and a table of the deliberate hard failures
with their fixes. Every claim was checked against the repo, which caught two real problems: the
evaluation commands were wrong (only the six `evaluate_features.py` scripts need the PYTHONPATH
wrapper), and `run_raw`/`run_eng`'s `predict.py` still offered `ridge`/`lasso`, which task 3 moved
to `skewed_models` only — selecting either would have failed on a missing `.joblib`. Both fixed.
Still open in task 30: moving the `.docx`/`.pptx` out of Git and renaming `claude sessions/`.

**Tasks 8–11 — the skewed demo, and the last broken stages** (`596da4f`). The nine
`train_skewed@*` stages ran for the first time. They died on `ImportError: cannot import name
'FEATURE_STEP'`, and as this plan predicted, moving the filter out of the loader *is* the fix:
`run_skewed/train_utils.py` becomes the same thin shim as `run_raw`/`run_eng`, and `FEATURE_STEP`
falls out of that. **No stage in the DAG is known-broken any more** — risk #1 is closed.

New `filter_skewed` stage writes both sides — `data/skewed_training_data.parquet` and
`data/skewed_holdout.parquet` — and refuses to write an empty one, because an empty holdout means
the demo's claim cannot be measured at all. `MAX_ELEVATION_DEG` is gone (task 9); two evaluation
scripts still referenced it, one *importing* it and so about to break outright, and both now read
the filtered artifact rather than re-deriving the slice.

**The claim is now measured rather than asserted** (tasks 10, 11), verified by running both stages:

| | in-distribution MAE | holdout MAE | |
|---|---|---|---|
| skewed | 4.383 | 8.847 | **2.0× worse** |
| balanced | 7.530 | 6.234 | 0.8× — no penalty |

That inversion is sharper than the original demo. The narrow model looks **better**
in-distribution (4.383 vs 7.530) precisely because its test set shares its blind spot, and is the
only one that collapses outside it. `train_balanced.py` is a real control now: same architecture,
same split from `params.yaml`, and sized by *reading* the skewed run's row count instead of
hardcoding 10,000 against its 8,525.

This also closes §2.5 — every model-producing stage now declares the `metrics_*.csv` it writes;
the three data stages correctly declare none.

**Task 13b — `clean: "range"` stops deleting valid data** (`6a90610`). Measured what that
strategy actually dropped, over the 50,000-row pool:

```
clean:"range" dropped                          964 rows
  of which the landing_distance_m >= 0 bound    807  (84%)
    of which is_outlier == "none"               727  <- physically valid
  every other bound, clean rows dropped           0  <- all doing their job
```

Nine of the ten bounds were precise; one was deleting real data in bulk. With the lower bound
removed, `clean:"range"` drops **167 rows, 100% of them `data_error`**. Verified on a 20,000-row
pool: zero valid rows lost, where the old bound would have taken ~290. The trade is worth showing
rather than hiding — it now catches 167 of ~500 corrupted rows instead of 237, because a
sign-flipped distance is genuinely indistinguishable from a headwind landing. Less sensitive, far
more precise.

**Task 4b, second half — every run responds to `n_samples`** (`dde1822`). `run_leakage` (10,000)
and `run_bias_variance` (20,000) now read `params.yaml` and prefix-slice via
`params.take_samples`. Verified they track the param, where before they were frozen:

```
n_samples=2000   leakage train 1,600   bias_variance train 1,600
n_samples=5000   leakage train 4,000   bias_variance train 4,000
n_samples=8000   leakage train 6,400   bias_variance train 6,400
```

The evaluation side had the same defect in **more places than this task listed**, all fixed: the
two plot scripts with the *"matches train_utils.py"* comment (and their docstrings, which claimed a
seed-matched random sample), two `evaluate_features.py`, and six `df.sample(n=…)` call sites across
both shared `feature_evaluation.py` modules. A plot describing a different subset than the model
trained on is worse than no plot. **No `.sample(n=…)` remains under `models/` or `evaluation/`.**

**Task 39 — the §5.6 items, with one deliberate divergence** (`26cdf74`).

- *Constant `drag_coeff`*: cannot be dropped (both producers must emit the same columns), so
  `check_contract` now **warns about any zero-variance feature on read**, judged on clean rows —
  generic, so the next constant column announces itself too. Verified it fires for `drag_coeff` and
  does not false-positive on a real Python pool.
- *Raw logs as parquet*: **measured, because the audit's "several times smaller" needed checking.**
  2.8× on a realistically-structured 8,000-throw log (1,418,962 → 508,263 bytes) but only 1.2× on
  random floats. The win is real and depends on the data's redundancy. Columns also come back typed.
- *The `landing_height_m` clip*: **this task asked for a column and I did not add one.** The audit
  itself says a clip over 3 m "suggests a mapping bug, not a ground hit" — so those rows should not
  reach a training set, and a column would faithfully record corrupt data rather than refuse it.
  A column would also be zero-variance in every pool either producer makes, which is the dead
  weight the *first* item of this same task criticises. A clip beyond 0.01 m now **fails the run**,
  naming the likely cause, with per-throw amounts in the raw log. No rows are affected today.

**Found while verifying task 39: `gorillas.generate()` was not validating the contract at all.**
`_check_contract` ran only in the CLI, so the DVC stage was covered but an in-process caller was
not — task 38's claim that *both* producers validate on write was only true of the Python one.
Fixed.

**Tasks 5, 6, 7 — the `run_*/train_utils.py` PYTHONPATH-injection pattern, gone (2026-09-28,
main).** Done at the repo owner's direct request, after they asked why every `models/regression/
run_*/` folder held only a `train_utils.py` and what it was for — pointed straight at C1. Full
scope agreed up front (not just removing the indirection): consolidate into one config registry
and one entrypoint per domain, not a lighter fix. `models/<domain>/runs.py` (one dict, the run
configs nine `train_utils.py` shims used to hold) + `models/<domain>/train.py` (one entrypoint:
`--run --algorithm --key --clean`) + `models/<domain>/algorithms/<name>.py` (one small module per
algorithm, everything that genuinely differs — estimator, Pipeline shape, search space — with
every hyperparameter grid preserved exactly). `decision_tree_overfit` stays standalone rather
than folding into the generic driver, deliberately (task 6's entry above); `run_leakage`/
`run_bias_variance`/the two `run_skewed` concept scripts stay on `scripts/run_with_pythonpath.py`
too, deliberately (their `train_utils.py` is folder-local, used by scripts that live beside it,
never the shared-script-multiple-configs ambiguity C1 is about — touching them would have been
scope creep without fixing the actual complaint).

Two real bugs found while wiring this into `pipelines/*/dvc.yaml`, both by testing rather than
assumed, both written up in the design decision on `pipelines/` above and the README's "5.
Running one pipeline at a time": `python -m` needs the PYTHONPATH wrapper for the same cwd reason
a script-path invocation used to, just pointed at the repo root; and `filter_skewed`'s pre-
existing `--out`/`--holdout-out` `cmd:` (from the earlier `pipelines/` split, not this task) had
the identical bug — confirmed live, it silently wrote `skewed_training_data.parquet` two
directories above the repo root rather than failing.

**Also found and fixed collaterally:** `evaluation/regression/run_raw/evaluate_features.py`,
`run_eng/evaluate_features.py`, and `evaluation/classification/run/evaluate_features.py`
imported `from train_utils import ...` from folders this task deleted — not caught by grepping
`dvc.yaml`/`models/`, only by grepping `evaluation/` for the same import before deleting anything.
All three now import `DATA`/`FEATURES`/`TARGET`/`N_SAMPLES` from `models.<domain>.runs` directly
and no longer need the wrapper at all — they were the only evaluation scripts that did.

Verified incrementally, not just at the end: every `models/<domain>/train.py` path smoke-tested
directly (`python -m ...`, no DVC involved) before any `dvc.yaml` was touched — a no-search
algorithm in each domain, the Gorillas effort pool with a real group-aware split (250 groups →
200/50, matching the task-32 verification numbers), and the standalone
`train_decision_tree_overfit.py`. Every `pipelines/*/dvc.yaml` stage that changed was then
re-verified with a real (non-`--dry`) `dvc repro` run: `train_raw_effort_decision_tree_overfit`
and `train_raw_effort@linear_regression`, `train_classification@logistic_regression`, and the
full `filter_skewed` → `train_skewed@ridge` chain plus `train_skewed_concept` (the only path
that exercises the shrunk `run_skewed/train_utils.py`). `dvc stage list --all` reported 93
before touching any `dvc.yaml` and 93 after every pipeline was rewritten.

**Left alone on purpose:** `models/regression/train_linear_regression_sqrt.py` also imports from
a deleted `train_utils.py` — it is the pre-existing orphan task 4 already covers (unreferenced by
`dvc_models_regression.yaml`, unreachable by any stage), not touched here to keep this change
scoped to what is actually wired into the DAG.

### Behaviour that changed — read this before the next run on an existing checkout

Three of these tasks deliberately turn a silent substitution into a hard failure. Each is correct
and each will stop a previously-working run until acted on:

| What now fails | Why | The fix |
|---|---|---|
| Every training stage, on an existing `data/*.parquet` | Task 38 validates the contract on read, and a pool generated before task 32 has 13 columns, no `group_id` | `dvc repro generate` / `dvc repro generate_gorillas`. The error says exactly this, and nothing is lost — both producers are deterministic for a fixed seed |
| Training on a Gorillas pool with the default `n_samples: 40000` | Task 35: the pool holds 5,000 rows, and silently training on 5,000 made the run incomparable with every other size tier (§5.5) | `--set-param n_samples=5000`, or lower it in `params.yaml` |
| `generate_gorillas` when a worker hits the timeout | Task 33: accepting a half-written CSV is what made the row set depend on machine speed rather than `--seed` | Raise `--timeout`, lower `--boards-per-chunk`, or pass `--allow-partial` to accept a non-reproducible run |

All four `dvc.lock` entries were already stale before any of this (§0), so the regeneration was
owed regardless.

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

**Task 13's premise is wrong: there are no non-terminating trajectories.** §1.2 attributed the
727 clean rows with negative `landing_distance_m` to `physics.simulate` returning the endpoint of
a trajectory that never landed. Enforcing the contract (task 38) forced the question, and
measuring 20,000 shots at the configured distribution answers it:

```
did not land within max_time=60s :     0  (0.00%)
negative endpoint x              :   287  (1.44%)
negative but DID land            :   287
```

Every negative-distance shot landed, in 1–10 s against a 60 s limit, every one with `wind_x < 0`
and a high launch angle. Speed 73 m/s at 79° into 17.4 m/s of headwind lands at **−65.0 m**;
the same shot windless lands at **+30.8 m**. A high-angle shot into a strong headwind genuinely
lands behind the launch point, so these are real measurements, not artefacts. Two consequences:

- **`loader.FEATURE_RANGES` is where the real defect is** — it bounds `landing_distance_m` at 0,
  so `clean: "range"` discards physically valid headwind landings. Since those are 727 of the 964
  rows it drops, the cleaning demo is mostly removing *real* data, not corrupt data. That is the
  same conclusion §1.2 reached ("the pathology demos teach the opposite of what they claim") from
  the wrong cause. New task **13b** replaces task 13.
- **`train_linear_regression_sqrt.py` is not the bug C8 says it is.** Its sign-preserving-sqrt
  comment — *"a strong headwind can land a shot behind the launch point"* — is correct physics.
  The ⚠️ verdict in C8 is withdrawn, and task 4 must not cite it as a reason to delete the file.

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

### Effort/velocity are permanent pipelines, each with its own generated file per domain

Two decisions the repo owner made directly, superseding this audit's own recommendations:

- **Permanent, not switchable.** `train_raw_effort`/`train_raw_velocity` (and their `_eng`/
  classification counterparts) are fixed pipelines that always exist on disk side by side,
  rather than one shared group reached through `training_data`. Chosen explicitly over the
  cheaper switchable-pool alternative, knowingly reintroducing some of the duplication task 3/4b
  removed — the trade-off was named before the decision was made.
- **Regression and classification each get their own generated Gorillas file per mode** — four
  datasets (`gorillas_{effort,velocity}_{regression,classification}`) instead of two, at
  different seeds, running DOSBox twice as often. The game has no actual "regression mode" vs
  "classification mode" (one throw carries both `initial_velocity_ms` and `hit_target`, same as
  the Python pool) — this is purely so the two domains are fully independent DVC pipeline
  branches sharing no stage node, at the cost stated above.

### Seven `dvc.yaml` files, not one — `pipelines/` holds every standalone group

DVC supports multiple `dvc.yaml` files in one project; cross-pipeline dependencies resolve by
file path, the same way two stages in one file depend on each other, no special syntax needed.
The root `dvc.yaml` now holds only the two shared data producers (`generate`/`generate_gorillas`,
5 stages); every training stage lives in its own file under `pipelines/<name>/dvc.yaml`, one per
{source} × {domain} — `standard_regression`, `standard_classification`, `effort_regression`,
`effort_classification`, `velocity_regression`, `velocity_classification`. Chosen over two
cheaper alternatives an explicit design question weighed: an explicit target-list script (no new
DVC concepts, but no structural separation), and staying in one `dvc.yaml` (simplest, but
provides no way to `cd` into "just effort" and run it). Genuine structural separation was worth
the cost of every path in a pipeline file needing a `../../` prefix.

Two DVC subtleties this surfaced, both confirmed by testing rather than assumed, both now load-
bearing across every `pipelines/*/dvc.yaml` file:

- A stage's `params:` staleness-tracking list is unrelated to the `vars:` block used for `${...}`
  template interpolation — it defaults to a `params.yaml` next to the *stage's own* `dvc.yaml`,
  which doesn't exist two directories under the repo root, and fails immediately
  ("Parameters '...' are missing from 'params.yaml'") until the file is named explicitly.
- `python -m package.module` resolves against the current working directory, which DVC sets to
  the calling `dvc.yaml`'s own directory — not the repo root — so it needs the same
  `scripts/run_with_pythonpath.py` wrapper a script-path invocation does, just pointed at the
  repo root instead of a run folder. See C1 for where this bit for real.

Stage count is conserved by construction and was checked by construction, not just once at the
end: `dvc stage list --all` before touching a `dvc.yaml`, after the `pipelines/` split, and again
after task 5's consolidation all report 93.

---

## Critical risks

Status as of the 2026-09-27 update (§0).

| # | Risk | Evidence | Status |
|---|---|---|---|
| 1 | **19 of 76 stages cannot run.** All 17 `train_skewed@*` die on `ImportError: cannot import name 'FEATURE_STEP'`; `train_raw@decision_tree_overfit_no_outlier` and its `train_eng` twin write the wrong metrics filename, so DVC errors on a missing output. | reproduced | **closed** — task 3 dropped the two `*_overfit_no_outlier` stages; tasks 8–9 fixed the skewed run, and one of the nine was verified running. No stage is known-broken now |
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

~~**A silent simulator failure is mislabelled as clean data.**~~ **This diagnosis was wrong —
corrected 2026-09-27 (§0), and replaced by task 13b.** The audited claim was that
`physics.simulate` returns `traj` unconditionally, so a trajectory that never landed within
`max_time=60` contributes a meaningless endpoint as `landing_x`:

```
clean rows (is_outlier == 'none') with negative landing_distance_m:  727 / 48,500
```

The row count is real; the cause is not. Measured over 20,000 shots at the configured
distribution, **0** failed to land, and all 287 negative-distance shots landed in 1–10 s with
`wind_x < 0` at a high launch angle. A high-angle shot into a strong headwind lands behind the
launch point — that is physics, not a defect. `physics.simulate_landing` now returns termination
explicitly and `sampling.py` raises if a shot ever really fails to land, so the distinction is
checked rather than assumed.

**The conclusion survives its own cause, via the cleaners.** `clean: "range"` drops all 727 (75%
of the 964 rows it drops) because [loader.FEATURE_RANGES](models/regression/common/loader.py#L28)
bounds `landing_distance_m` at 0 — so it is discarding *physically valid* rows, and the range
demo mostly removes real data rather than corrupt data. `clean: "no_outlier"` — the oracle — keeps
every one. The pathology demos still teach the opposite of what they claim; the fix is the range
bound (task **13b**), not the integrator.

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

### C1 — PYTHONPATH dependency injection (worst offender) — **fixed (task 5)**

A reader used to open `train_random_forest.py` and see `from train_utils import load_data,
FEATURE_STEP`, with **twelve** different `train_utils.py` files (nine after task 3's cut) and
nothing in the file saying which one this was:

```
dvc.yaml  cmd: python scripts/run_with_pythonpath.py models/regression/run_raw  models/regression/train_random_forest.py  TRAIN_MODEL_NAME=…  TRAIN_CLEAN=…
   └─> run_with_pythonpath.py sets PYTHONPATH=<run_dir>        (35 LOC of wrapper)
         └─> child's sys.path[0] = models/regression/ ; falls through to PYTHONPATH
               └─> resolves to models/regression/run_raw/train_utils.py
                     └─> which does sys.path.insert(0, "../../..") to find `params`
                           └─> and reads TRAIN_CLEAN out of os.environ at load time
```

**Gone.** `models/<domain>/runs.py` is one config registry per domain; `models/<domain>/train.py`
is one entrypoint taking `--run --algorithm --key --clean` as explicit CLI arguments, not
environment variables; the sixteen duplicated algorithm scripts collapsed into
`models/<domain>/algorithms/<name>.py`, one small module per algorithm holding only what
genuinely differs (estimator, Pipeline shape, search space). `scripts/run_with_pythonpath.py`
was **not** deleted as task 5 originally proposed — it's still needed for the non-matrix stages
(`run_leakage`/`run_bias_variance`/the two `run_skewed` concept scripts, out of scope for this
task — see the design decision below) and, as it turns out, for a reason task 5 didn't
anticipate: `python -m package.module` resolves against the *current working directory*, and
every `pipelines/*/dvc.yaml` stage runs with cwd set to its own directory, not the repo root —
so a bare `-m` invocation there fails immediately with `ModuleNotFoundError: No module named
'models'`. The wrapper was widened to pass through arbitrary `python` args instead of only
`<script> KEY=VALUE`, and every `-m` invocation now routes through it with the repo root as its
PYTHONPATH argument. Confirmed by testing: this is exactly how the gap surfaced, and a second,
unrelated instance of the same bug was found and fixed in `filter_skewed`'s pre-existing `cmd:`
while this file was open (a `../../`-prefixed `--out` path silently wrote two directories above
the repo root, because `data_generation/io.py`'s `resolve_output()` anchors relative paths to
its own hardcoded repo root, not cwd).

Nine `run_*/train_utils.py` shims and sixteen algorithm scripts deleted. Verified: every
`models/<domain>/train.py` path smoke-tested directly before any `dvc.yaml` was touched, then
every `pipelines/*/dvc.yaml` stage that uses it re-verified with a real (non-`--dry`) `dvc repro`
run — including the full `filter_skewed` → `train_skewed@ridge` chain and `train_skewed_concept`
(exercising the shrunk `run_skewed/train_utils.py`, which still exists for the two standalone
concept scripts). Repo-wide stage count unchanged throughout: 93 before, 93 after.

### C2 — 44 regression stages for a talk

`regression_models` holds 22 entries, expanded across `train_raw` and `train_eng` = **44
stages**, covering `polynomial_range`, `knn_no_outlier`, `xgboost_range` and similar. No talk
shows 44 models, and the asymmetry (§2.7) means a reader cannot infer the rule.

### C3 — Two ways to engineer features — **fixed for the matrix path, deliberately not
elsewhere (task 5/7)**

| API | Derives features… |
|---|---|
| `add_engineered_columns` (plain function) | at load time, *outside* the Pipeline |
| `EngineeredFeatures` (sklearn transformer) | on the fly, *inside* the Pipeline |

`models/<domain>/runs.py`'s `feature_step()` now builds `EngineeredFeatures` uniformly for
**every** run `models/<domain>/train.py` serves — `train_raw`/`train_eng`/`train_skewed` and all
four Gorillas regression pipelines — so the nine-algorithm matrix has one API, not two.

**Deliberately left on `add_engineered_columns`:** `run_leakage`, `run_bias_variance` (out of
scope this task — see the design decision on why), `train_decision_tree_overfit` (no sklearn
Pipeline at all — it fits `DecisionTreeRegressor` directly, so there is no step to put the
transform in), and classification's loader (classification never had a raw-vs-engineered
contrast to begin with; its `load_data` has always derived features at load time, unchanged by
this task). Four call sites, not zero — task 7's original text asked for all five to convert;
one (`classification`) turned out not to need converting since it isn't part of the raw/eng
contrast, and the other three are the same scope boundary as C1's fix.

### C4 — `FEATURE_STEP = ("engineer", "passthrough")` — **fixed (task 5/7)**

The do-nothing sentinel is gone. `run_raw`'s config in `models/regression/runs.py` uses the same
`feature_step()` as every other run, just with `RAW_FEATURES` (the 9 raw columns) as
`output_columns` — `EngineeredFeatures.transform` computes the 3 derived columns and discards
them, which is free, and it means one step type with one config difference, not a sentinel a
reader has to stop and puzzle over.

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

~~**⚠️ `train_linear_regression_sqrt.py` contains a bug rationalised as physics.**~~
**Withdrawn 2026-09-27 — the file's comment is correct and this audit's claim was not.** The
comment reads *"landing_distance_m can be negative (e.g. a strong headwind can land a shot behind
the launch point) — preserve sign and magnitude"*, and measurement backs it exactly: every
negative-distance row is a genuine landing under headwind, none is a non-terminating trajectory
(§0, §1.2). The file may still be orphaned — 0 references in `dvc_models.yaml` — but task 4 must
delete it on that ground alone, not this one.

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

### 5.1 Group leakage — **fixed (tasks 32 + 36)**, and now measured at 20% optimistic

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

**Both remaining gaps are now closed too:**

- ~~The split cannot see the key (finding N2)~~ — **fixed in task 36.** Both shared loaders return
  `(X, y, groups)`, and `splitting.py` decides what to do with it.
- ~~Chunk seeds can collide across the two gravity sessions (finding N3)~~ — **fixed in task 33.**
  The key is built by one `_chunk_key(tag, chunk_seed)` helper and carries the session tag.

Measured effect of the fix, on a pool with this structure: a random split reports test MAE 12.944
where the group-aware split reports 16.157 — the leaky number is 20% better than the truth.

### 5.2 Determinism is timing-dependent — **fixed (task 33)**

`chunk_schedule()` is now a pure function of `(want, throws, boards_per_chunk)`, every round's
chunk seeds are drawn before any worker starts, and a worker timeout fails the run unless
`--allow-partial` is passed. The description below is the audited behaviour, kept as the record
of what was wrong.

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

### 5.5 `n_samples` silently truncates — **fixed (task 35)**

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
**Done: 1, 2, 2b, 3, 4b, 8, 9, 10, 11, 13b, 32, 33, 34, 35, 36, 37, 38, 39, plus 30's README
half.** **Task 13 is withdrawn** — its premise was disproved, and **13b** replaced it.
**No stage in the DAG is known-broken**, so task **12** (`dvc repro`) is unblocked and is the next
thing that matters. **All of Phase 1, Phase 1b and Phase 4 is complete.** Everything else below is
open.

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
- [x] **2b.** ~~Make a *rarely*-reachable range as loud as an unreachable one.~~ **Done —
      `259bea7`.** `_bounded` warns once per column when observed acceptance drops below
      `MIN_ACCEPTANCE_RATE` (10%), naming the attempt count and what it implies for the marginal;
      `reset_rejection_warnings()` exists so a test can assert on it. Verified: fires for
      `gauss(100, 5)`, silent for the configured default, once per column not once per draw.
      Original finding N1, for the record:
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
- [x] **33.** ~~Close the timing-dependent determinism hole (§5.2).~~ **Done — `a0bdb76`,
      taking both routes the task offered.** `chunk_schedule(want, throws, boards_per_chunk)` is a
      pure function that never sees the observed yield, sized from a constant
      `KEEP_ESTIMATE = 0.75`; every round's chunk seeds are drawn up front, before any worker
      starts, so a plan entry always consumes the same number of draws from `rng` and stopping
      early only skips already-drawn seeds. On top of that a worker timeout now fails the run
      unless `--allow-partial` is passed, since accepting a half-written CSV is what made the row
      set machine-dependent. Verified against a controllable fake `_run_worker`: the schedule is
      pure with equal-sized rounds, `--workers 2` == `--workers 8` still holds exactly, a
      simulated timeout raises, and all 75 groups have a single wind value. Still to run for real
      on the Windows machine, where DOSBox exists.
- [x] **34.** ~~Add a DVC remote and push the four Gorillas outputs.~~ **Reframed by the repo
      owner and done — `1f08340`: the requirement is that results be regenerable from the Windows
      machine, so regeneration is the recovery path and no remote is needed.** That is the stronger
      guarantee and it matches the reasoning the Python pool already got — a remote holding a copy
      would not have made the *pipeline* runnable, only the bytes recoverable. What actually blocked
      regeneration was §5.3: `find_dosbox` searched four hardcoded install paths, so a machine with
      DOSBox anywhere else failed outright. Resolution order is now `--dosbox` → `$GORILLAS_DOSBOX`
      → `dosbox` on `PATH` → the known locations, and the not-found error names all three. Combined
      with tasks 33 and 2, a fixed `--seed` now reproduces the pool rather than approximately
      reproducing it. **If the datasets ever need to be shareable with someone who has no DOSBox,
      that is a new task, not this one.**
- [x] **35.** ~~Make `n_samples` fail loudly when it exceeds the pool (§5.5).~~ **Done —
      `259bea7`.** `params.take_samples(df, n_samples, pool_name)` returns the prefix slice or
      raises `PoolTooSmall` naming both row counts and the two ways to fix it; `n_samples=None`
      still means "the whole pool", deliberately. Both shared loaders and the six
      evaluation/model scripts that sliced the pool themselves now route through it. The
      `.sample(n=…)` sites already raised from pandas, so they needed nothing. Verified at, below
      and above the pool size, on `None`, and through the real `loader.load_data`.
- [x] **36.** ~~Use a group-aware split wherever the active pool has groups.~~ **Done —
      `f08ad2c`.** New `splitting.py` holds `split()` (GroupShuffleSplit, or StratifiedGroupKFold
      when the labels must be balanced too — §5.4's 5–7% hit rate needs that and GroupShuffleSplit
      cannot do it) and `cv_for()` (a **materialised** list of `(train_idx, test_idx)` pairs, not a
      splitter object, because `RidgeCV`/`LassoCV` reject a `groups=` at fit time where
      `RandomizedSearchCV` accepts it — one `cv=` argument then works for every estimator).
      Finding **N2** is fixed with it: both shared loaders return `(X, y, groups)`. Converted 15
      top-level training scripts and the six concept scripts.
      **The leakage is now measured:** on a 60-board × 32-throw pool a random split reports test
      MAE 12.944 against the group-aware 16.157 — 20% optimistic. Verified by running real
      training stages on both pool shapes; with no group structure it degrades to an ordinary
      random split, byte-identical, so one path serves both producers.
- [x] **37.** ~~Handle the 5–7% hit rate (§5.4).~~ **Done — `fd4f61b`.** Precision, recall, F1 and
      PR-AUC lead the report; accuracy is last, beside the always-miss baseline, with a NOTE when
      the baseline wins. `class_weight="balanced"` on the logistic/tree/forest models,
      `scale_pos_weight` from the training labels only for XGBoost, and searches retuned on
      `average_precision`. **`KNeighborsClassifier` and `MLPClassifier` have no `class_weight`
      parameter at all** — both now carry a note saying so rather than quietly going unweighted,
      and the KNN note warns that `weights="distance"` weights neighbours, not classes, so it is
      not a substitute. Verified on a 6.8%-hit-rate pool: accuracy 0.9237 against a 0.9350
      baseline, PR-AUC 0.2058 against 0.0650 random. The lesson now lands from the output itself.
- [x] **38.** ~~Wire `_check_contract` into the pipeline.~~ **Done — `eb923f5`.** The checks are
      now `data_generation/contract.py`: both producers validate on write (`generate()` and
      `gorillas._check_contract`, which delegates rather than keeping a private copy) and both
      shared loaders validate on read. The read half is the one that was missing and the one that
      earns its keep — `data/` is DVC-cached rather than in Git, so the pool on disk can predate
      the code. Verified: a simulated pre-task-32 13-column pool is rejected on read instead of
      training a model on the wrong schema. Violations raise `ContractViolation`;
      degenerate-but-legitimate states (one group, one class) warn. This also covers the schema
      half of task 28, leaving the determinism test and the `pytest`/`ruff` config for it.
      **Enforcing it is what disproved task 13** — see §0.
- [x] **39.** ~~Smaller items from §5.6.~~ **Done — `26cdf74`, with one item deliberately not done
      as written.**
      - *Constant `drag_coeff`*: documented **and** made self-announcing — `check_contract` warns
        about any zero-variance feature on read, judged on clean rows. Generic, so the next constant
        column announces itself too.
      - *Raw logs as parquet*: done, and the size claim measured rather than repeated — **2.8×** on a
        realistically-structured log, only **1.2×** on random floats. The win depends on the data's
        redundancy. Columns also come back typed instead of as strings.
      - *The `landing_height_m` clip*: **no column added.** This audit says a clip over 3 m "suggests
        a mapping bug, not a ground hit", so those rows should not reach a training set — a column
        would faithfully record corrupt data instead of refusing it, and would be zero-variance in
        every pool, which is the dead weight this task's own first item criticises. A clip beyond
        0.01 m now **fails the run** with the likely cause named, and per-throw amounts go to the raw
        log. No rows are affected today.

### Phase 2 — Delete

- [x] **3.** ~~Cut `regression_models` from 22 entries.~~ **Done — `1f08340`, at the repo owner's
      chosen breadth: a seven-algorithm sweep rather than the ~4 this task first proposed.**
      `linear_regression`, `decision_tree`, `knn`, `polynomial`, `random_forest`, `mlp`, `xgboost`,
      each once per run at the default cleaning, plus three documented deviations
      (`decision_tree_overfit`; `random_forest_no_outlier` as the single cleaning contrast;
      `ridge`/`lasso` in `skewed_models` only, where regularised-linear vs tree is the point). All
      `*_range` variants dropped. `skewed_models` mirrors the same seven so the only difference from
      `train_raw` is the data; `classification_models` is the sweep minus `polynomial`, which has no
      classifier script. **76 stages → 42.** The rule is written at the top of
      `dvc_models.yaml`, which is what §2.7 asked for — a gap now reads as a decision. Verified no
      dropped key is referenced anywhere and no `train_*.py` is newly orphaned.
- [ ] **4.** Delete orphans: `train_linear_regression_sqrt.py` (unreachable — 0 references in
      `dvc_models.yaml`. **Not** for the reason first given: its sign-preserving-sqrt rationale is
      correct physics, and the C8 warning about it is withdrawn, so if anything it is the file
      that handled negative distances *properly*), `simulate.py`, `flowchart TD.mmd`, and any
      `train_*.py` left unreferenced by task 3.
- [x] **4b (first half).** ~~Delete the 18 stale size-tier directories (§5.7).~~ **Done —
      `790cbfe`.** 30 files, 1,963 lines: `run_raw_10k/20k/40k`, `run_eng_10k/20k/40k`,
      `classification/run_10k/20k/40k` under both `models/` and `evaluation/`. Verified
      unreferenced first — zero matches in `dvc.yaml`, and self-contained rather than users of the
      shared loaders, so nothing live imported them. Done *before* task 36 for the plan's own
      reason: each carried its own `train_test_split`.
- [ ] **4b (second half).** Make the three runs that still hardcode their own size read
      `n_samples` and slice with `.iloc[:n]` rather than `df.sample(n=…)`, so every tier is a
      nested prefix of one draw and responds to `dvc exp run --set-param n_samples=…`:
      `run_leakage` (10,000), `run_bias_variance` (20,000), `train_balanced.py` (10,000), plus the
      two evaluation plot scripts that hardcode `N_SAMPLES` with a *"matches train_utils.py"*
      comment. They already fail loudly on an impossible size (task 35) — this is about making
      them comparable with the tiers, which they still are not.
- [ ] **4c.** Delete the five superseded DOSBox builds — `dosbox`, `dosbox-modified`,
      `dosbox-modified-physics`, `dosbox-modified-physics-metrics`,
      `dosbox-modified-physics-metrics-var`. Keep `dosbox-datagen/`, which `generate_gorillas`
      depends on.

### Phase 3 — Collapse the indirection — **done (§0, C1/C3/C4)**

- [x] **5.** ~~Replace the PYTHONPATH injection with one `train.py` taking explicit
      arguments.~~ **Done.** `models/<domain>/train.py` takes `--run --algorithm --key --clean`;
      `models/<domain>/runs.py` holds each run's config in a plain dict; the sixteen algorithm
      scripts collapsed into `models/<domain>/algorithms/<name>.py`. One deviation from the
      original text: `scripts/run_with_pythonpath.py` was **kept**, not deleted — it's still
      needed for the stages task 5 didn't cover (see task 6's design-decision note) and, found
      while wiring this in, for `python -m` itself (DVC runs `cmd:` with cwd set to the calling
      `dvc.yaml`'s directory, so a bare `-m` invocation from a `pipelines/*/dvc.yaml` stage
      can't find the `models` package). Nine `run_*/train_utils.py` shims deleted (task 3's cut
      had already reduced twelve to nine). See §0 for what was verified and how.
- [x] **6.** ~~Fold `train_decision_tree_overfit.py` into that contract.~~ **Kept standalone
      instead, deliberately** — its whole point is a different control flow (fits on 100% of the
      data, no split, scores on training data only), and forcing that into the generic
      `train.py` harness would either break the one thing it exists to demonstrate or make the
      harness worse for every other algorithm to accommodate one exception. It moved out of
      `${regression_models}`'s foreach matrix into its own explicit stage per regression
      pipeline instead. `--clean` now applies (routes through the shared loader) and `--key`
      names its output — both bugs this task named are fixed, by a different mechanism than
      proposed.
- [~] **7.** ~~One feature-engineering API.~~ **Done for the matrix path, deliberately not
      elsewhere** — see C3/C4. `EngineeredFeatures` is now uniform across every run
      `train.py` serves, including `run_raw` (no more `"passthrough"` sentinel). Not converted:
      `run_leakage`, `run_bias_variance` (same scope boundary as task 5), `decision_tree_overfit`
      (no Pipeline to put a step in), and classification (never had a raw/eng contrast to begin
      with). Three of the five call sites this task named, not five.

### Phase 4 — Restructure the skew demo

- [x] **8.** ~~Add a `filter_skewed` stage.~~ **Done — `596da4f`, and the prediction held: the
      filter leaving the loader *was* the fix.** `data_generation/filter_skewed.py` reads the pool,
      splits on `skew.max_angle_deg`, and writes both `data/skewed_training_data.parquet` and
      `data/skewed_holdout.parquet`; it refuses to write an empty side, since an empty holdout means
      the demo's claim cannot be measured at all. `run_skewed/train_utils.py` is now the same shim
      as `run_raw`/`run_eng`, which is where `FEATURE_STEP` comes from — **the nine `train_skewed@*`
      stages ran for the first time** (verified on `random_forest`). All six hand-written stages now
      declare their `metrics_*.csv`, closing §2.5 as well.
- [x] **9.** ~~Delete `MAX_ELEVATION_DEG`.~~ **Done — `596da4f`.** The bound is
      `params.yaml`'s `skew.max_angle_deg`, so it is sweepable and appears in `dvc exp show`; the
      pool-coverage figures for choosing it are recorded in the comment there. Two **evaluation**
      scripts also referenced the constant, which this task did not mention: one *imported* it from
      `train_utils` and so would have broken the moment it was deleted, the other kept a hardcoded
      copy. Both now read the filtered artifact instead of re-deriving the slice, so neither can
      drift from what the models trained on.
- [x] **10.** ~~Report MAE on **both** test sets.~~ **Done — `596da4f`, and the demo finally
      measures what it claims.** Verified by running it: in-distribution MAE **4.383**,
      out-of-distribution MAE **8.847** — 2.0× worse. Both are written to the metrics file, so the
      comparison survives in `dvc metrics diff` rather than only on stdout.
- [x] **11.** ~~Fix `train_balanced.py`.~~ **Done — `596da4f`.** It uses the shared loader, takes
      `test_size` from `params.yaml`, engineers features in the Pipeline as the skewed model does,
      and **size-controls by reading the skewed run's row count** rather than hardcoding a number —
      so the two can never drift apart again. It is also scored on the same holdout, which makes
      them directly comparable:

      | | in-distribution MAE | holdout MAE | |
      |---|---|---|---|
      | skewed | 4.383 | 8.847 | 2.0× worse |
      | balanced | 7.530 | 6.234 | 0.8× — no penalty |

      The inversion is the lesson: the narrow model looks *better* in-distribution precisely
      because its test set shares its blind spot. The balanced model's weaker in-distribution
      number is honest — it covers the whole angle range from the same 1,359 rows.

### Phase 5 — Reproduce — milestone

- [ ] **12.** `dvc repro`, then commit `dvc.lock` so it covers the whole DAG. **Nothing below
      is measurable before this point.** Verify `dvc exp show` produces a populated table and
      `dvc metrics diff` works against HEAD.

### Phase 6 — Correctness

- [~] **13.** ~~Add a `did_land` flag to `physics.simulate`; drop or mark non-terminating
      trajectories.~~ **WITHDRAWN — the premise was disproved, see §0.** There are no
      non-terminating trajectories: 0 of 20,000 shots fail to land, and all 287 negative-distance
      shots are genuine headwind landings. The flag was added anyway
      (`physics.simulate_landing`, plus a `NonTerminatingShotError` raise in `sampling.py`) since
      it costs nothing and keeps the claim checkable if the ranges or `max_time` ever change —
      but it reclassifies **no** rows, because there were none to reclassify.
- [ ] **13b.** Fix the range bound instead, which is where the real defect was all along.
      [loader.FEATURE_RANGES](models/regression/common/loader.py#L28) bounds
      `landing_distance_m` at `(0, None)`, so `clean: "range"` throws away every headwind
      landing — 727 of the 964 rows it drops. The demo therefore mostly removes *physically
      valid* data while claiming to remove corrupt data, which is §1.2's conclusion reached
      through the right cause. Either drop the lower bound (a signed distance is meaningful) or
      make the target `abs(landing_distance_m)` with direction as its own column, and re-measure
      what `range` then drops so the slide can quote a real number.
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
- [~] **30.** Purge unrelated content. **README half done — `28bffe7`:** `README.md` was
      `need to add some stuff here` and is now an end-to-end guide — the two producers, the
      14-column contract, `group_id` and the measured 20%, how to run each stage group, the one
      edit that demonstrates each lesson, and a table of the deliberate hard failures with their
      fixes. Writing it found two real bugs (see §0). **Still open:** move the `.docx`/`.pptx` out
      of Git and rename `claude sessions/` → `claude_sessions/`. **`qbasic_gorillas/` stays** — it
      is a pipeline dependency now. Only its five superseded builds go, in task 4c.
- [ ] **31.** Fix doc drift: the `run_10k` / `run_raw_10k` / `train_all.py` /
      `generate_data.py` references in 8 live docstrings, and the intended `foreach` matrix
      after task 3.

---

**Where things stand as of 2026-09-28.** Phases 1, 1b, 4 and now **3** are complete. Open:
**4** (delete two remaining orphans), **4b second half**, **4c** (five superseded DOSBox builds)
to finish Phase 2; then **12** — the `dvc repro` milestone everything in phases 6–9 still waits
on, unmoved by tonight's work since it's a reproduce-and-lock step, not a code change. Task **14**
(move cleaning behind the split, risk #5) is the last open *correctness* risk and does not depend
on 12.

C1 — the worst clarity offender this audit found — is fixed, not just planned: `models/<domain>/
runs.py` + `train.py` + `algorithms/*.py` replaced twelve `run_*/train_utils.py` shims and
sixteen duplicated algorithm scripts, verified with real `dvc repro` runs across every
pipeline that changed, not just parsed. See §0's task-5/6/7 entry and the two new design
decisions above it for what changed and why — the `pipelines/` split (seven `dvc.yaml` files, not
one) and the four-way Gorillas dataset split (regression/classification each get their own
generated file per mode) are both the repo owner's decisions, made and implemented directly, not
proposed anywhere in this plan before they happened.

**Task 12 still needs a machine with DOSBox** for the four `generate_gorillas` variants (`§5`'s
count, current as of the effort/velocity split); everything else runs anywhere. `dvc.lock`
coverage is partial rather than zero as of tonight (`pipelines/effort_regression/dvc.lock` and
`pipelines/standard_{regression,classification}/dvc.lock` each cover the one or two stages
exercised during verification, not their whole file) — task 12 is a full `dvc repro -P` and a
`dvc.lock` commit per pipeline, still entirely ahead.

If the talk is going to use a Gorillas pool at all, tasks 32 and 36 (both done) are why a random
split no longer leaks across boards — 32 throws share one board's wind and skyline, and a plain
split used to put them on both sides of train/test.
