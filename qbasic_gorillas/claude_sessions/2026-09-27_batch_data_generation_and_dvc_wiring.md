# Session: Gorillas as a training-data generator — batch build, Python harness, DVC wiring

**Date:** 2026-09-27
**Scope:** New folder `qbasic_gorillas/dosbox-datagen/` (copied from
`dosbox-modified-physics-metrics-var-display`), new `data_generation/gorillas.py`,
and — outside `qbasic_gorillas/` entirely — the `training_data` centralisation
and `generate_gorillas` DVC stage across the main repo (`params.yaml`,
`params.py`, `dvc.yaml`, `dvc_datasets.yaml`, and ~11 Python files that used to
hardcode `data/standard_training_data.parquet`). All six original DOSBox
folders are untouched except for a pre-existing mount-path bug fixed across
all of them in its own commit (Phase 0, below) — that bug meant none of them
launched at all before this session.

Full design plan (context, decisions, phase-by-phase rationale) is at
`C:\Users\MichaelVictor\.claude\plans\synthetic-squishing-castle.md` — this
file is the "what actually happened and what to watch for" complement to it,
in the style of the other files in this folder.

## Outcome (what to use)

- **Four launchers in `dosbox-datagen/`**: `Play Effort.bat` / `Play
  Velocity.bat` (interactive), `Generate Effort.bat` / `Generate Velocity.bat`
  (unattended, full graphics, writes `THROWS.CSV`). One `gorilla.bas`, config-
  driven (`GORCFG.TXT`), not four separate copies.
- **`data_generation/gorillas.py`** — the volume/CI path. Drives DOSBox in
  parallel worker processes, maps the throw log onto `generate.COLUMNS`,
  writes parquet. `--dry-run` / `--overwrite` / default-refuse write modes.
- **`params.yaml`'s `training_data` key + `params.py.data_path()`** — the
  single place that selects which generated pool every training/evaluation
  script reads. Was hardcoded in 11 files before this session.
- **`dvc_datasets.yaml`'s `gorillas_datasets:` + `dvc.yaml`'s
  `generate_gorillas` stage** — real, not just designed: `dvc repro` was run
  for both variants and the training pipeline both for real.
- **Two real datasets on disk**, produced by the actual DVC stage:
  `data/gorillas_effort_data.parquet` and `data/gorillas_velocity_data.parquet`
  (n=5000 each, seed 42), plus their raw throw logs
  (`data/gorillas_*_throws.csv`, which keep the `board`/`outcome` columns the
  13-column contract deliberately omits).

## 1. Phase 0 — the mount bug, and why originals got committed first

`qbasic_gorillas/` was **not tracked by git at all** (0 files in the index,
not gitignored) when this session started, despite `AUDIT.md` treating it as
part of the repo. Committed it byte-for-byte pristine first (`a7f89ff`), *then*
fixed the bug (`407e231`), specifically so the bug and its fix are both on
record rather than silently absorbed into one "add the subproject" commit.

The bug: every `play.conf`/`edit.conf` in all six folders (12 files) mounted
`C:\Projects\qbasic_gorillas\<folder>` — a path that doesn't exist; the repo
actually lives under `...\talk_Introduction_to_ml\qbasic_gorillas\`. **None of
the six folders launched as committed.** Fixed portably (`mount c "."` +
`pushd "%~dp0"` in the launcher bats) rather than re-hardcoding the new
absolute path, and verified via an unattended `dir gorilla.bas` probe over the
mounted drive for all six folders before touching anything else.

Also added `qbasic_gorillas/.gitattributes` (`* -text`, binaries marked)
*before* committing — `core.autocrlf=true` is set globally on this machine and
would otherwise have rewritten the LF-only `.bas`/`.conf`/`.bat` files to CRLF
on checkout, silently changing their bytes the moment anyone cloned or
recheckout'd.

**Mistake made and recovered:** first attempt used `git commit --amend`, which
pulled the user's ~74 already-staged unrelated files into the same commit.
Caught immediately, fixed with `git reset --soft HEAD~1` (loses nothing) and
`git commit --only -- qbasic_gorillas/` instead. Lesson below.

## 2. `dosbox-datagen/`: one source, two input variants, config-driven

Copied from `-var-display` (the only folder with both the corrected geometry
*and* all the physics features — an older `-physics` folder has a raw
`Velocity:` prompt but predates the 2026-09-06 geometry fixes, so its
`GorillaCenterX/Y` don't exist and its distances measure the wrong thing; that
variant had to be built forward, not recovered by reverting).

**`INPUTMODE=EFFORT|VELOCITY`** plus `EFFORTJITTER`/`BANANAVAR`/`HUD`/
`ANIMATE`/`DESTRUCT`/`FASTDRAW` flags, all read from `GORCFG.TXT` by a new
`ReadCfg`. Four launchers = four small `.conf`/`.CFG` pairs over one
`gorilla.bas`, not four copies of the whole game — the `-metrics` →
`-var-display` diff was already only 121 lines for a much bigger feature set;
duplicating the whole file to vary ~15 lines of input mode is the trap this
tree already fell into six times.

**D1 (velocity precision), applied:** `DEFINT A-Z` makes `Velocity` an
INTEGER, and QuickBASIC *rounds* on assignment, so the original
`SQR(2*Force#*ThrowStrokeM#/BananaMass#)` could only ever produce ~45 distinct
whole-m/s values — a hard floor on the regression target's resolution. Changed
to `Velocity#` in this folder only; the six originals keep the integer.

**D11, withdrawn — do not "fix" this if it comes up again:**
`MaxThrowVelocity#` is 44 here vs 60 in `-metrics`. Both are correctly
calibrated, for *different bananas*. `-metrics` uses a real 0.15 kg banana
(high drag, needs up to 53.4 m/s at p99.9 → cap 60); this folder uses the
giant ~150 kg banana (Area/Mass ratio ~10× lower → ~10× lower drag → only
needs ≤39.2 m/s at p99.9 → cap 44, same ~12% margin). Raising 44→60 here would
be wrong, not a fix.

**QBasic 1.1 has no `ON ERROR RESUME NEXT`** (that's QuickBASIC/VB) and
`RETURN` out of an error handler isn't valid either — the config-exists probe
uses `RESUME <label>` at module level instead, and that only works with the
handler in the same scope, which is why `CfgFound` is set once, at module
level, before `ReadCfg` runs, rather than inside `ReadCfg` itself.

**DOS text files need CRLF, not just line-count-correct content.** `LINE
INPUT` terminates on CR; an LF-only `GORCFG.TXT` is read as *one single
enormous line*. Only the first `KEY=VALUE` before the first embedded `\n`
parses (as `DATAMODE=<everything else concatenated>`, whose `VAL()` happens to
be 1) and every other setting silently keeps its default. This produced a very
confusing first failure: the config file existed, had the right content, and
was silently 95% ignored — looked exactly like "the harness isn't passing
settings through" rather than a line-ending bug. `data_generation/gorillas.py`
always writes configs via `_write_dos_text`, which normalises to CRLF.

**`dosbox -c "QBASIC.EXE ..."` does not reliably launch the interpreter here —
`[autoexec]` does.** This exact finding is already in the 2026-09-05 session
notes and got rediscovered the hard way: several `-c`-driven probes exited 0
with no output and no error, looking like a silent QBasic compile failure,
before isolating it down to "even a trivial file-I/O-only `.bas` fails via
`-c` but works via `[autoexec]`."

**Collision-neutral drawing skips, verified rather than assumed.** Same seed,
`FASTDRAW=0` vs `FASTDRAW=1`, 48-row comparison: every physics/outcome field
bit-identical; only `flags`, `timer_s`, and `pointval` on the 3 rows that hit a
*window* differ (5/6/7 building colour instead of 8/14 window colour). Why
each is safe: the banana is erased *before* `POINT` samples and redrawn
*after* (loop order in `PlotShot`), so it's never on screen during its own
probe; windows are filled rects painted *on top of* an already-solid building
rect, so `Impact` can't change; the sun sets `ShotInSun` but never `Impact`.

**`Rest`'s wait scales *up* with emulated CPU speed** (`t2# = MachSpeed * t# /
SPEEDCONST`), so raising `cpu_cycles` alone makes the unmodified game slower,
not faster — confirmed again this session, matching 2026-09-05. One line
(`IF NOT Animate THEN EXIT SUB` as `Rest`'s first line) de-blocks all five call
sites at once.

**Throughput, measured:** 1 instance, `Animate=0`: 15.1 rows/s. Parallel
workers on this 12-core machine: 10 workers/throws=32 → 174.9 rows/s; 12
workers + small window → 195.4 rows/s; + a 60-tick flight cap → 209.4 rows/s.
~13× a single instance. `cpu_cycles`/`core=dynamic` only help *after* `Rest`
is short-circuited — before that, raising them is actively counterproductive.

## 3. `data_generation/gorillas.py`

Maps the throw CSV onto `generate.COLUMNS`. `is_outlier` is populated
honestly rather than left all-`"none"`: `"gravity"` comes from a genuinely
separate session run at an out-of-distribution gravity (the game already
takes gravity as an input), `"data_error"` from the *same*
`outliers.corrupt_row` the Python generator uses.

**A bug I introduced and then caught in my own contract checker, not
downstream:** the first `_check_contract` asserted `FEATURE_RANGES`-style
bounds on every row, including `data_error` rows — which are corrupted
*on purpose* and are supposed to violate those bounds (that's what
`clean:"range"` exists to demonstrate). It failed on the very first real run.
Fixed by excluding `is_outlier == "data_error"` rows from the range
assertions, same as the actual training-time cleaning logic does.

**A real determinism bug, caught before it shipped:** the first version drew
each parallel worker's per-session seed from one shared `random.Random(seed)`
stream *inside* the `for w in range(workers)` dispatch loop. That meant
`--workers 4` vs `--workers 8` consumed a different number of seeds per round
and sized each worker's board count differently — i.e. **changing `--workers`
silently changed the resulting dataset**, which would have made the DVC stage
non-reproducible (same `--seed`, different machine core count → different
cached data under an identical cache key). Fixed by drawing *all* of a round's
chunk seeds single-threaded, up front, from a fixed formula
(`n_chunks = ceil(remaining / (boards_per_chunk * throws * keep_estimate))`)
before touching the thread pool at all — `workers` now only controls how many
of those predetermined chunks run concurrently. Added `--boards-per-chunk` as
the (correctly) data-affecting replacement for the old dead `boards=None`
parameter, which was accepted but never actually read anywhere in the
function body.

**Verified, not just fixed:** same seed/throws/boards-per-chunk, `--workers 4`
vs `--workers 8` → `DataFrame.equals()` **True**, byte-for-byte identical, 800
rows. Same seed with `--boards-per-chunk 10` vs `20` → correctly **not**
equal, confirming that knob is genuinely data-affecting and belongs in the DVC
`cmd:` (which it does — `workers` deliberately does not).

## 4. Phase 7a — centralising the hardcoded data path

`data/standard_training_data.parquet` was hardcoded in ~11 Python files (six
`run_*/train_utils.py`, `train_balanced.py`, four `evaluation/regression/*`
plotting scripts) and 10 `dvc.yaml` `deps:` lines. Added `training_data:
standard_training_data` to `params.yaml`, `params.data_path()` in `params.py`,
pointed every one of those ~11 files at it, added `params.yaml` to `dvc.yaml`'s
`vars:` and interpolated `data/${training_data}.parquet` into all 10 `deps:`
lines plus a `training_data` entry in every stage's `params:` list (including
the six hand-written single-instance stages that previously had no `params:`
block at all).

**Mistake made and recovered:** while proving the switch worked, used
`yaml.safe_dump` to write `training_data` back to its default — this silently
stripped every comment in `params.yaml` and reformatted the compact
`{n_iter: null, cv: 5}` inline dicts into expanded block style. **Never
round-trip a hand-formatted YAML config through `yaml.safe_load`/
`yaml.safe_dump` — it is not comment-preserving.** Restored by rewriting the
file from the known-good content (I had it from my own prior edit) via `Write`,
verified line-by-line; all subsequent test-and-revert cycles used exact-string
`Edit` instead.

**Verified, in order:** `dvc dag --full` and `dvc repro --dry` compile
cleanly; the *one* stage that actually had a `dvc.lock` entry before this
session (`generate@standard_training_data` — confirmed via direct inspection
that literally nothing else was ever locked, so the pipeline had never been
fully reproduced) reports **clean** both before and after the refactor,
proving zero unintended staleness; `repo.stage.collect(...)` inspection
confirms `${training_data}` resolves to the literal path and that
`training_data` shows up as its own tracked param key (not just a file-level
dep) on every stage, including the six that previously had no `params:` block;
setting `training_data` to a nonexistent name fails cleanly with a
"file not found," not a crash.

## 5. Phase 7b — the `generate_gorillas` DVC stage, run for real

`dvc_datasets.yaml` gained a `gorillas_datasets:` section (two entries,
`gorillas_effort_data`/`gorillas_velocity_data`, n=5000, seed 42, same board
count so they're directly comparable) and `dvc.yaml` gained a
`generate_gorillas: foreach: ${gorillas_datasets}` stage — deliberately a
*separate* foreach over its own manifest/key namespace rather than a
`source:`-flag dispatcher with union deps, because the rejected alternative
would make editing `gorilla.bas` invalidate the Python-generated dataset too
and cascade into all 56 training stages. `--workers` is deliberately absent
from the stage's `cmd:` (per §3, it isn't part of the dataset's identity);
`gorilla.bas` is a dep, `QBASIC.EXE` deliberately is not (never changes, DVC
would cache it per-dataset for nothing).

**Actually run, not just compiled:** `dvc repro generate_gorillas@
gorillas_effort_data` and `...@gorillas_velocity_data` both executed for real
(117s and 148s respectively) and are now genuinely recorded in `dvc.lock`.
Switched `training_data` to `gorillas_effort_data` and ran `dvc repro
train_raw@linear_regression` for real too — it trained on the actual game
data purely because of the `params.yaml` setting, no other file touched. Then
reverted `training_data` to the default and re-ran that same stage once more
so `dvc.lock` doesn't sit in a stale/mismatched state relative to the
reverted config.

## 6. Findings from the real generated data (talk material)

**Occlusion is real and now measured, not just described.** Across the raw
throw logs (8000 + 8400 throws, both variants): **zero** throws reached open
ground unobstructed (`outcome == 5`) in either sample. ~71% of throws in the
effort variant hit a building (`outcome == 1`) before landing. Re-simulating
those occluded shots' exact `(velocity, angle, wind, mass, radius)` in open
air via `physics.simulate` (the same drag formulation the game itself uses)
shows the actual logged landing distance averages **43.5 m** against **88.7 m**
of true open-air range for that shot — the median occluded throw loses about
a third of its true range to a building that isn't in any of the 13 logged
columns.

**Why there are zero ground-clearing throws:** buildings are drawn
essentially edge-to-edge across the whole 640px field (only a 2px gap between
neighbours), so at the sampled angle/velocity ranges a descending trajectory
that stays in bounds long enough to reach ground level has almost certainly
already crossed some building's x-range on the way down. This means the
plan's proposed `--clean-flights-only` filter (train only on unobstructed
throws, to restore a clean regression target) **would currently yield zero
rows** — it needs angle/velocity ranges deliberately tuned toward near-vertical
lobs that clear the whole skyline, which is a real follow-up experiment, not
a one-line flag flip.

**A claim I got wrong and had to correct in-thread:** I first attributed a
6→17 RMSE jump (comparing an ad-hoc small-sample test against the real
`dvc repro` run) mainly to occlusion. It wasn't. `dvc_models.yaml` sets
`clean: ""` for `linear_regression` — i.e. that stage deliberately trains on
*everything*, including the 1% of rows `corrupt_row` scales/flips/zeroes on
purpose. Isolating it: all rows (= the real `dvc repro` run) → RMSE 16.98;
excluding just the `data_error` rows → RMSE 6.87; clean rows only → RMSE 6.54.
~90% of the gap was the deliberately-corrupted rows, not occlusion. Lesson:
when a metric looks surprising, reproduce the *exact* run condition (here,
`TRAIN_CLEAN` per `dvc_models.yaml`) before attributing it to a new-feature
hypothesis, however plausible that hypothesis is.

**`landing_height_m` already helps, but only for a nonlinear model.** It's
already one of the 9 raw features (`models/regression/common/loader.py`
`FEATURES`) — not something to add later. With it vs. without, on clean
`gorillas_effort_data` rows: LinearRegression RMSE 6.54 → 6.57 (~no help);
RandomForestRegressor RMSE 3.68 → 4.25 (**13% worse without it**, MAE 19%
worse). It can't fully resolve occlusion because the same `(distance, height)`
endpoint is reachable from very different original velocities depending on
*which* board's specific skyline did the blocking — information that lives
only in the raw throw log's `board` column, deliberately excluded from the
13-column contract.

## Lessons for next time

- **Never round-trip a hand-formatted YAML/config file through a generic
  parse-and-dump cycle** (PyYAML, or anything else) just to test-then-revert a
  single value — it silently destroys comments and reformats compact inline
  structures. Use exact-string edits for the test *and* the revert.
- **`git commit --amend` uses the current index**, not just the previous
  commit's tree — it will happily absorb whatever else is staged. Use
  `git reset --soft HEAD~1` + `git commit --only -- <path>` instead when a
  path-scoped commit's message needs fixing and other unrelated work is
  staged alongside it.
- **A CRLF/LF mismatch in a DOS text file doesn't fail loudly** — it reads as
  one giant line and only the first key takes effect, which looks exactly
  like "settings aren't being read" rather than a line-ending problem. Any
  new file written for QBasic to `LINE INPUT` needs `_write_dos_text`-style
  CRLF normalisation, not just correct content.
- **Before attributing a surprising metric to an interesting new hypothesis,
  reproduce the exact run condition first.** The occlusion story was real,
  but a bigger and more mundane effect (uncleaned corrupted rows, by design)
  was sitting underneath it and had to be isolated before the smaller,
  genuinely-new effect could be measured cleanly.
- **A parameter that's threaded through a parallel-dispatch loop can leak into
  the data even when it's "obviously" just a performance knob.** `--workers`
  looked purely about wall-clock time; because per-worker seeds were drawn
  from a shared RNG stream *during* dispatch, it wasn't. Draw all
  data-determining randomness single-threaded before any concurrency touches
  it, and write a same-seed-different-worker-count equality test to prove it,
  not just eyeball the row counts.
