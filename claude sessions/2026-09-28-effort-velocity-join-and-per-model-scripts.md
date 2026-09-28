# Session: real EFFORT mode, pipelines/ reshaped per-mode then per-experiment, every matrix-driven stage converted to one script per model, a same-day cross-source join built then reversed (2026-09-28)

Continues from `2026-09-28-audit-pipelines-split-train-utils-cleanup.md`, same day, later.
That session left the repo with 6 `pipelines/*` files (one per {source}×{domain}), 4 Gorillas
datasets (`gorillas_{effort,velocity}_{regression,classification}`), and the fresh
`train.py`/`runs.py`/`algorithms/*.py` pattern from its `run_*/train_utils.py` cleanup. This
session reversed and rebuilt most of that shape again, twice, entirely in response to the user
reviewing the live file tree and reconsidering out loud — no task list drove this one. **Read
`AUDIT.md` tasks 40 and 41 first** for the structured technical record; this file is the
narrative — the sequence of pivots, the bugs actually found, and what's still unverified.

## Starting point

User opened `dvc_datasets.yaml` and asked why `datasets:`/`gorillas_datasets:` were both in one
file (informational — answered, no change). Then, unprompted, three real requests arrived in
quick succession: collapse the 4-way Gorillas split back to 2 (regression and classification
were never actually different game data), split `effort`/`velocity` into genuinely separate
pipelines, and give the standard/physics pool "effort and velocity subsections" too — which,
after a clarifying question, turned out to mean *new generation logic*, not just relabeling: the
physics generator had no effort/velocity distinction at all before this session.

## What happened, roughly in order

1. **New `EFFORT` input mode for `data_generation/generate.py`.** Mirrors `gorilla.bas`'s own
   force-capped model (work-energy over a fixed stroke length). First implementation
   rejection-sampled the derived velocity into the existing `SPEED_RANGE` — wrong, and it failed
   immediately at n=50,000 (`UnreachableRangeError`): light-mass draws can't reach `SPEED_RANGE`
   at any effort level. Fixed by not bounding the derived velocity at all — it has its own
   natural ceiling per mass, same as the game's EFFORT mode has a different ceiling than its
   VELOCITY mode. Verified with a distribution comparison (EFFORT mean 63.5 m/s / long tail to
   163; VELOCITY mean 45.1 / capped at 80).

2. **`dvc_datasets.yaml`: 5 datasets → 4** (`standard_effort`/`standard_velocity`,
   `gorillas_effort`/`gorillas_velocity`), **`pipelines/`: 6 files → 3**, one per mode instead of
   {source}×{domain}.

3. **Mid-implementation pivot ("on 2nd thought"):** join the standard and Gorillas pools per
   mode, not just keep them parallel. Both producers already emit the identical 14-column
   contract, confirmed before committing to this. Built `data_generation/join.py` +
   `join_effort`/`join_velocity` root stages. Found and fixed a real bug here: both producers
   number `group_id` from zero independently, so concatenating without adjustment collides
   unrelated rows from different sources onto the same group, which `splitting.py`'s
   `GroupKFold`/`GroupShuffleSplit` would then wrongly force onto the same split side. Confirmed
   by a synthetic test (50+30 rows → 50 distinct groups without the fix, 80 with it), then
   re-confirmed at real 50k+50k scale (100,000 groups, no collisions).

4. **`experiments/` → `experiments_results/` rename**, and a genuinely new architectural
   question: given one DVC stage per model instead of a `foreach` matrix, "should I be able to
   switch models on/off per experiment" — answered by explaining the two mechanisms (comment out
   a matrix entry vs. delete a stage block), which then became the actual direction: build
   `experiment_skew/` and `experiment_row_count/` (a new sample-size-convergence experiment, not
   previously a first-class thing) as standalone `train_<model>.py` scripts, no `--run`/
   `--algorithm` dispatch. Scoped explicitly via AskUserQuestion to "one worked example" first
   rather than migrating everything at once.

5. **User noticed `run_skewed`/`run_leakage`/`run_bias_variance` still existed** ("why is there
   still...") — these were the bespoke concept-demo scripts task-5/6/7 had deliberately left
   alone in the previous session. Migrated all three into the same `experiment_<name>/` pattern;
   `run_skewed/train_utils.py` read `RUNS["skewed"]`, which had just been deleted from
   `runs.py` — had to convert the two concept scripts in the same breath or they'd have broken.

6. **`experiment_row_count` extended to classification** ("continue to classification") — 18
   more standalone scripts, then a real stage-naming collision found wiring them: regression and
   classification share five algorithm names (`decision_tree`, `knn`, `random_forest`, `mlp`,
   `xgboost`), and the first pass gave both domains' row-count stages the same name in the same
   `dvc.yaml` file. `dvc dag` rejected the duplicate keys immediately; classification's stages
   renamed with a `classification_` infix.

7. **Effort/velocity split requested for the new experiments too** — resolved via
   AskUserQuestion into `models/<domain>/{standard,effort,velocity}/experiment_<name>/`, mirroring
   `pipelines/` exactly. Moved the already-built `experiment_skew`/`experiment_row_count` under a
   new `standard/` layer, added `effort`/`velocity` siblings.

8. **User asked how to deactivate models generally, then said it directly: "all the matrix
   driven should be converted to the one script per model kind."** The largest single piece of
   the session — every remaining `train_raw`/`train_eng`/`train_classification` (+ Gorillas
   `_effort`/`_velocity` counterparts) converted, 78 new files
   (`experiment_raw`/`experiment_eng`/`experiment_classification` × 3 domains, including the
   standalone `decision_tree_overfit` variant). `train.py`/`runs.py` deleted entirely for both
   domains once every consumer had migrated.

9. **User caught a real, unrelated staleness bug**: a plotting script's output was titled
   `gorillas_effort_regression.parquet` — a file from *before* this session's restructuring, still
   sitting in the gitignored `data/` directory alongside the correctly-named new files. Removed
   three stale files (`gorillas_effort_regression.parquet`(+`_throws`), `standard_training_data.parquet`).

10. **AUDIT.md updated** (tasks 40 and 41, plus C10 marked fixed and the two now-reversed "Design
    decisions" sections rewritten in place rather than left stale) and **README.md fully
    rewritten** to match every structural change — old version still described the 6-pipeline,
    5-dataset, `train.py`/`runs.py`-dispatch shape throughout. Asked to also save these session
    notes at this point.

11. **One more architectural pivot, arrived at through continued back-and-forth on "how do I
    disable a model."** Walked through three options (a shared per-experiment disabled-list, an
    inline `ENABLED` flag per script, commenting out the DVC stage directly) and their tradeoffs
    — the first two both hit the same wall: a script that skips without writing its declared
    `outs:` makes `dvc repro` report a failure, not a clean skip, so neither actually saves the
    "also touch the dvc.yaml" step it was meant to avoid. Landed back on stage-commenting (already
    free, no new code) as the real answer, then a follow-up question ("can I do a separate
    pipeline for each experiment") turned into task 42: `pipelines/<mode>/dvc.yaml` (3 files) split
    again into `pipelines/<mode>/experiment_<name>/dvc.yaml` (15 files), confirmed cheap since
    `dvc repro -P`/`-R <dir>`/`dvc dag` already discover arbitrarily nested `dvc.yaml` files with
    no new mechanism needed.

12. **AUDIT.md (task 42) and README.md updated again** for the 15-file split — Quick start, the
    file tree, the pipeline table, "Running one pipeline at a time" (all paths now
    `../../../`-prefixed, one level deeper again), the `--set-param` sweep examples (now need
    `-R <dir>` or a narrower `cd`, since `training_data`/`n_samples` span six separate files
    under `pipelines/standard/`), and the "adding an experiment" walkthrough.

13. **The task-40 join reversed entirely, same day, once the user actually understood what it
    did.** After a plain-language explanation of `join_effort` (concatenates standard + Gorillas
    per mode), the direct response: "please remove it, I dont want to join them, they are ment
    for two different experiments." `data_generation/join.py` and the `join_effort`/
    `join_velocity` stages deleted; every `effort`/`velocity`-domain script and pipeline file
    reverted from `data/{effort,velocity}.parquet` back to `data/gorillas_{effort,velocity}.
    parquet` directly — `pipelines/effort`/`pipelines/velocity` are, once again, purely the
    Gorillas-domain pipelines they always were before task 40. `standard_effort`/
    `standard_velocity` stay reachable through `pipelines/standard`'s pre-existing switchable
    pool, no new dedicated pipeline needed for them.

    Two real bugs surfaced by the revert, both caught by actually running a real stage
    afterward rather than trusting the text edits: `experiment_row_count`'s `TIERS` list was
    sized for the ~55,000-row joined pool (silently invalid against the Gorillas pool's actual
    5,000 rows — `pandas.iloc[:n]` doesn't raise when `n` exceeds the length, it just returns
    fewer rows, so the three largest tiers would have silently trained on the same 5,000-row
    slice three times, a fabricated convergence curve); and after fixing the tier VALUES in the
    `.py` scripts, the DVC stage's own `outs:` list still named the OLD tier-suffixed filenames,
    so `dvc repro` failed on "output does not exist" even though the script itself had run
    correctly — the tier list is encoded in two separate places (script + YAML `outs:`) and only
    one had been updated.

    Also discovered along the way: `data/gorillas_effort.parquet` genuinely exists in this
    session's environment (unclear from when/how — possibly DOSBox output from earlier in this
    repo's history, still on disk), which let two stages run for real against actual Gorillas
    game data for the first time this entire session, confirming the group-aware split
    engages correctly on genuine board-correlated data ("250 groups -> 200 train / 50 test").
    `gorillas_velocity.parquet` was not present, so velocity stayed dry-run-only.

    Also respected without touching: the two stages the user had hand-commented in
    `pipelines/effort/...` earlier in the session, AND — discovered only while investigating
    this task — nearly the entire `pipelines/effort/experiment_row_count/dvc.yaml` (13 of 15
    stages commented out by the user, leaving only `linear_regression`/`decision_tree` active).
    Verification for this task used only what was left enabled.

14. **AUDIT.md (task 43) and README.md updated once more** — every join-related sentence in both
    docs either removed or rewritten to state plainly that standard and Gorillas pools are
    trained on separately by design, not merged; the "does more data help" section's tier list
    split into two (standard's `[1000...50000]` vs. effort/velocity's `[500...5000]`) with the
    `pandas.iloc` silent-truncation reasoning spelled out so a future reader doesn't reintroduce
    the same bug by copying `standard`'s tier list into a Gorillas-domain experiment.

## Bugs found and fixed (not pre-existing design intent)

- **EFFORT-mode velocity incorrectly bounded to `SPEED_RANGE`** — a derived quantity rejection-
  sampled against a range built for a *different* mode's independent draw; failed deterministically
  for light-mass draws. Fixed by removing the bound.
- **A separate, pre-existing simulator edge case surfaced by the new mode's seed choice**: seed 43
  hits a `NonTerminatingShotError` (very-low-gravity outlier + high angle exceeds
  `physics.simulate_landing`'s `max_time`) at n=50,000 — confirmed unrelated to EFFORT mode itself
  (the triggering speed was ordinary). Not fixed at the simulator level; `standard_effort`'s seed
  moved to 47.
- **`join.py`'s `group_id` collision** across concatenated source files — see item 3 above.
- **`data_generation/generate_all.py` was about to silently generate wrong data** — it never
  passed `input_mode` through to `generate()`, so every dataset would have generated in the
  default `VELOCITY` mode regardless of its own `dvc_datasets.yaml` entry. Caught before ever
  being run.
- **Stage-name collision** between regression and classification `experiment_row_count` stages —
  see item 6.
- **A bulk-templating bug in the Python file generator**: the first pass at effort/velocity
  variants of `experiment_raw`/`experiment_eng` used a `sed` line-range deletion to strip the
  `n_samples` paragraph out of each docstring; it cut into the *next* paragraph instead, leaving a
  dangling half-sentence, and left every file's opening line saying "the standard pool's"
  regardless of actual domain. Caught by reading the first generated file rather than trusting the
  substitution; redone with precise Python multi-line-string replacement.
- **A bulk-templating bug in the DVC YAML generator**: the conditional `params:` block for
  `train_*_decision_tree_overfit` stages was appended as a string directly after the last `deps:`
  entry with no line break establishing a new top-level key, so it parsed as a 6th `deps:` list
  item instead of a sibling `params:` key. Caught immediately by `dvc dag --full` before anything
  was run (`expected str, in stages -> train_raw_decision_tree_overfit -> deps -> 4`).
- **Six `evaluation/` scripts broken by the `runs.py`/`train_utils.py` deletions**, found across
  two separate passes (once for `run_skewed`/`run_leakage`/`run_bias_variance`'s
  `evaluate_features.py`, once for `run_raw`/`run_eng`/classification `run`'s) — same shape of
  breakage the previous session's task 5/6/7 already found and fixed elsewhere in `evaluation/`,
  same fix (constants inlined). Three more `predict.py` files and both `compare_models.py`
  scripts hardcoded old `experiments/regression/run_raw/...`-style paths, found only by a second,
  broader grep for the literal string `"experiments"` — the `RUNS`-import grep didn't catch these
  since they never went through `runs.py` in the first place.
- **Stale pre-restructuring `data/` files** — see item 9.
- **The task-42 stage-splitting script's naive `"../../"` → `"../../../"` string replace missed
  the bare PYTHONPATH argument** in every `cmd:` line (`python ../../scripts/
  run_with_pythonpath.py ../.. -m ...` — the second `../..` has no trailing slash, since it's
  followed by a space, not another path segment). Would have silently pointed every `-m`
  invocation at the wrong directory the moment the files actually ran. Caught by reading the
  first generated file before trusting the rest; fixed with a regex matching `../..` regardless
  of what follows it.
- **The same script had to preserve two stages the user had manually commented out** in
  `pipelines/effort/dvc.yaml` mid-session, as a live test of the "just comment out the stage"
  answer — the block-splitting regex needed to match commented-out stage headers (`  # name:`)
  as well as active ones, or it would have silently dropped them during the split. Verified by
  grepping for both before and after.

## Open items, explicitly not done

- **The entire Gorillas-dependent path is unverified in this session's environment** — no DOSBox
  available. `generate_gorillas`, `join_effort`/`join_velocity`, `pipelines/effort`,
  `pipelines/velocity`, and every effort/velocity experiment script (60 regression + 18
  classification files from item 8 alone) are wired and `--dry`-clean but have never actually run.
  One representative standard-domain stage (`train_eng_ridge`) *was* run for real (non-`--dry`)
  and produced `dvc.lock` entries — the only proof so far that the new pattern works, not just
  parses.
- **`dvc_datasets.yaml` splitting into separate files was discussed and explicitly declined** by
  the user ("leave as is for now") after being talked through why a mode-based split wouldn't map
  cleanly onto the root `dvc.yaml`'s two generator-module-based stages.
- **`evaluation/regression/run_leakage/plot_training_data_relationships.py` and
  `run_bias_variance`'s sibling still hardcode `data/raw_500k.parquet`** — a file that does not
  exist anywhere in this repo. Confirmed pre-existing (predates this session entirely, unrelated
  to anything touched here) and left alone rather than guessed at.
- **Task 12** (full `dvc repro -P`, `dvc.lock` committed) — still not done, now additionally
  blocked on DOSBox availability for the Gorillas half.
- Tasks 13b, 14, 26, 28 — unchanged from the previous session, still open.
- **Nothing from this session has been pushed** — same as the previous session ended.

## Repo-specific gotchas worth remembering (carried forward + new)

- Everything in the previous session's gotchas list still applies (`PIPENV_IGNORE_VIRTUALENVS=1`,
  `python -m` needing the PYTHONPATH wrapper, `params:` needing the file spelled out,
  `resolve_output()`'s repo-root anchoring, `--glob` being unreliable, `dvc` subcommand flags not
  being consistent).
- **`data/` is entirely gitignored**, so a dataset rename/restructure does not clean up what it
  superseded — old files silently persist and can be picked up by a script (or a person) that
  points at a name that used to be current. Worth an explicit `ls data/*.parquet` sanity check
  after any `dvc_datasets.yaml` renaming, not just a check that the *new* names generate correctly.
- **When bulk-templating many near-identical Python files via `sed`, prefer a Python script with
  precise multi-line string replacement over line-range deletion (`/pattern/,+N d`)** — a
  line-count-based delete silently cuts into whatever happens to be N lines below the match in
  *every* file, which is only safe if every file's surrounding text is byte-identical at that
  offset. It wasn't, for the effort/velocity docstring variants.
- **When bulk-generating DVC stage YAML programmatically, a conditional block appended without an
  explicit blank-line/key boundary can silently nest under the wrong parent key** rather than
  becoming a sibling top-level key — YAML's indentation-based structure doesn't complain until
  parse time. `dvc dag --full` catches this immediately and names the exact stage; run it right
  after any bulk YAML edit, not just after the templating script reports success.
- **`physics.simulate_landing`'s `max_time` can be exceeded by a legitimate, rare draw** (very low
  gravity-outlier + high launch angle) — this is seed-dependent, not specific to any particular
  input mode, and not something to "fix" by tuning the simulator; trying a different seed for that
  one dataset entry is the accepted, documented workaround (see `standard_effort`'s seed 47 vs. 43).
- **Regression and classification share several algorithm names** (`decision_tree`, `knn`,
  `random_forest`, `mlp`, `xgboost`). Once both domains' stages can land in the *same* `dvc.yaml`
  file (true for every `pipelines/<mode>/dvc.yaml` now that pipelines are per-mode, not
  per-{source}×{domain}), any new pair of same-named stages across domains needs an explicit
  disambiguating infix in the stage name — DVC errors clearly on the collision, but only at
  `dvc dag`/`dvc repro` time, not at file-write time.
