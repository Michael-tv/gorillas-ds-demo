# Session: Repo audit, effort/velocity + pipelines/ split, run_*/train_utils.py cleanup (2026-09-27 -> 2026-09-28)

Continues from `2026-09-27-dvc-experiments-and-data-manifest.md`. That session built the
single-`dvc.yaml`/`dvc_models.yaml` pipeline this one restructures into seven files and a
different training-script layout. **Read `AUDIT.md` first** for the full structured record
(findings, numbered task list, §0 implementation log with what was verified and how) -- this
file is the narrative of how the session actually went, session-handoff detail AUDIT.md doesn't
carry: the sequence of user decisions/corrections, bugs found, and gotchas.

## Starting point

The repo had accumulated a *second* Claude Code session's parallel work (branch
`claude/dreamy-gauss-9h4w73`, merged into `main` via 4 GitHub PRs while this session was
independently committing to its own `audit/phase-1` branch) -- discovered mid-session via a git
log/branch scan, not announced anywhere in the conversation. AUDIT.md's own top summary was
lagging its own detailed log by one commit at that point (a real doc-staleness bug, confirmed by
diffing the exact commit that should have updated it and hadn't).

## What happened, roughly in order

1. **Wrote AUDIT.md from scratch** -- the four-dimension audit as requested (data generation
   rigour, DVC pipeline structure, code quality, CLAUDE.md check), then a complexity-focused
   re-pass after being asked specifically for that lens. Found C1 (PYTHONPATH injection) as the
   worst clarity offender at that point -- not acted on until step 10, much later.

2. **Implemented tasks 1 (metrics tracking) and 2 (deterministic seeding)** directly, each
   verified concretely: `git check-ignore` before/after; two seeded `generate()` calls
   byte-identical, a third unseeded call diverges; regenerating the committed pool from
   `dvc.lock`'s exact recorded params reproduced it exactly (`df.equals() == True`).

3. **Three-round decision cycle on skewed/Gorillas/feature-engineering design** (D1/D2/D3 in
   AUDIT.md's Decisions log, each kept visible with its superseded reasoning rather than
   overwritten):
   - D1: skewed models read from the base pool via a filter *stage*, not a separate generated
     dataset. First proposal deleted `skewed_models` entirely; corrected -- keep it, move the
     filter into a stage.
   - D2: no DVC remote, the user's own call, confirmed independent of the local-cache-only
     experiment workflow (`dvc exp run`/`show`/`diff` need no remote).
   - D3: engineered features generated on the fly (sklearn Pipeline step), not as a materialized
     `features_eng` stage. First reading proposed a stage; the next message ("generated on the
     fly in the pipeline") reversed it. Landed on `EngineeredFeatures` uniform across every run
     including `run_raw` -- the `"passthrough"` sentinel replaced by the same step type with a
     raw column list.

4. **Progress scan** ("scan audit.md latest pull requests and repo status") -- surfaced the other
   session's 4 merged PRs, reconciled with local work.

5. **Task 32** (real Gorillas group key) verified against *live DOSBox output*, not just code
   review: measured `wind_ms` non-constant in 25/25 groups before the fix, constant in 0/33 after,
   from a real n=150 generation.

6. **`dvc repro` run/interrupt cycle.** User ran a full `dvc repro` in their own terminal
   (typo "dvc repo" corrected first); a redundant attempt to also run one from this session
   collided on DVC's rwlock and failed safely; the user's kept running until asked to stop it
   (to make room for the effort/velocity redesign). Stopped via `Stop-Process`; stale
   `.dvc/tmp/rwlock` removed by hand; `dvc.lock` had orphaned entries once the redesign renamed
   stages and was deleted outright rather than hand-edited -- nothing valid left to preserve.

7. **Effort/velocity redesign**, corrected twice within one exchange:
   - First reading: switchable `training_data` pool, same mechanism as everything else.
     Corrected -- permanent parallel groups instead.
   - Next message: regression and classification should be "completely different datasets and
     dvc pipeline runs." Scoped via a two-question AskUserQuestion (just effort/velocity, not the
     standard pool; separate generated files, not just separate branches reading one file) into 4
     Gorillas datasets instead of 2 (`gorillas_{effort,velocity}_{regression,classification}`,
     different seeds). Flagged plainly and built anyway: the game has no actual "regression mode"
     vs "classification mode" -- one throw carries both targets -- so this is purely about
     independent DVC branches, at 2x the DOSBox cost. Also split `dvc_models.yaml` into
     per-domain files in the same exchange, and added two explicitly requested README sections
     ("adding experiments and data" / "switching a model off").

8. **`pipelines/` split** -- "how do I have separate dvc repro pipelines [for effort/velocity/
   regression]? do they need separate projects?" Answered via AskUserQuestion (six groups by
   {source}x{domain}; separate `dvc.yaml` files chosen over wrapper scripts, for genuine
   structural separation). Built and fully verified **one** pipeline (`effort_regression`) before
   replicating to the other five -- the first attempt surfaced a real DVC subtlety (`params:`
   file resolution) that would have silently broken all six if replicated blind. `--glob` on
   `dvc repro` was tested and found unreliable at this scope (matched unrelated stages from a
   different foreach group); not used anywhere as a result.

9. **Merge to `main`.** Local `main` was ~30 commits stale; `origin/main` already had everything
   via the other session's PRs. Fast-forwarded local `main` to `origin/main`, merged
   `audit/phase-1` in with an explicit merge commit (no `gh` CLI available to open a real PR, so
   this is the closest match to the repo's existing PR-merge convention), pushed. Then asked to
   delete `audit/phase-1` -- deleted locally and on origin after confirming full merge via
   `git branch --merged main`.

10. **The `run_*/train_utils.py` cleanup (the largest single piece).** Triggered by the user
    looking directly at the VS Code file tree and asking why every `run_*/` folder held only a
    `train_utils.py`. Scoped via AskUserQuestion before touching anything -- full consolidation
    into `runs.py`/`train.py`/`algorithms/*.py`, not just removing the PYTHONPATH indirection.
    Full technical record is in AUDIT.md §0's task-5/6/7 entry. Verified incrementally: every new
    `train.py` path smoke-tested directly (`python -m ...`, no DVC) before any `dvc.yaml` was
    touched; every `pipelines/*/dvc.yaml` stage re-verified with a real, non-`--dry` `dvc repro`
    run after. Two real bugs found by testing, not assumed (see gotchas below).

11. **AUDIT.md updated** to match all of the above -- new design-decision entries for the
    `pipelines/` split and the 4-way Gorillas split (neither ever appeared in the audit's own
    plan; both were the user's decisions, implemented directly, not proposed here first), C1/C3/
    C4 marked fixed, Phase 3 checked off, health score 5→6. One overclaim caught before
    committing: an early draft of the summary said "4b both halves done" — checked the actual
    checkbox, found the second half still open, corrected before writing it.

## Bugs found and fixed (not pre-existing design intent)

- **`gorillas.py`'s `corrupt_row` call was never actually seeded.** The module created its own
  `random.Random(seed)` but called `corrupt_row`'s old global-`random`-based signature, so that
  call silently drew from the unseeded global module. Fixed as a side effect of task 2's
  rng-threading, not sought out separately.
- **`_run_worker` futures/chunk-seed scoping bug.** The `ThreadPoolExecutor` futures list lost
  its association with chunk seeds (list-comprehension scope), so `_to_contract` had no way to
  know which chunk a completed future's rows came from. Found while implementing the group-key
  task; fixed by pairing `(chunk_seed, future)` explicitly.
- **`filter_skewed`'s `--out`/`--holdout-out` path resolution** (from step 8, caught in step 10)
  -- silently wrote two directories above the repo root when given a `../../`-prefixed path.
- **Three `evaluation/*/evaluate_features.py` scripts** imported from `train_utils.py` files
  deleted in step 10's cleanup -- caught by grepping `evaluation/` specifically before declaring
  the cleanup done, not by the `dvc.yaml`/`models/` grep that covered everything else.

## Open items, explicitly not done

- **Task 12** (`dvc repro -P`, full reproduce + `dvc.lock` commit) — still not done. `dvc.lock`
  coverage is partial (a few stages per pipeline, from verification runs), not zero, but nowhere
  near complete. Needs a machine with DOSBox for the four `generate_gorillas` variants.
- **Task 4** (delete `train_linear_regression_sqrt.py`/`simulate.py`/`flowchart TD.mmd`), **4b
  second half** (three runs still hardcode their own sample size), **4c** (delete five superseded
  DOSBox builds) — Phase 2 incomplete.
- **Task 14** (move `_clean_iqr`/`_clean_range` behind the split — risk #5) — untouched, the last
  open correctness risk.
- `run_leakage`/`run_bias_variance`'s `train_utils.py` still call `add_engineered_columns`
  directly rather than through a Pipeline step — deliberately out of scope for step 10's cleanup
  (folder-local config, not the shared-script ambiguity C1 was about).
- `models/regression/train_linear_regression_sqrt.py` still imports from a deleted
  `train_utils.py` — left broken on purpose: it's the pre-existing orphan task 4 already covers,
  unreferenced by any `dvc_models_regression.yaml` entry and unreachable by any DVC stage.
- Two commits on `main` (the `train_utils.py` cleanup + the AUDIT.md update) were **not pushed**
  as of session end — not explicitly requested this round, unlike the earlier merge.

## Repo-specific gotchas worth remembering (carried forward + new)

- `PIPENV_IGNORE_VIRTUALENVS=1` still required every session — this machine auto-activates an
  unrelated project's venv (`V-and-V-*`) unless forced.
- **`python -m package.module` resolves against the current working directory, not the script's
  own location.** DVC always runs a stage's `cmd:` with cwd set to the directory of the
  `dvc.yaml` that declares it. A stage two directories under the repo root
  (`pipelines/<name>/dvc.yaml`) running a bare `python -m models.regression.train ...` fails with
  `ModuleNotFoundError: No module named 'models'` even though the identical command works from
  the repo root. Route through `scripts/run_with_pythonpath.py <pythonpath_dir> -m module ...`
  with the repo root as `<pythonpath_dir>`.
- **DVC's per-stage `params:` list is a separate mechanism from `vars:`.** `vars:` only controls
  `${...}` template interpolation at parse time; a stage's own `params:` (staleness tracking,
  `dvc exp show`) defaults to a `params.yaml` *next to that stage's own dvc.yaml* unless given an
  explicit file: `params: - ../../params.yaml: [key, ...]`. Symptom: `dvc repro` errors
  immediately with "Parameters '...' are missing from 'params.yaml'" — the fix is always making
  the file explicit, never something wrong with the keys themselves.
- **A relative CLI argument passed to a script is not automatically cwd-relative.**
  `data_generation/io.py`'s `resolve_output()` anchors relative paths to its own hardcoded
  `Path(__file__).resolve().parent.parent` — always the true repo root, regardless of cwd. A
  `../../`-prefixed path (correct for `deps:`/`outs:`/`cmd:` paths, which DVC resolves relative
  to the calling `dvc.yaml`) is *wrong* for this script's own arguments and silently writes
  outside the repo. Check what a script's own argument parsing actually does before assuming
  DVC's path-resolution convention applies to it too.
- **`--glob` on `dvc repro` is unreliable** at the sub-stage-name scope this repo needed — three
  tested patterns each matched a different, mostly-wrong set of stages. Prefer explicit target
  lists or separate `dvc.yaml` files.
- **Flag names are not consistent across `dvc` subcommands.** `dvc stage list`'s "every pipeline"
  flag is `--all`; `dvc repro`'s is `-P`/`--all-pipelines`. `dvc dag` (no target) finds stages
  across every `dvc.yaml` by default with no flag needed at all, unlike `repro`. Check `--help`
  per subcommand rather than assuming.
