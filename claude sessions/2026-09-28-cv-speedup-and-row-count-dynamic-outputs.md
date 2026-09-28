# Session: standard-pool CV speed-up, TIERS-to-params.yaml fix, row_count dynamic outputs (2026-09-28)

Same day as `2026-09-28-audit-pipelines-split-train-utils-cleanup.md` and
`2026-09-28-landing-sites-scatter-plot.md`, a separate conversation. Started from "investigate
removing cross-fold validation to speed up runs" and grew into three linked, separately-committed
fixes as each one surfaced the next problem.

## What happened, roughly in order

1. **CV investigation, not a decision taken on faith.** Explored where CV is actually used
   (`RandomizedSearchCV`/`GridSearchCV`/`RidgeCV`/`LassoCV`, all wired through `splitting.cv_for()`)
   and whether it's statistically load-bearing. Verdict, backed by data characteristics not
   opinion: the standard pool (50,000 i.i.d. rows off a smooth low-noise physics function) gets
   no real benefit from 5-fold averaging over a single validation split; the Gorillas pools
   (`effort`/`velocity`) genuinely need `cv_for()`'s group-aware folds, because a board's 32
   throws share wind/skyline and group-aware CV is what stops leakage across a split, not just a
   variance estimate. User's framing: fast for the basics, keep a "before" comparison, and
   they'd add a noisier experiment later (not done this session — their own follow-up).

2. **Standard-pool CV fix, commit `1cc846f`.** Added `splitting.single_split_cv()`
   (`ShuffleSplit(n_splits=1)`), swapped it in at every standard-pool call site (`experiment_raw`,
   `experiment_eng`, `experiment_row_count`, `experiment_skew` × regression, `experiment_classification`
   + `experiment_row_count` × classification — 42 files), removed `linear_regression.py`'s
   `cross_val_score` diagnostic (printed a CV-RMSE that never influenced the model), and added
   `experiment_cv_baseline/` — a verbatim copy of `experiment_raw` that deliberately keeps the
   original 5-fold search, as a runnable before/after. Verified with a real timed run, not just a
   code read: `experiment_raw`'s random forest went 6m05s → 1m12s with near-identical test MAE
   (6.44 → 6.42). Effort/velocity confirmed untouched by grepping for the new helper's name under
   their trees — it appears nowhere there.

3. **The effort row_count bug, found by the user running `dvc repro` themselves,** not by this
   session — they pasted the failure directly. `pipelines/effort/experiment_row_count/dvc.yaml`
   declared a `model_linear_regression_n2000.joblib` output; the script's `TIERS` was
   `[500, 1000, 2500, 5000, 10000, 20000]` — no `2000`, and two tiers (`10000`/`20000`) silently
   reprocessed the same 5,000-row pool twice since `effort`/`velocity` never had more than 5,000
   rows.

4. **TIERS-to-per-experiment-params.yaml fix, commit `77cec19`.** Rather than just fixing the one
   number, moved `TIERS` out of a hardcoded Python constant into a `params.yaml` colocated with
   each `experiment_row_count/` folder (new: `params.load_experiment_params(script_file)` in
   `params.py`), across all 45 files (regression ×9 algorithms, classification ×6, × 3 pools).
   Scoped deliberately narrow — user chose "row_count only, for now" over "every experiment gets
   a params.yaml" when asked directly, since nothing else currently needs per-experiment config.

5. **User pushed further: "is it required to hand-sync `outs:` against `tiers:`?"** Answer: yes,
   still fragile — DVC's `outs:`/`metrics:` are a hand-enumerated list, and neither `foreach` nor
   anything else auto-derives them from a params value. Explored whether a different orchestrator
   (Snakemake, specifically its `checkpoints`) would handle this natively; recommended against
   switching for this reason alone — the migration cost (rewriting 16+ pipeline definitions,
   losing DVC's git-native versioning/`dvc exp show`) dwarfs the problem it would fix.

6. **`foreach`-per-tier design, built out in a plan file, then rejected before implementing.**
   Would have made `outs:` genuinely dynamic (one DVC stage per tier), but at real cost: ~7x more
   stage instances repo-wide, rewriting all 45 scripts to `argparse` instead of looping
   internally, a new finalize/combiner stage per algorithm, and unverified DVC-syntax risk
   (stage-scoped `vars:` to avoid regression/classification's separate `params.yaml` files
   colliding on the same `tiers` key in one combined `dvc.yaml`; directory-level `deps:` spanning
   multiple stage instances). Mid-design, the user separately suggested colocating `dvc.yaml`
   with the experiment's own folder (out of `pipelines/`) — investigated as genuinely clean for
   13 of 16 experiments, more work for the 3 mixed-domain `experiment_row_count` files — but this
   whole direction was set aside once a simpler mechanism was found (next step). **Not done, on
   purpose**: dvc.yaml colocation was never implemented this session.

7. **The actual fix: directory-level `outs:`, commit `938bf4a`.** DVC supports declaring a whole
   directory as one `outs:` entry — tracks whatever's in the folder, no per-file list. Kept every
   script's internal `for n in TIERS:` loop and one-stage-per-algorithm shape exactly as they
   were; two small changes instead: each algorithm's `.joblib` files move into their own
   `tiers/<key>/` subdirectory (so one stage's directory-out can't claim another algorithm's
   files), and each script deletes any tier file no longer in `TIERS` before training — that's
   the actual "remove old outputs on rerun" fix, which `foreach` alone would not have given for
   free either (DVC never auto-deletes a removed stage instance's old output). Validated on one
   representative case first (`regression/effort/linear_regression`) — added a tier, confirmed
   the new output appears with zero `dvc.yaml` edits; removed a tier, confirmed the old
   `.joblib` and its metrics row are gone; restored the original list, confirmed identical
   values to the first run — before replicating to the other 44 scripts and 3 `dvc.yaml` files.
   Then ran the full `standard` pool (all 15 stages) end to end via `dvc repro`, not just the
   representative case, to confirm the rollout at scale.

8. **README + this file's first version written and pushed** (commit `d4e26f0`) — documenting
   steps 1-7. Then the user, working from the results, asked two follow-up questions that led to
   step 9: how to see `experiment_row_count`'s results (answered: `metrics_<key>.csv` per
   algorithm, `dvc metrics show` works since it's declared as `metrics:`; flagged that nothing
   currently *plots* it — `compare_models.py` in both `evaluation/` trees never references
   `experiment_row_count`), then whether outliers are excluded there.

9. **Outlier cleaning added to `experiment_row_count`, not yet committed.** Investigation first:
   `loader.load_data()` already supported a `clean` parameter (`"iqr"`/`"range"`/`"no_outlier"`)
   for regression, but **nothing in the whole `models/` tree actually passed it** — confirmed by
   grepping for `clean=` repo-wide. So no experiment currently cleans, not just `row_count`, and
   the README's "Does cleaning help?" section describing a `dvc_models_regression.yaml`-driven
   demo is stale, left over from the pre-`train_utils.py`-cleanup system. User chose
   `"no_outlier"` (the dataset's own ground-truth `is_outlier` label) over `"range"`/`"iqr"` when
   asked directly. Classification's loader had **no** `clean` mechanism at all (regression-only) —
   added a matching `_clean_no_outlier`/`clean=""` parameter to
   `models/classification/common/loader.py` before wiring it in, so both domains' 45
   `experiment_row_count/train_*.py` scripts now call `loader.load_data(DATA,
   clean="no_outlier")`. Verified live: standard pool loses 1,500/50,000 rows (3%) in both
   domains; effort loses 600/20,000 (same 3%, against its currently-WIP-inflated pool size).
   **Real issue found, not glossed over**: standard's top row_count tier (`n_samples=50000`)
   silently trains on only 48,500 rows once cleaning removes 1,500 — confirmed by a live run
   completing with no error. This is exactly the silent-substitution pattern
   `params.take_samples()`'s own docstring says this codebase already fixed once (for Gorillas
   pools vs `n_samples: 40000`) — pointed out explicitly rather than left implicit. Clarified for
   the user that cleaning already happens *before* the tier `.iloc[:n]` slice (inside
   `load_data()`), so reordering wouldn't help — the constraint is the *total* clean-row count,
   not sequencing. **User's chosen fix: increase the number of raw samples generated**, so the
   post-cleaning pool still reaches 50,000+ rows — their own follow-up, not done this session.
   **As of this save, the `clean="no_outlier"` code change is uncommitted** (see Open items) — it
   works correctly as written, but committing it now would bake in the known top-tier-shortfall
   caveat rather than wait for the user's data-regeneration fix.

10. **Answered where dataset selection lives**, prompted by the user opening `dvc_datasets.yaml`
    directly: that file only defines what datasets *exist* and their generation params, not which
    one an experiment *reads*. The standard pool is switchable via `params.yaml`'s
    `training_data:` key (`params.data_path()`); `effort`/`velocity` are **not** switchable at
    all — each hardcodes its own `DATA = .../data/gorillas_effort.parquet` (or `_velocity`)
    constant per script, by design (permanent pipelines).

## Bugs found and fixed

- **`effort`'s row_count `TIERS`/`dvc.yaml` `outs:` mismatch** (see step 3) — the trigger for
  this whole session's back half.
- **Stale flat-layout `models/model_<key>_n<tier>.joblib` files**, left over first from before
  step 4's fix and then again from before step 7's fix, sitting alongside the new
  `tiers/<key>/model_n<tier>.joblib` layout — the user spotted this directly in the VS Code file
  tree ("why is there models under models as well as under tiers") and asked; confirmed via
  mtimes (stale files predated the relevant script edits by hours) before deleting.
- **A regex-heredoc escaping quirk, not a repo bug but worth remembering** (see gotchas below).
- **No experiment anywhere actually applied outlier cleaning**, despite `loader.py` supporting
  three methods (`iqr`/`range`/`no_outlier`) and the README describing a cleaning demo — every
  `load_data()` call repo-wide omitted `clean=`, defaulting to no-op. Not fixed repo-wide this
  session (out of the asked scope), only for `experiment_row_count`; the README's stale
  "Does cleaning help?" section describing the old demo was flagged but not corrected.

## Open items, explicitly not done

- **`velocity`'s row_count wasn't verified with a live run** — `data/gorillas_velocity.parquet`
  is missing (pre-existing drift, unrelated to this session — confirmed via `dvc status` showing
  it deleted). Only checked `params.load_experiment_params()` resolves the right tier list
  directly; the actual training/cleanup code path on velocity is unexercised.
- **`effort`'s row_count results were excluded from both commits.** A separate, still-uncommitted
  `dvc_datasets.yaml` change (`gorillas_effort.n: 10000 → 20000`, not made by this session — found
  already dirty in the working tree) means any `dvc repro` touching `effort` regenerates
  `data/gorillas_effort.parquet` at 2x size. User's explicit call: leave that WIP alone,
  uncommitted, exactly as-is. So `effort`'s `metrics_*.csv`/`tiers/`/`dvc.lock` from this
  session's verification runs were reverted with `git checkout`/`rm -rf` before each commit
  rather than committed — the *code* changes for `effort` are committed and were verified to
  work correctly (add/remove-tier behavior confirmed via `regression/effort/linear_regression`,
  the representative case), just not with results baked from the currently-committed dataset
  size.
- **`dvc.yaml` colocation** (moving pipeline definitions out of `pipelines/` into
  `models/<domain>/<mode>/<experiment>/`) — discussed at length, investigated as feasible
  (13 of 16 experiments map 1:1, `experiment_row_count`'s 3 would need to split per-domain), but
  never implemented. If revisited: no other file references `pipelines/` by path except
  `README.md` and the moving files themselves (confirmed by a repo-wide grep), so the blast
  radius is contained to those two things.
- **AUDIT.md not updated this session** — only `README.md` and this file, per what was actually
  asked.
- **`experiment_row_count`'s `clean="no_outlier"` change (step 9) is uncommitted as of this
  save.** Code is written and verified correct across 46 files (45 `train_*.py` +
  `models/classification/common/loader.py`), but standard's top tier (`50000`) silently shrinks
  to ~48,500 rows once cleaning is on — the user is going to fix this at the source (increase
  generated sample count) before this should be committed as final. Don't commit as-is without
  checking whether that data-regeneration happened first, or the tiers list still needs a
  companion fix (lower the top tier to fit, discussed but not chosen).
- **The README's "Does cleaning help?" section is stale** (see bugs above) — describes a
  `dvc_models_regression.yaml`-driven demo (`random_forest_no_outlier`) that doesn't match how
  `clean` is actually wired today (an explicit `load_data(..., clean=...)` argument per script,
  used by nothing before this session). Worth a README pass if anyone builds on step 9.

## Repo-specific gotchas worth remembering (new this session)

- **DVC directory-level `outs:` works exactly as hoped, and is worth reaching for before
  `foreach`.** `outs: - some/dir/` tracks whatever's currently in that folder — confirmed via a
  live add-tier/remove-tier test, not just documentation. DVC also auto-manages a `.gitignore`
  inside the directory once it's DVC-cached (created on first `dvc repro`, not something to write
  by hand). It does **not** auto-delete files when a stage that used to produce them changes —
  the script itself still has to clean up anything it no longer wants there.
- **Running `dvc repro` on one pipeline can regenerate an unrelated upstream dataset if that
  dataset's params file is already dirty.** `dvc repro pipelines/effort/.../dvc.yaml` pulls in
  `generate_gorillas@gorillas_effort` as an upstream dependency; if `dvc_datasets.yaml`'s
  `gorillas_effort.n` is uncommitted-changed, DVC correctly (if surprisingly) regenerates
  `data/gorillas_effort.parquet` to match *before* running the stage you actually asked for. Check
  `git status` on `dvc_datasets.yaml`/root `dvc.lock` before and after any `dvc repro` on a
  Gorillas pipeline, not just the pipeline's own `dvc.lock`.
- **A python heredoc (`python3 << 'EOF' ... EOF` via the Bash tool) can silently mishandle
  backslash escaping in ways an actual `.py` file run the normal way does not** — a regex
  replacement string containing `\d`/`\.` inside a nested lambda worked when written to a real
  file and executed, but produced a different (wrong) result inline in a heredoc in this
  environment. When a regex transform's behavior looks inexplicable, rule this out by writing the
  exact same code to a scratch `.py` file and running that, before assuming the regex itself is
  wrong.
- **This machine's `python3` on PATH lacks the project's dependencies** (no `sklearn`, no
  `yaml`) — the actual venv with everything installed is
  `C:\Users\MichaelVictor\.virtualenvs\V-and-V-13OSdAjM\Scripts\python.exe` (also what `dvc`
  itself resolves to via `python -m dvc`). `py -3` resolves to a *different*, bare `C:\Python314`
  with `sklearn` but not `yaml`. Use the venv path explicitly for anything that needs to actually
  run a training script or `dvc repro`.
- **A specific `experiments_results/.../metrics_linear_regression.csv` file under `effort/`
  hit a persistent (not transient) `PermissionError` on write, repeatedly, across separate
  process invocations** — something external (editor, sync client, AV) had it locked for an
  extended period. Confirmed it wasn't a code bug: the script's own logic completed correctly up
  to that point every time (right MAE values printed for every tier), only the final file write
  failed. Don't assume a retry will clear a Windows file lock the way it often does for other
  transient issues — check whether something in the IDE/editor is specifically watching that file.
- **Dataset selection is split across two different mechanisms depending on pool type** — easy to
  point someone at the wrong one. `params.yaml`'s `training_data:` key only affects the
  *switchable* `standard` pipelines; `effort`/`velocity` are permanently wired to one Gorillas
  dataset each via a hardcoded `DATA = .../data/gorillas_<mode>.parquet` constant in every script,
  by design, and `--set-param training_data=...` does nothing for them. `dvc_datasets.yaml` is a
  third, unrelated thing again — it defines what datasets exist/how they're generated, not which
  one any given experiment reads.
