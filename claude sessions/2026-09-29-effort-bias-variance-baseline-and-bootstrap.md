# Session: regression/effort baseline, bias/variance proxy + bootstrap decomposition, per-experiment analysis.py (2026-09-29)

Started from "add a baseline experiment to regression/effort" (default LinearRegression, 1000 rows,
no outliers) and grew into a full bias/variance investigation once the user noticed
`experiment_row_count`'s linear_regression results showed no real improvement from more data —
first a cheap proxy, then a real bootstrap decomposition, then per-experiment plotting scripts.

## What happened, roughly in order

1. **`experiment_baseline` added**, following `experiment_raw`'s one-script-per-model pattern:
   a single default `LinearRegression().fit()` on a fixed 1000-row slice of `gorillas_effort.parquet`
   with outliers excluded (`clean="no_outlier"`). Initially included a local `cv`/`n_iter` pair for
   consistency with other experiment scripts; **user pushed back** — "no cv for baseline, this model
   should be as simple and default as they come" — so `cv` was stripped entirely, along with the
   unused `search_params()`/`cv_folds` machinery. Lesson banked: for this repo, matching an
   established per-experiment pattern is not a good enough reason to carry config a model doesn't
   use — a genuinely parameter-free model should have a genuinely parameter-free script.

2. **User pasted `experiment_row_count`'s `metrics_linear_regression.csv`** (500→5000 rows) and
   observed no real improvement in MAE/RMSE as `n_samples` grows. Diagnosed this as linear
   regression's error being bias-dominated, not variance-dominated: a flat learning curve is the
   classic "more data won't help, the model's functional form is the bottleneck" signature, as
   opposed to a curve still declining at the largest tier.

3. **Explained why bias/variance isn't a standard reported ML metric** (asked directly): it's a
   property of the *fitting procedure* under repeated resampling, not of one fitted model, so a
   single train/test split can't measure it directly; the classic decomposition only holds cleanly
   for squared-error loss (0/1 loss has several competing, disagreeing decompositions); and it's a
   diagnostic for a decision (more data vs. more model flexibility), not an optimization target like
   MAE/accuracy.

4. **Cheap proxy added**: `train_mae` (bias signal) and `variance_proxy = mae - train_mae` (variance
   signal) added as extra columns to `experiment_row_count/train_linear_regression.py`,
   `train_decision_tree.py`, and `experiment_baseline/train_linear_regression.py`. Explicitly
   documented as a heuristic, not a real decomposition, in every docstring it touched.

5. **User asked for the practical bootstrap approach**, then referenced a GeeksforGeeks article
   (`bias-vs-variance-in-machine-learning`) doing the same thing. Fetched it (via WebFetch) to
   confirm it matched what had just been described independently: K bootstrap-resampled model fits,
   stacked predictions on one fixed test set, `bias2 = ((y_test - preds.mean(axis=0))**2).mean()`,
   `variance = preds.var(axis=0).mean()`. Also noted `mlxtend.evaluate.bias_variance_decomp` exists
   as a ready-made version of the same method.

6. **`experiment_bias_variance_bootstrap` built**: real bootstrap decomposition for
   `linear_regression` at the same six `n_samples` tiers as `experiment_row_count`, so the two
   experiments' numbers line up directly. N_BOOTSTRAP=30 (matching the article), one fixed
   group-aware test set per tier, one plain (non-bootstrapped) model also fit and saved per tier
   purely for inspectability. **Bootstrap resampling done at the `group_id` level, not per-row** —
   Gorillas throws are grouped by board (shared wind/skyline per `splitting.py`), so a naive
   row-level bootstrap would treat correlated throws as independent draws and understate the real
   variance. Implemented via a group→row-indices lookup, sampling group *positions* with
   replacement, reusing `splitting.split()`'s already-returned `groups_train` rather than
   recomputing group membership from scratch.

7. **Result, verified via a real `dvc repro` run**: bias² is ~99% of mean test MSE at every tier
   (33.7→45.0 m²/s², noisy/flat, not decreasing), while variance shrinks monotonically and exactly
   as theory predicts as `n` grows 10x (0.900→0.085). `noise_estimate` (`mean_test_mse - bias2 -
   variance`) is ≈0 at every tier, confirming the decomposition is internally consistent. This
   conclusively answers step 2's observation: linear regression on this pool is bias-limited, not
   variance-limited — more effort-pool data will not materially help it; only a more flexible model
   would.

8. **`analysis.py` added to all three experiments** (`experiment_baseline`,
   `experiment_row_count`, `experiment_bias_variance_bootstrap`), on request ("add an analysis.py
   script to each experiment ... used to read the metrics etc and generate plots"). Followed
   `evaluation/regression/compare_models.py`'s existing style: module-level script, no PYTHONPATH
   wrapper, no repo imports — just pandas + matplotlib reading the CSV(s) the experiment's own
   `dvc.yaml` already produces, console table + a saved PNG next to the script. Scoped to the three
   experiments this conversation built/touched, not the whole repo.

## Bugs found and fixed (this session, in my own new code)

- **`experiment_row_count/train_decision_tree.py`'s `csv.DictWriter` fieldnames list was missing
  `variance_proxy`** even though the row dict included it — crashed `dvc repro` with `ValueError:
  dict contains fields not in fieldnames` on the very next full run. Caught by actually running the
  pipeline (not just reading the diff back), fixed, reran successfully.
- **`experiment_row_count/analysis.py`'s original "Read" heuristic judged bias/variance from the
  raw test-MAE trend** (last tier minus first tier) rather than from `variance_proxy` itself. This
  misclassified `linear_regression` as "variance-limited (more data is helping)" because its test
  MAE bounces ±0.4 m/s tier to tier with no real trend — a noisy-but-flat curve that a naive
  first-vs-last check reads as movement. Caught by comparing the printed read against the bootstrap
  experiment's rigorous result (which says bias-limited) and finding they disagreed. Fixed to read
  from `variance_proxy`'s average share of mean test MAE instead; the two experiments' reads now
  agree (linear_regression: 4% share → bias-limited; decision_tree: 31% share → real variance role).

## Open items, explicitly not done

- **A concurrent Claude Code session appears to be committing to this same branch.** Mid-session,
  `git log` showed commit `8da832a` ("Add experiment_baseline and experiment_bias_variance_bootstrap
  (effort, regression)") had landed, authored as the user but with a `Co-Authored-By: Claude Sonnet
  5` line and a first-person commit message — something I never ran `git commit` to produce. Diffed
  the working tree against that commit and confirmed **zero drift** for the two experiments it
  captured (no data lost), but this explains the repeated "file changed on disk since you last read
  it" notices all session (row_count's `tiers`/`n_iter` move to per-experiment `params.yaml`, the
  `cv_for()`→`single_split_cv()` swap, etc.) — those were a different, concurrent session's own
  work, not corruption. **Flagged to the user; not resolved.** Nothing in this session was committed
  by me, per standing instructions — only that one commit exists, and it wasn't mine.
- **`experiment_bias_variance_bootstrap` only covers `linear_regression`**, by scope/cost choice —
  extending the same bootstrap method to `decision_tree` (or others) for comparison is a natural
  follow-up but wasn't asked for or done.
- **README not updated** to mention the three experiments or the new `analysis.py` convention —
  worth a pass if these become permanent fixtures, matching how `experiment_row_count`'s
  directory-`outs:` design got documented in README in an earlier session.
- **Working tree, as of this save**: `experiment_baseline` and `experiment_bias_variance_bootstrap`
  are already committed (by the concurrent session, commit `8da832a`). Still uncommitted: the two
  bug fixes above (`train_decision_tree.py`'s fieldnames list, `analysis.py`'s heuristic), the
  regenerated `metrics_decision_tree.csv`/`dvc.lock` that came from rerunning after the fieldnames
  fix, and the new `experiment_row_count/analysis.py` + `analysis.png` files.

## Repo-specific gotchas worth remembering (new this session)

- **The Windows file-lock issue on a specific `experiments_results/.../metrics_*.csv` recurred**
  (also hit and documented in `2026-09-28-cv-speedup-and-row-count-dynamic-outputs.md`, same class
  of file). Confirmed genuinely OS-level this time via a direct exclusive-open probe
  (`[System.IO.File]::Open(path, 'Open', 'ReadWrite', 'None')` in PowerShell) rather than just
  retrying `dvc repro` — the probe fails with the same `WinError 32` independent of DVC, confirming
  it's not a DVC-side transient. Cleared once the user closed whatever had it open. Don't burn
  retries on this class of error; check for an editor/preview tab on the *exact* file instead.
- **`git status`/`git add --dry-run` can look like they're under-reporting untracked files when the
  real explanation is that another session already committed them.** Several expected-untracked
  files (`experiment_baseline/__init__.py`, `params.yaml`, etc.) didn't show up in `git status
  --short -uall` or `git add -n` — not a git bug, but because a concurrent session's commit already
  captured them. `git ls-tree HEAD -- <path>` is what actually revealed this; worth reaching for
  directly (rather than re-running `git status` variants) when file state looks inconsistent with
  what you expect to have changed, especially with more than one agent/session possibly active on
  the same working tree.
- **Bootstrap resampling on a grouped Gorillas pool must resample whole groups, not rows** — see
  step 6. `splitting.split()` already returns `groups_train`; reuse it rather than re-deriving group
  membership.
- **`PYTHONIOENCODING=utf-8` is needed to run these scripts' box-drawing-character console tables
  from this Bash/git-bash-over-Windows-Python environment** — the default `cp1252` stdout encoding
  raises `UnicodeEncodeError` on `─` otherwise. Existing scripts like `compare_models.py` use the
  same characters, so this is a pre-existing environment quirk, not something new to work around in
  future scripts specifically — just remember to set the env var when running any of them by hand
  here.
