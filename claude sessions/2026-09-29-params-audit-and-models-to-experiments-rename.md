# Session: params.yaml audit, cv/n_iter moved to per-experiment params, models/ -> experiments/ rename, results/ split into models/+metrics/ (2026-09-29)

Started from "audit the parameters in the root params.yaml and remove all that are not used
anymore," which uncovered a real (if not immediately obvious) design flaw rather than dead
config, and grew through several rounds of user pushback into two large mechanical repo
restructurings.

## What happened, roughly in order

1. **First pipelines/ -> models/<domain>/<mode>/<experiment>/ colocation**, at the user's request
   ("I requested that all the pipelines be moved to their respective experiment directories" --
   a move discussed but explicitly *not* done in an earlier session per
   `2026-09-28-cv-speedup-and-row-count-dynamic-outputs.md`). Moved all 16 `pipelines/<mode>/
   <experiment>/dvc.yaml` files into `models/<domain>/<mode>/<experiment>/dvc.yaml`, splitting
   `experiment_row_count`'s three mixed-domain files into one-per-domain (regression +
   classification) since it was the only pipeline covering both. Deepened every relative path by
   one `../` (`../../../` -> `../../../../`). Verified via stage-name diffing (134 stages,
   identical set before/after) and real `dvc repro` runs across several shapes. README's whole
   "Running one pipeline at a time" section rewritten to match.

2. **params.yaml audit**: confirmed every top-level key (`training_data`, `test_size`,
   `n_samples`, `skew.max_angle_deg`) was genuinely used, but the `search:` block's `cv` sub-key
   was **not** what it looked like -- commit `1cc846f` (an earlier session) had switched the
   standard pool's hyperparameter search from `splitting.cv_for()` (k-fold) to
   `splitting.single_split_cv()` (a single validation split, no fold-count argument at all), which
   meant 6 of the 9 non-demo experiments read a `cv` value from params.yaml that was never
   consumed. Only effort/velocity (grouped CV protects against board-level leakage) and
   `experiment_cv_baseline` (a deliberate "before" 5-fold comparison) still genuinely used it. Also
   found the actual bug behind "the search block cannot be used": `train_linear_regression.py` (4
   files) and `train_logistic_regression.py` (2 files) on the standard pool were missed by
   `1cc846f` and still called `cv_for()` like their siblings used to -- cosmetic only (both
   algorithms' `fit()` ignore the search config entirely), but a real inconsistency the user caught
   from a stray IDE glance at the params file.

3. **`cv` moved out of the root params.yaml**, on explicit instruction ("any cv parameters etc
   should be placed in the relevant experiment params file"). New local `params.yaml` (or an
   addition to an existing one) in each of the 11 experiment folders that genuinely still call
   `cv_for()`: effort/velocity's `experiment_raw`/`experiment_eng`/`experiment_row_count`
   (regression + classification) and `experiment_cv_baseline`. One `cv: 5` scalar per folder
   (every existing non-null value in the old shared block was already 5). 87 `train_<model>.py`
   scripts updated to read it via `params.load_experiment_params(__file__)["cv"]`; the 6 straggler
   scripts fixed to `single_split_cv()` instead. Every affected `dvc.yaml`'s `params:` block
   updated to match (135 stale `search.*.cv` tracking lines removed, 73 local-file tracking lines
   added). Verified with real runs covering every shape (grouped CV, RidgeCV, stratified
   classification CV, row_count, the straggler fix producing byte-identical metrics).

4. **`n_iter` moved out too**, on the same instruction extended to "the rest of the parameters
   under search" (visible in a screenshot of the now cv-less `search:` block, still showing
   `n_iter` for every algorithm). Same mechanism, wider scope: all 17 experiment folders that call
   `params.search_params()` at all got a local `n_iter: {<algo>: <budget or null>, ...}` mapping
   (13 already had a local file from step 3 or from `experiment_row_count`'s pre-existing `tiers`;
   4 didn't and got one created). 135 scripts updated; `params.search_params()` deleted from
   `params.py` since nothing calls it anymore; root `params.yaml`'s `search:` block removed
   entirely. Same verification rigor (135 stages before/after, real runs across every shape).

5. **User pushed to disable grouped CV entirely for `experiment_row_count` on effort/velocity**
   ("what do I do if I don't want to run cv"). This one needed real pushback before acting: swapping
   `cv_for()` for `single_split_cv()` on a *grouped* pool (Gorillas throws share a board's wind and
   skyline) loses group-awareness in RandomizedSearchCV's *inner* validation split specifically,
   which is exactly what `cv_for()` exists to prevent -- explained the outer-split-vs-inner-split
   distinction, the mechanism (near-duplicate rows from the same board landing on both sides of the
   inner split), and was honest that the real-world magnitude was untested, not settled. **User's
   final call**: "the only split that is important is the train test split" -- and after that,
   "if I am correct, please remove the unnecessary complexity." Applied to all 30 scripts
   (regression + classification, effort + velocity): `cv_for()` -> `single_split_cv()`, the `cv`
   key removed from those 4 local params.yaml files and their `dvc.yaml` tracking (not just left
   unused -- actually deleted, per the "remove unnecessary complexity" framing). `experiment_raw`/
   `experiment_eng`/`experiment_classification`'s real grouped CV, and the outer train/test split
   everywhere, untouched.

6. **Two pre-existing untracked experiments discovered mid-session, unrelated to this work**:
   `experiment_baseline` and `experiment_bias_variance_bootstrap` (effort, regression), both
   already fully built and self-consistent with the new per-experiment-params conventions (neither
   searches, so neither ever touched `n_iter`/`cv`). Blocked for a long stretch on a genuine
   Windows file lock on `experiment_baseline`'s `metrics_linear_regression.csv` (`WinError 32`/
   `Permission denied`, confirmed via direct Python `open()` probes, not just retried `dvc repro`);
   cleared eventually and both were verified (real `dvc repro`, py_compile) and committed
   (`8da832a`). **This later turned out to be the commit that a *different*, concurrent Claude Code
   session -- see `2026-09-29-effort-bias-variance-baseline-and-bootstrap.md`, written from that
   other session's perspective -- flagged as mysterious and unattributed**, since it landed on the
   shared branch mid-way through that session's own work on the same two experiments. Neither
   session caused data loss; the confusion was two agents working the same repo without visibility
   into each other, each periodically seeing "file changed on disk since you last read it" for
   files the other had just touched. Also found and left alone, both times: a genuinely
   pre-existing broken orphan file, `experiments/regression/train_linear_regression_sqrt.py`
   (imports a `train_utils` module removed in an ancient session).

7. **`git add -A` mistake and correction**: the first `experiment_baseline`/bootstrap commit
   accidentally swept in a stray `models/regression/effort/experiment_baseline/` bundle that
   hadn't been reviewed yet, via a blanket `git add -A`. Caught immediately after the commit
   (`git show --stat HEAD` looked wrong), fixed with `git rm --cached` + `git commit --amend`
   before doing anything further -- the fix landed before any other work built on top of the bad
   commit. Lesson: `git add -A` right before a commit is risky in a working tree that might have
   *other* untracked content sitting around for reasons unrelated to the current task; diff the
   commit immediately after making it, not just the working tree before.

8. **`models/` renamed to `experiments/`; `experiments_results/` folded into each experiment's own
   `results/`**, on request ("rename the models folder in the root to experiments and move the
   experiment results to a results folder within each experiment"). `git mv models experiments`
   for the package; each of 15 `experiments_results/<domain>/<mode>/<experiment>/` subtrees moved
   into `experiments/<domain>/<mode>/<experiment>/results/` (file-by-file where the whole-directory
   `mv` hit the same class of Windows lock as step 6 -- `experiment_baseline` again). Every
   `models.<domain>.` / `models/<domain>/` / `experiments_results/<domain>/<mode>/<experiment>/`
   reference repo-wide (211 files) rewritten via a Python codemod matching on literal
   `regression`/`classification` tokens, plus ~30 hand-edited call sites that used variables or
   multi-line `os.path.join()` calls the codemod's regex couldn't safely touch (two
   `evaluation/*/compare_models.py`, three `analysis.py` scripts) and ~25 README/dvc.yaml/
   params.py/run_with_pythonpath.py mentions using a `<domain>` placeholder instead of a literal
   domain name. `AUDIT.md` and `claude sessions/` left alone throughout, per this repo's established
   convention for historical records.

9. **`results/` split into `models/` (actual `.joblib` files) and `metrics/` (actual `.csv`
   files)**, on request after a screenshot showed the confusing state the previous step's straight
   relocation had inherited: a folder literally named `models/` held `metrics_<key>.csv` (not
   models), and for swept experiments the real `model_n<size>.joblib` files were hidden a level
   away under a separate `tiers/<algo>/` branch. Fixed uniformly: `results/models/` now holds only
   `.joblib` files (`results/models/<algo>/` for swept experiments, replacing `tiers/<algo>/`);
   `results/metrics/` holds every `metrics_<key>.csv`. 150 scripts updated (104 flat-shape, 46
   tiered-shape) to add an explicit `METRICS_DIR` alongside `MODELS_DIR`; every `dvc.yaml`'s
   `outs:`/`metrics:` paths and every physical file on disk moved to match; the per-tier
   `.gitignore` DVC writes for a swept experiment's cached joblib subfolders moved from
   `tiers/.gitignore` to `models/.gitignore` (same per-algo patterns, just relocated).

## Bugs found and fixed (in earlier sessions' work, not new code from this one)

- The straggler `train_linear_regression.py`/`train_logistic_regression.py` cv_for() calls (step
  2) -- cosmetic, but a genuine leftover from an incomplete earlier refactor.
- `1cc846f`'s dvc.yaml `params:` blocks were over-tracking `search.<domain>.<algo>.cv` on every
  standard-pool stage even after that commit made the value unused there -- DVC would have marked
  those stages stale on a `cv` edit that could never actually change their output. Fixed as a side
  effect of step 3's relocation (removed everywhere it wasn't genuinely read).

## Open items, explicitly not done

- **The concurrent-session collision (step 6/7) was never really "resolved," just navigated.**
  Both sessions' work is now on the branch and verified non-conflicting, but there was no
  coordination -- worth telling the user directly that two Claude Code sessions appear to have
  been active on this same working tree around 2026-09-29, in case that was unintentional.
- **`experiments/regression/train_linear_regression_sqrt.py` still sits there, still broken.**
  Never asked about; left alone both times it was noticed (this session and, per its own notes,
  the concurrent one).
- **README's `experiment_row_count` tier-list example is stale** (`effort`'s/`velocity`'s tiers
  are documented as `[500, 1000, 2000, 3000, 4000, 5000]`, but at least the effort regression one
  was edited by the user mid-session to `[500, 1000, 2000, 4000, 8000, 16000]`) -- noticed, not
  fixed, since it was out of scope for the specific requests this session handled.
- **`dvc.lock` hashes for scripts whose only change was a renamed path in a comment** (e.g.
  `data_generation/generate.py`) are now stale relative to their recorded stage, same as every
  other refactor this session -- a real `dvc repro -P` would re-run them for no functional reason.
  Not forced through; left for whenever the user next does a full repro.

## Repo-specific gotchas worth remembering (new or reinforced this session)

- **A "shared root params.yaml key" is a smell worth checking, not just accepting, when different
  experiments consume it through genuinely different code paths** (`single_split_cv()` vs.
  `cv_for()` here). The same numeric value being *present* everywhere doesn't mean it's *read*
  everywhere -- confirm by tracing the call site, not by grepping for the key name.
- **`splitting.single_split_cv()` must never be substituted for `splitting.cv_for()` on a pool with
  real group structure** (Gorillas' board-level wind/skyline correlation) without the person who
  owns the tradeoff explicitly signing off -- it silently trades away leakage protection in the
  hyperparameter search's inner validation split specifically, not the outer train/test split,
  which makes it easy to miss in a quick review of "does the pipeline still work."
- **The Windows file-lock class of error (`WinError 32`/`Permission denied` on a `metrics_*.csv`
  or an enclosing directory) recurred multiple times this session and in the concurrent one** --
  confirmed genuinely OS-level (not a DVC transient) via a direct Python `open()`/`os.rename()`
  probe rather than blind `dvc repro` retries; falls back to moving files individually rather than
  the whole directory when the directory-level move is what's actually blocked.
- **`git add -A` immediately before a commit, in a working tree shared with other agents or
  sessions, needs a post-commit diff check** (`git show --stat HEAD`), not just a pre-commit
  `git status` read -- by the time `git add -A` runs, anything else sitting untracked in the tree
  is one keystroke from being swept in under the current commit's authorship and message.
- **When "file changed on disk since you last read it" notices don't line up with anything this
  session did, check `git log` for a commit this session never ran** before assuming corruption or
  a tool bug -- it may be a second agent on the same branch, as it was here (see the concurrent
  session's own notes on the same discovery from its side).
