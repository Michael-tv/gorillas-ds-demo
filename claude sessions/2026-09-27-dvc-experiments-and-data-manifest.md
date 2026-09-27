# Session: DVC experiment management + dataset-generation manifest (2026-09-13 -> 2026-09-27)

Continues directly from `2026-09-15-ml-pipeline-refactor.md` -- that session
built the first `dvc.yaml`/`dvc_models.yaml`/`dvc.lock` (all still untracked
in git) with foreach-based training stages. This session started from "does
DVC have experiment management" and ended up reworking most of the pipeline
that answer depends on: params, data generation, loaders, and directory
layout. Read the prior file first; this one assumes its decisions as given
except where noted as superseded below.

## Starting point discovered this session

`dvc.yaml`/`dvc_models.yaml`/`dvc.lock` already existed (untracked) with
`train_raw_10k/20k/40k`, `train_eng_10k/20k/40k`, `train_classification_10k/
20k/40k` as separate foreach stages, each pointing at its own generated
`data/raw_Nk.parquet`. Metrics already tracked as `cache: false` CSV outs.
No `params.yaml` existed -- every model script hardcoded its own
`N_ITER`/`CV`/search grid.

## Decisions made, in order (each supersedes the last where they conflict)

1. **`params.yaml` + `params.py`** added: `test_size`, `n_samples`, and
   `search.<regression|classification>.<model>.{n_iter, cv}` (the
   `RandomizedSearchCV`/`GridSearchCV`/`*CV` budget every script was
   hardcoding). `dvc_models.yaml` entries got a `model:` field mapping each
   foreach key (e.g. `random_forest_no_outlier`) back to its base algorithm
   name for params lookup.

2. **Convergence experiment (10k/20k/30k/40k) -- design corrected mid-session.**
   First proposal: a `data_generation/slice.py` DVC stage producing
   `raw_10k.parquet`/`raw_20k.parquet` as cached prefix-slices of a pool.
   **User rejected this**: "the selection of the rows to use should be done
   in the sklearn pipeline." Final design: `n_samples` is a `params.yaml`
   value; `loader.load_data(path, n_samples=...)` slices in Python at load
   time; sweeping tiers is `dvc exp run --set-param n_samples=X`, not
   separate cached files.

3. **Single loader, feature engineering moved into the model's own
   `sklearn.Pipeline`.** User: "there should only be a single loader called
   loader, the feature engineering should all happen in sklearn pipelines in
   the models." `models/regression/common/{raw_loader.py,eng_loader.py}` ->
   one `loader.py` that always returns the 9 raw columns. The already-written
   but never-wired `feature_engineering.EngineeredFeatures` transformer
   became each `run_eng_*` script's first `Pipeline` step;
   `run_raw_*`/`train_utils.py` uses `("engineer", "passthrough")` instead.
   Classification's loader kept its own inline engineering (it never had a
   raw/eng split to begin with) -- left as a judgment call, not acted on.

4. **Train/test split moved out of the loader into each training script.**
   User: "shouldnt the train test split section be moved to the training
   script for each model?" -- `load_data()` now returns `(X, y)`; each script
   calls `train_test_split` itself (with `stratify=y` for classification).
   Rationale settled during the discussion: outlier/range cleaning and
   `n_samples` slicing are row-count-changing and can't be Pipeline steps
   (a `Pipeline.transform` must preserve row correspondence with `y`, which
   never passes through `transform()`); splitting isn't a Pipeline concept at
   all. Feature engineering (row-count-preserving) is the only piece that
   correctly belongs in the Pipeline.

5. **Every model script's estimator wrapped in a `Pipeline` starting with
   `FEATURE_STEP`**, `model__`-prefixed `param_dist` keys added where a bare
   `RandomizedSearchCV(estimator, ...)` used to be called directly
   (`random_forest`, `xgboost`, `decision_tree` for both domains;
   classification kept no `FEATURE_STEP` per (3)).

6. **Per-size directories collapsed.** `run_raw_10k/20k/40k` ->
   one `run_raw/`; same for `run_eng_*` and classification's `run_10k/20k/
   40k` -> `run/`. Each shim now reads `N_SAMPLES = params.load_params()
   ["n_samples"]` instead of a hardcoded per-directory constant -- this
   *is* the convergence-experiment mechanism from decision 2.

7. **Output directories renamed to stop colliding with source trees.**
   `regression/`, `classification/` (top-level, holding `.joblib`+metrics)
   -> `experiments/regression/`, `experiments/classification/`, since they
   sat one path segment away from `models/regression/`, `models/
   classification/` (source code) and had started actively colliding once
   `run_raw`/`run_eng` existed under both. Every `MODELS_DIR` in
   `train_utils.py`/`predict.py`/`compare_models.py` and every `dvc.yaml`
   `outs:`/`metrics:` path updated.

8. **Data generation: one seeded pool, not one file per size.**
   `generate.py` shuffles rows before writing (rows come out of
   `generate_rows` grouped by section -- normal/gravity-outlier/data-error --
   so an unshuffled prefix would contain zero outliers) and gained a
   `--seed` flag. Pool named `data/standard_training_data.parquet` per
   explicit user naming request (went through `raw_pool.parquet` ->
   `base.parquet` -> this). `run_raw`/`run_eng`/classification's `run` all
   slice `n_samples` rows from this one file instead of three independently
   generated ones -- makes the tiers genuinely nested samples of one draw,
   not three independent draws with confounded resampling noise.

9. **`train_skewed` reworked: filter, not a separate generation.**
   Previously an independently-generated `data/skewed.parquet`
   (`generate_skewed` stage, `elevation_dist=(15, 8)` Gaussian draw). User:
   "train_skewed should use the same standard data but pre filters out
   certain rows." Now `run_skewed/train_utils.py` reads
   `standard_training_data.parquet` and filters `launch_angle_deg <= 30`
   (`MAX_ELEVATION_DEG`). This is actually *more* correct than the old
   mechanism -- the code's own docstrings already claimed "5-30 degrees"
   training data, but a Gaussian(15, 8) draw still had probability mass
   above 30; the hard filter makes the documented claim true.
   `generate_skewed` stage and `data/skewed.parquet` removed entirely.

10. **Dataset-generation manifest pattern**, mirroring `dvc_models.yaml`.
    User: wants a `generate_data` function plus a script/config listing
    every dataset to produce by calling it with different parameters, and
    asked how to make DVC rerun only the changed dataset. Answer implemented:
    `dvc_datasets.yaml` (dict of dataset-name -> `generate()` kwargs);
    `dvc.yaml`'s `generate` stage became `foreach: ${datasets}`, expanding to
    `generate@<name>` per entry -- each instance's `cmd:` embeds only its own
    entry's values, so DVC hashes/caches each independently (this is the
    direct answer to "how do I make it only rerun the changed dataset": it's
    a property of `foreach` expansion + per-stage `cmd:` hashing, not
    something requiring extra config). `generate_all.py` rewritten to read
    the same manifest instead of its own hardcoded `JOBS` list -- one source
    of truth for both `dvc repro` and manual regeneration.

11. **`raw_30k.parquet` gap closed.** `train_leakage_*`/`train_bias_variance_*`
    depended on `data/raw_30k.parquet`, which no stage ever produced (a
    longstanding dangling dependency, confirmed visible as disconnected
    islands with zero edges in a `dvc dag --mermaid` dump). Both already did
    `df.sample(n=N_SAMPLES, random_state=seed)` (10k/20k) from whatever file
    `DATA` pointed at, so the fix was purely the source path: point at
    `standard_training_data.parquet` directly, delete `raw_30k.parquet`
    and the on-demand-generation comments referencing it.

## Bugs found and fixed (not pre-existing intent, actual fixes)

- **`dvc_datasets.yaml` said `n: 50000`, the actual pool file had 40,000
  rows.** User edited the manifest via IDE after I'd manually regenerated
  the file at n=40000 for smoke-testing; no `dvc.lock` existed to catch the
  drift (the old one was deleted for referencing pre-refactor stage names
  that no longer exist). Fixed with `dvc repro generate`, which also created
  the **first real `dvc.lock`** for the current pipeline shape -- `dvc
  status` is only precise (names the specific changed param) from this point
  forward; before it, everything reports stale because there's no baseline.
- **`evaluation/{regression,classification}/simulate_predict.py`** built the
  model-existence-check path wrong: `os.path.join(RUNS_ROOT, ver,
  f"model_{model_name}.joblib")` (i.e. under the `evaluation/` tree) instead
  of the actual `experiments/.../models/` location -- always failed the
  `os.path.exists` check silently. Fixed to use `mod.MODELS[model_name]`
  (the loaded `predict.py` module's own, correct, dict) instead of
  reconstructing the path.
- **`run_eng`'s `predict.py`** (and classification's, structurally) used to
  build an already-engineered 5-column array and feed it straight to
  `model.predict()`. Once the model became a `Pipeline` expecting raw
  9-column input (decision 3), this would have silently fed the
  `EngineeredFeatures` step's `add_engineered_columns()` call a DataFrame
  missing the raw columns it needs. Rewritten to build a raw-column
  `DataFrame` (same shape as `run_raw`'s `predict.py`) and let the
  `Pipeline` do the engineering -- both `predict.py`s now take *identical*
  raw physical arguments, differing only in which `MODELS_DIR` they load.

## Verification performed (not just written-and-assumed)

Every layer was actually exercised against real generated data, not just
read back: loader slicing/cleaning (`no_outlier` dropped 308/10000, `range`
dropped a different 185/10000, confirming they're independent filters);
`FEATURE_STEP` passthrough vs. `EngineeredFeatures` on the identical
post-split `X_train`; a `Pipeline`-wrapped `RandomizedSearchCV` for both a
tree-based model (needed the `model__` param-key prefix) and a
already-Pipeline model (knn); classification's stratified split + search;
the `decision_tree_overfit` bypass path (doesn't call `load_data` at all,
needed its own `N_SAMPLES` slice); `train_skewed`/`train_leakage_clean` on
their new data sources; `dvc dag`/`dvc params diff` after every structural
`dvc.yaml` change (catches foreach/params-path typos across all ~90 stage
instances that a single script test can't); a full `dvc repro generate`;
and a complete generation-to-inference walkthrough (raw row -> sliced/cleaned
-> split -> Pipeline feature step -> search+fit -> persisted `.joblib` ->
loaded in `predict.py` -> scored a brand-new input, 62.41 m/s).

## Open items, explicitly not done

- `run_skewed`/`run_leakage`/`run_bias_variance`'s bespoke scripts
  (`train_skewed.py`, `train_balanced.py`, `train_clean.py`, `train_leaky.py`,
  `train_underfitting.py`, `train_overfitting.py`) still call
  `add_engineered_columns()` directly rather than going through a `Pipeline`
  step -- deliberately out of scope (fixed pedagogical demos, not swept via
  foreach/search), not touched.
- Whether to inline classification's `common/loader.py` into
  `classification/run/train_utils.py` (it's no longer shared by multiple
  run-dirs, unlike regression's) was raised and left as the user's call.
- Physically splitting `dvc.yaml` into per-domain files (e.g.
  `models/regression/dvc.yaml`) was discussed as an alternative to scoped
  `dvc dag`/`dvc repro` calls; user's follow-up was answered by demonstrating
  the graph is *already* two disconnected legs (no cross-edges), not by
  doing the physical file split. Single `dvc.yaml` still.
- No full `dvc repro` of the whole pipeline (~90+ stage instances, most
  running a real `RandomizedSearchCV`) has been run -- only `generate` plus
  individual `@key`-scoped smoke tests. Not requested.
- Nothing from this session is committed to git. `dvc.yaml`,
  `dvc_models.yaml`, `dvc_datasets.yaml`, `params.yaml`, `params.py`, the new
  `loader.py` files, and every touched `train_*.py`/`predict.py` are
  untracked/modified in the working tree.
- Published a grouped Mermaid DAG diagram as a Claude Artifact ("Pipeline
  Legs") since `dvc dag`'s default ASCII renderer is unreadable past ~50-70
  nodes and DVC has no native stage-grouping concept -- it was hand-clustered
  from the real edge list, not generated by a tool; will go stale if the
  pipeline's stage set changes without re-publishing it.

## Repo-specific gotchas worth remembering (carried forward + new)

- `PIPENV_IGNORE_VIRTUALENVS=1` is still required every session (see prior
  file) -- this machine auto-activates an unrelated project's venv.
- `dvc dag`'s default ASCII renderer degrades into unreadable box-art past
  roughly 50-70 nodes, which this pipeline exceeds by default (foreach
  expansion across `regression_models`/`skewed_models`/`classification_models`).
  Use `dvc dag <target>` scoped to a specific stage, or `--mermaid`/`--dot`
  for anything at or above this pipeline's scale.
- Every foreach-expanded stage that reads a **global** `params.yaml` key
  (`n_samples`, `test_size`) reruns *every* variant in that foreach block
  together when the key changes -- e.g. changing `n_samples` and running
  `dvc repro`/`dvc exp run train_raw` retrains all ~20 `regression_models`
  entries, not one model. Target `stage@key` explicitly (e.g.
  `train_raw@random_forest`) to scope a one-model experiment.
