# Session: `plot_landing_sites.py` -- scatter of throw landing sites (2026-09-28)

Small, self-contained session: one new script, no pipeline/data changes. Started from a plain
question about `simulate.py`, ended with a reusable landing-site scatter plot living under
`evaluation/data/`.

## What happened, roughly in order

1. **"Is there a simple projectile simulation with plotting script?"** -- confirmed `simulate.py`
   (root) is exactly that: single-throw animation via `physics.py`'s `simulate()`.

2. **"Is there a place to see a scatterplot of all the throw landing sites on an xy plot?"** --
   searched `evaluation/regression/`; found the closest thing (`run_*/plot_training_data_relationships.py`)
   plots the regression target against *input features*, never landing site in physical space
   (`landing_distance_m` vs `landing_height_m`), and none of the existing scripts take a
   dataset/run as an explicit argument -- each hardcodes which file it reads. Confirmed via
   `params.py`/`dvc_datasets.yaml`/`models/regression/common/loader.py` that there's no existing
   "pick a run via CLI" precedent anywhere under `evaluation/` (zero `argparse` hits).

3. **Plan-mode design** (one Explore agent + direct reads of `params.py`, `dvc_datasets.yaml`,
   `data_generation/contract.py`, `models/regression/common/loader.py`, `dvc.yaml`,
   `data_generation/join.py`). Two decisions confirmed with AskUserQuestion:
   - Plain single-color scatter (no hit/miss or outlier split).
   - `--data PATH` (a literal parquet path) rather than a `--dataset NAME` resolved against
     `dvc_datasets.yaml` -- chosen because the repo is mid-refactor (`git status` showed the
     `pipelines/` split and dataset-join work in progress; `data/` doesn't fully match
     `dvc_datasets.yaml`'s keys yet), so a literal path is robust to whatever files actually exist.
     Defaults to `params.data_path()` when omitted, matching every sibling script's convention.

4. **Built and verified** `evaluation/regression/plot_landing_sites.py` (its first location):
   `check_contract` on read (same validation every training loader applies), `--max-points`
   random subsample via `numpy.random.default_rng` so a 40k+ row pool doesn't overplot, `--save`
   to write a PNG instead of `plt.show()`. Verified against both `data/standard_training_data.parquet`
   (50k rows, subsampled) and `data/gorillas_effort_regression.parquet` (5k rows, no subsampling
   path) by actually rendering and reading back the PNGs.

5. **"Set the max/min x and y to the gameboard min and max x and y."** Traced the real board
   extent from `qbasic_gorillas/dosbox-datagen/gorilla.bas`: batch data-gen always runs in Mode 9
   (`SCREEN 9`, 640x350 EGA) -- `ScrWidth = 640`, `GroundY = BottomLine = 335`,
   `MetersPerPixel# = .2` (confirmed against the same constant in `data_generation/gorillas.py`).
   Board = 128 m wide x 67 m tall, ground-to-screen-top. Added `BOARD_WIDTH_M`/`BOARD_HEIGHT_M`
   constants and `ax.set_xlim(0, BOARD_WIDTH_M)` / `ax.set_ylim(0, BOARD_HEIGHT_M)`, replacing
   autoscale. Verified this clips the dataset's injected `gravity`/`data_error` outlier rows
   (previously stretching the y-axis to 650 m) out of view on both datasets, leaving the genuine
   throw data filling the frame.

6. **Moved the script** from `evaluation/regression/plot_landing_sites.py` to
   `evaluation/data/plot_landing_sites.py` (new folder) on request -- the script is dataset-driven
   rather than regression-specific, and `evaluation/data/` didn't exist before. `REPO_ROOT`'s
   `../..` path is unchanged (same nesting depth); re-verified it still runs and finds `params.py`
   correctly from the new location.

## Where it ended up

`evaluation/data/plot_landing_sites.py`:
```
python evaluation/data/plot_landing_sites.py [--data PATH] [--max-points N] [--seed N] [--save PATH]
```
No `__init__.py` added to `evaluation/data/` -- nothing imports the script as a module, it's
run-only, same as `data_generation/gorillas.py`'s CLI-only design.

## Not done / worth knowing

- Uncommitted: this is new, untracked work on top of an already-uncommitted mid-refactor working
  tree (see `git status` at session start -- `pipelines/` split, `dvc.yaml`/`dvc_datasets.yaml`
  changes, `data_generation/join.py`). Nothing here was committed.
- The board-bounds constants (`BOARD_WIDTH_M`/`BOARD_HEIGHT_M`) are duplicated by hand from
  `gorilla.bas` rather than read from a shared Python constant -- there wasn't one to reuse
  (grepped `*.py` for `640`/`BottomLine`/`ScrWidth`, no hits). If the game build's screen mode or
  `GroundY` ever changes, this script's axis limits would silently go stale along with it.
- Offered but not requested: a `--clean` flag to filter to `is_outlier == "none"` instead of (or
  in addition to) clamping the axes. User went with axis clamping only.
