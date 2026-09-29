"""Generates the projectile-shot dataset by driving the real QBasic Gorillas
game via DOSBox, instead of physics.py's RK4 integrator -- both fill the same
contract. Worth the extra cost because the game's data has structure no
independent-marginal sampler produces: buildings truncate flights, wind is a
per-board integer shared across throws, and the skyline has real artefacts.

DOSBox emulates on a single thread, so throughput comes from running several
instances at once, each in its own worker directory mounted as D:.

Run modes: default refuses to overwrite; --overwrite replaces; --dry-run
validates without writing.
"""

import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from data_generation import generate as gen
from data_generation.contract import check_contract
from data_generation.io import resolve_output, write_parquet
from data_generation.outliers import corrupt_row
from data_generation.sampling import (
    DATA_ERROR_FRAC,
    GRAVITY_HIGH,
    GRAVITY_LOW,
    OUTLIER_FRAC,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
GAME_DIR = REPO_ROOT / "qbasic_gorillas" / "dosbox-datagen"

DEFAULT_GRAVITY = 9.8
HIT_TOLERANCE = gen.HIT_TOLERANCE

# The game reports the banana's fate as one of these. Only the three that
# actually produced a landing point are usable.
OC_EDGE, OC_TERRAIN, OC_GORILLA, OC_SELF, OC_GROUND, OC_MAXTICKS = 0, 1, 2, 3, 5, 6
KEEP_OUTCOMES = (OC_TERRAIN, OC_GORILLA, OC_GROUND)

# Fixed constants of the game build, needed to complete the mapping.
METERS_PER_PIXEL = 0.2
# CONST BananaCd# in gorilla.bas -- a real game constant, so this is zero-variance
# in every Gorillas pool (kept for contract parity with the Python producer).
BANANA_CD = 0.6

# Largest landing_height_m clip treated as float noise rather than a mapping bug.
# Anything above this fails the run -- see _report.
CLIP_TOLERANCE_M = 0.01

# Mirrors CsvOpenFile's header exactly. Validated on every parse: it is the one
# real coupling between gorilla.bas and this module, and a silent field-order
# change would corrupt the mapping without raising anything.
CSV_HEADER = [
    "board", "throw", "tosser", "tossee", "seed", "gravity_ms2", "buildings",
    "input_mode", "flags", "wind_ms", "wind_rel_ms", "angle_deg", "angle_sim_deg",
    "effort_pct", "actual_effort_pct", "force_n", "velocity_ms",
    "banana_mass_kg", "banana_diam_m", "banana_area_m2",
    "launch_px_x", "launch_px_y", "target_px_x", "target_px_y", "ground_px_y",
    "launch_height_m", "target_height_m", "target_dist_m", "target_dy_m",
    "land_dx_m", "land_dy_m", "land_px_x", "land_px_y",
    "flight_s", "outcome", "pointval", "timer_s",
]

DOSBOX_CANDIDATES = [
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "DOSBox Staging" / "dosbox.exe",
    Path(r"C:\Program Files\DOSBox Staging\dosbox.exe"),
    Path(r"C:\Program Files (x86)\DOSBox Staging\dosbox.exe"),
    Path(r"C:\Program Files (x86)\DOSBox-0.74-3\DOSBox.exe"),
]


class DosboxNotFound(RuntimeError):
    pass


class GameRunFailed(RuntimeError):
    pass


DOSBOX_ENV_VAR = "GORILLAS_DOSBOX"


def find_dosbox(explicit=None):
    """Locate the DOSBox executable.

    Resolution order, most explicit first: `--dosbox` argument, GORILLAS_DOSBOX
    env var, `dosbox` on PATH, then known install locations. The env var and
    PATH lookup matter because regeneration is this dataset's only recovery
    path -- there is no DVC remote holding a copy.
    """
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise DosboxNotFound(f"--dosbox given but not a file: {p}")
        return p

    from_env = os.environ.get(DOSBOX_ENV_VAR)
    if from_env:
        p = Path(from_env)
        if not p.is_file():
            raise DosboxNotFound(
                f"{DOSBOX_ENV_VAR} is set to {p}, which is not a file. Fix or "
                f"unset it, or pass --dosbox <path>."
            )
        return p

    on_path = shutil.which("dosbox") or shutil.which("dosbox.exe")
    if on_path:
        return Path(on_path)

    for p in DOSBOX_CANDIDATES:
        if p.is_file():
            return p

    searched = "\n  ".join(str(p) for p in DOSBOX_CANDIDATES)
    raise DosboxNotFound(
        f"DOSBox not found. Checked ${DOSBOX_ENV_VAR}, `dosbox` on PATH, then:\n  "
        + searched +
        "\nInstall DOSBox Staging (winget install DOSBoxStaging.DOSBoxStaging), "
        f"pass --dosbox <path>, or set {DOSBOX_ENV_VAR} to its full path so every "
        "later run finds it."
    )


def _short_path(p):
    """Windows 8.3 form of a path, when it has one.

    DOSBox's [autoexec] does not reliably handle a `mount` path containing
    spaces even when quoted -- the line gets split and the tail is executed as
    DOS commands. Worker directories therefore must not contain spaces, which
    the default temp location often does ("C:\\Users\\First Last\\...").
    """
    if os.name != "nt":
        return p
    import ctypes
    buf = ctypes.create_unicode_buffer(1024)
    if ctypes.windll.kernel32.GetShortPathNameW(str(p), buf, 1024):
        return Path(buf.value)
    return p


def _write_dos_text(path, text):
    """Write a DOS text file.

    CRLF is not cosmetic here. QBasic's LINE INPUT terminates on CR, so an
    LF-only config is read as a single enormous line: the first key parses and
    every other setting silently keeps its default, which looks like the config
    being ignored rather than misread.
    """
    path.write_bytes(text.replace("\r\n", "\n").replace("\n", "\r\n").encode("ascii"))


def _config_text(opts, boards, seed, gravity):
    return "\n".join([
        "' generated by data_generation/gorillas.py -- do not edit",
        "DATAMODE=1",
        f"INPUTMODE={opts['input_mode']}",
        f"ANIMATE={1 if opts['animate'] else 0}",
        f"FASTDRAW={0 if opts['animate'] else 1}",
        "HUD=0",
        # Craters are painted into the framebuffer, and the framebuffer is the
        # collision model, so leaving destruction on would make throw N land on
        # terrain shaped by throws 1..N-1 -- a hidden variable in no column.
        "DESTRUCT=0",
        f"BOARDS={boards}",
        f"THROWS={opts['throws']}",
        "ALTERNATE=1",
        f"SEED={seed}",
        f"GRAVITY={gravity}",
        f"ANGLEMIN={opts['angle_min']}",
        f"ANGLEMAX={opts['angle_max']}",
        f"EFFORTMIN={opts['effort_min']}",
        f"EFFORTMAX={opts['effort_max']}",
        f"VELMIN={opts['vel_min']}",
        f"VELMAX={opts['vel_max']}",
        f"EFFORTJITTER={1 if opts['effort_jitter'] else 0}",
        f"BANANAVAR={1 if opts['banana_var'] else 0}",
        f"DT={opts['dt']}",
        f"MAXTICKS={opts['max_ticks']}",
        "OUTFILE=D:THROWS.CSV",
        "FLUSHEVERY=200",
        "",
    ])


def _conf_text(worker_dir, animate):
    # In animated mode cpu_cycles is a pacing parameter, not a speed knob: Rest
    # scales its wait by the measured CPU speed, so a bigger number makes the
    # game slower. Once Rest is short-circuited it becomes monotonically useful.
    cycles, core = (3000, "auto") if animate else (60000, "dynamic")
    window = "1280x700" if animate else "320x240"
    return "\n".join([
        "[sdl]",
        "output = texturenb",
        "fullscreen = off",
        f"window_size = {window}",
        "vsync = off",
        "",
        "[render]",
        "aspect = auto",
        "",
        "[cpu]",
        f"core = {core}",
        "cputype = auto",
        f"cpu_cycles = {cycles}",
        "",
        "[autoexec]",
        'mount c "."',
        f'mount d "{worker_dir}"',
        "c:",
        "QBASIC.EXE /RUN GORILLA.BAS",
        "exit",
        "",
    ])


# ~Quarter of random throws leave the field without landing and get filtered.
# A fixed constant, not the run's observed keep rate -- using the observed
# rate would make the chunk count (and so the number of RNG draws) depend on
# machine speed, breaking reproducibility under a fixed --seed.
KEEP_ESTIMATE = 0.75
MAX_ROUNDS = 5   # rounds of over-generation before giving up on a plan entry


def chunk_schedule(want, throws, boards_per_chunk, max_rounds=MAX_ROUNDS,
                   keep_estimate=KEEP_ESTIMATE):
    """How many chunks to run in each round, for one plan entry.

    A pure function of its arguments -- it never sees how many rows actually
    came back, so the number of RNG draws is fixed by (seed, n, throws,
    boards_per_chunk) alone and doesn't depend on machine speed.
    """
    per_chunk = boards_per_chunk * throws
    need = int(want / keep_estimate) + throws
    per_round = max(1, -(-need // per_chunk))
    return [per_round] * max_rounds


def _chunk_key(tag, chunk_seed):
    """The group key prefix for one chunk: rows sharing it share a board's
    wind and skyline.

    `tag` is included because normal and out-of-distribution-gravity chunk
    seeds are drawn from the same RNG stream and can collide; without it, two
    chunks thrown under different gravity could get the same group_id.
    """
    return f"{tag}_{chunk_seed}"


def _run_worker(dosbox, worker_dir, opts, boards, seed, gravity, timeout, chunk_key):
    """Run one DOSBox instance to completion and return its parsed rows."""
    worker_dir.mkdir(parents=True, exist_ok=True)
    # Recorded so _dump_raw can recover this chunk's group key later -- board
    # numbers restart at 1 in every worker's own CSV, so `board` alone collides
    # across workers.
    (worker_dir / "CHUNKKEY.TXT").write_text(chunk_key)
    _write_dos_text(worker_dir / "GORCFG.TXT", _config_text(opts, boards, seed, gravity))
    conf = worker_dir / "run.conf"
    _write_dos_text(conf, _conf_text(worker_dir, opts["animate"]))

    cmd = [str(dosbox), "--conf", str(conf), "--noprimaryconf", "--nolocalconf"]
    timed_out = False
    try:
        subprocess.run(cmd, cwd=str(GAME_DIR), timeout=timeout,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        timed_out = True

    csv_path = worker_dir / "THROWS.CSV"
    if not csv_path.is_file():
        raise GameRunFailed(
            f"worker seed={seed} produced no CSV"
            + (" (timed out)" if timed_out else "")
            + f"\n  config: {worker_dir / 'GORCFG.TXT'}"
        )
    rows = _parse_csv(csv_path)
    return rows, timed_out


def _parse_csv(path):
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            raise GameRunFailed(f"empty CSV: {path}")
        if header != CSV_HEADER:
            for i, (got, want) in enumerate(zip(header, CSV_HEADER)):
                if got != want:
                    raise GameRunFailed(
                        f"CSV header mismatch at field {i}: game wrote {got!r}, "
                        f"expected {want!r}. gorilla.bas's CsvOpenFile and "
                        f"gorillas.py's CSV_HEADER have diverged."
                    )
            raise GameRunFailed(
                f"CSV header length mismatch: game wrote {len(header)} fields, "
                f"expected {len(CSV_HEADER)}."
            )
        # A run killed mid-write can leave a short final line; drop it rather
        # than fail, since FLUSHEVERY exists precisely to salvage partial runs.
        return [r for r in reader if len(r) == len(CSV_HEADER)]


def _to_contract(raw_rows, hit_tolerance, outlier_tag, chunk_key):
    """Map the game's throw records onto generate.COLUMNS.

    `chunk_key` (see _chunk_key) plus the game's `board` column forms the
    real group key -- `board` alone restarts at 1 in every worker's own CSV
    and so collides across workers.
    """
    idx = {name: i for i, name in enumerate(CSV_HEADER)}
    out, clipped = [], []

    for r in raw_rows:
        if int(float(r[idx["outcome"]])) not in KEEP_OUTCOMES:
            continue

        f = lambda n: float(r[idx[n]])

        # Raw, pre-mirror angle -- never angle_sim_deg (95-170 for Player 2),
        # which would fail the pipeline's 0-90 range check.
        angle = f("angle_deg")
        velocity = f("velocity_ms")

        # Magnitude/sign decomposition in the tosser's downrange frame -- keep
        # wind_dir in {-1, 1} only, since the Python path never produces 0.0.
        wind_rel = f("wind_rel_ms")
        wind_speed = abs(wind_rel)
        wind_dir = 1.0 if wind_rel >= 0 else -1.0

        mass = f("banana_mass_kg")
        radius = f("banana_diam_m") / 2.0

        launch_h = f("launch_height_m")
        # land_dy_m is signed and measured from the launch point.
        landing_h = launch_h + f("land_dy_m")
        if landing_h < 0:
            clipped.append(-landing_h)
            landing_h = 0.0

        # land_dx_m is signed (negative for Player 2, who throws toward -x);
        # abs() puts both players in the Python path's downrange convention.
        landing_distance = abs(f("land_dx_m"))
        target_distance = f("target_dist_m")

        # Same semantics as generate.py: there the target is placed at
        # landing_x +/- offset and hit_target is exactly this test.
        hit = int(abs(landing_distance - target_distance) <= hit_tolerance)

        board = int(float(r[idx["board"]]))
        group_id = f"{chunk_key}_{board}"

        out.append([
            velocity, angle, wind_speed, wind_dir, mass, radius, BANANA_CD,
            launch_h, landing_h, landing_distance, round(target_distance, 4),
            hit, outlier_tag, group_id,
        ])
    return out, clipped


def generate(n, seed=42, throws=32, workers=8, boards_per_chunk=25, input_mode="EFFORT",
             hit_tolerance=HIT_TOLERANCE, animate=False, effort_jitter=True,
             banana_var=True, dt=0.1, max_ticks=0, angle_min=10.0, angle_max=88.0,
             effort_min=20.0, effort_max=100.0, vel_min=10.0, vel_max=80.0,
             dosbox=None, timeout=1800, keep_raw=None, allow_partial=False,
             with_outliers=True):
    """Run the game until at least `n` usable rows exist, then map and shuffle.

    Total board count is derived from `n`, `throws`, and a fixed keep-rate
    estimate (see chunk_schedule), not from this run's observed keep rate --
    that keeps chunk seeding independent of machine speed. `boards_per_chunk`
    DOES change the resulting data (it changes how many per-chunk seeds get
    drawn), so unlike `workers` it's part of the dataset's identity, not a
    tuning knob.
    """
    dosbox = find_dosbox(dosbox)
    if not (GAME_DIR / "gorilla.bas").is_file():
        raise GameRunFailed(f"game build not found at {GAME_DIR}")

    opts = dict(input_mode=input_mode, throws=throws, animate=animate,
                effort_jitter=effort_jitter, banana_var=banana_var, dt=dt,
                max_ticks=max_ticks, angle_min=angle_min, angle_max=angle_max,
                effort_min=effort_min, effort_max=effort_max,
                vel_min=vel_min, vel_max=vel_max)

    n_out = round(n * OUTLIER_FRAC) if with_outliers else 0
    n_err = round(n * DATA_ERROR_FRAC) if with_outliers else 0
    n_normal = n - n_out - n_err

    rng = random.Random(seed)
    base = Path(tempfile.mkdtemp(prefix="gorillas_"))
    base_short = _short_path(base)
    if " " in str(base_short):
        raise GameRunFailed(
            f"worker directory contains a space ({base_short}); DOSBox cannot "
            "mount it reliably. Set TMP to a path without spaces."
        )

    rows, clipped_all, timeouts = [], [], 0
    try:
        # Normal rows, then a smaller session at out-of-distribution gravity --
        # gravity is a real game input, so is_outlier=="gravity" rows are
        # physically different, not just relabeled.
        plan = [(n_normal + n_err, DEFAULT_GRAVITY, "none")]
        if n_out:
            g = rng.uniform(*GRAVITY_LOW) if rng.random() < 0.5 else rng.uniform(*GRAVITY_HIGH)
            plan.append((n_out, round(g, 4), "gravity"))

        for want, gravity, tag in plan:
            # The full chunk plan and every seed it needs are fixed here, before
            # any worker starts -- this plan entry always consumes the same
            # number of RNG draws regardless of how the run goes, which is what
            # keeps a fixed --seed reproducible.
            #
            # `workers` only controls how many chunks run concurrently; the same
            # chunks run either way, just faster or slower.
            schedule = chunk_schedule(want, throws, boards_per_chunk)
            round_seeds = [[rng.randint(1, 30000) for _ in range(n_chunks)]
                           for n_chunks in schedule]

            got = []
            rounds_run = 0
            for attempt, chunk_seeds in enumerate(round_seeds):
                if len(got) >= want:
                    # Stopping early skips seeds that were already drawn above,
                    # so the RNG stream position after this loop is unchanged.
                    break
                rounds_run += 1
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    # Paired with its chunk key explicitly -- without this,
                    # _to_contract has no way to know which chunk a completed
                    # future's rows came from.
                    futures = [
                        (_chunk_key(tag, cs),
                         pool.submit(_run_worker, dosbox,
                                     base_short / f"w{tag}{attempt}_{i}", opts,
                                     boards_per_chunk, cs, gravity, timeout,
                                     _chunk_key(tag, cs)))
                        for i, cs in enumerate(chunk_seeds)
                    ]
                    for key, fut in futures:
                        raw, to = fut.result()
                        timeouts += int(to)
                        mapped, clipped = _to_contract(raw, hit_tolerance, tag, key)
                        got.extend(mapped)
                        clipped_all.extend(clipped)

            # A timeout means a worker's CSV was accepted half-written, making
            # the result depend on machine speed -- fail by default rather than
            # silently yield a machine-specific dataset; --allow-partial opts out.
            if timeouts and not allow_partial:
                raise GameRunFailed(
                    f"{timeouts} worker(s) hit the {timeout}s timeout; their "
                    f"partial output would make this dataset depend on machine "
                    f"speed rather than --seed alone. Raise --timeout, lower "
                    f"--boards-per-chunk, or pass --allow-partial to accept a "
                    f"non-reproducible run."
                )

            if len(got) < want:
                msg = (f"only produced {len(got)} of {want} '{tag}' rows after "
                       f"{rounds_run} of {len(schedule)} planned rounds")
                if not allow_partial:
                    raise GameRunFailed(msg + " (pass --allow-partial to accept)")
                print(f"  WARNING: {msg}")
            rng.shuffle(got)
            rows.extend(got[:want])

        if keep_raw:
            _dump_raw(base_short, Path(keep_raw))
    finally:
        if not keep_raw:
            shutil.rmtree(base, ignore_errors=True)

    if not rows:
        raise GameRunFailed("no usable rows after filtering")

    # data_error rows are corrupted here with the same function the Python
    # path uses -- a data-entry mistake was never physical in either producer.
    if n_err:
        for i in range(min(n_err, len(rows))):
            row = rows[i]
            row[:11] = corrupt_row(rng, row[:11], no_zero_indices={gen.MASS_COLUMN_INDEX})
            row[12] = "data_error"

    # Shuffle before truncating: n_samples is applied downstream as a prefix
    # slice, so the first n rows must carry the same outlier mix as the pool.
    rng.shuffle(rows)
    rows = rows[:n]

    df = pd.DataFrame(rows, columns=gen.COLUMNS)
    for c in gen.COLUMNS[:11]:
        df[c] = df[c].astype("float64")
    df["hit_target"] = df["hit_target"].astype("int64")
    df["group_id"] = df["group_id"].astype(str)

    _report(df, clipped_all, timeouts)
    # Validated here (not just in the CLI) so any in-process caller gets the
    # same guarantee as the DVC stage.
    return _check_contract(df)


def _dump_raw(base, dest):
    """Concatenate every worker's CSV into one parquet, keeping the columns
    the 14-column contract omits (`board`, `outcome`, pixel geometry) plus
    chunk_key/group_id so groups are reconstructable and joinable to the main
    output.

    Parquet, not CSV: pyarrow is already a dependency, columns come back
    typed, and these logs are multi-MB of text that DVC would otherwise cache
    uncompressed.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    board_idx = CSV_HEADER.index("board")
    rows = []
    for p in sorted(base.glob("*/THROWS.CSV")):
        key_path = p.parent / "CHUNKKEY.TXT"
        chunk_key = key_path.read_text().strip() if key_path.is_file() else ""
        with open(p, newline="") as fh:
            rd = csv.reader(fh)
            next(rd, None)
            for r in rd:
                if len(r) == len(CSV_HEADER):
                    group_id = f"{chunk_key}_{r[board_idx]}" if chunk_key else ""
                    rows.append(r + [chunk_key, group_id])

    raw = pd.DataFrame(rows, columns=CSV_HEADER + ["chunk_key", "group_id"])
    # The game writes everything as text; restore numeric types so the log can be
    # analysed without re-parsing every column at the call site.
    for c in CSV_HEADER:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    write_parquet(raw, dest)
    print(f"Raw throw log -> {dest}")


def _report(df, clipped, timeouts):
    hits = int(df["hit_target"].sum())
    print(f"  Rows  : {len(df):,}   Hits : {hits:,}   Misses : {len(df) - hits:,}")
    print("  is_outlier : " + ", ".join(
        f"{k}={v}" for k, v in df["is_outlier"].value_counts().items()))
    if timeouts:
        print(f"  WARNING: {timeouts} worker(s) timed out; used their partial output")
    if clipped:
        worst = max(clipped)
        print(f"  landing_height_m clipped to 0 on {len(clipped)} rows (max {worst:.2f} m)")
        if worst > CLIP_TOLERANCE_M:
            # A clip this large means land_dy_m/launch_height_m disagree about
            # ground level -- likely a mapping bug, not a ground hit. Fail
            # rather than silently write corrupt rows; the raw log records the
            # per-throw clip amounts for diagnosis.
            raise GameRunFailed(
                f"landing_height_m was clipped by up to {worst:.2f} m (tolerance "
                f"{CLIP_TOLERANCE_M} m) on {len(clipped)} of {len(df):,} rows. A clip "
                f"this large means land_dy_m and launch_height_m disagree about where "
                f"the ground is -- a mapping bug between gorilla.bas and _to_contract, "
                f"not a banana landing below the launch point. Fix the mapping rather "
                f"than training on these rows; the per-throw clip amounts are in the "
                f"raw log (--raw-out)."
            )


def _check_contract(df):
    """Assert the 14-column contract (checks live in data_generation/contract.py,
    shared with the Python producer and both training loaders)."""
    return check_contract(df, source="gorillas")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hit-tolerance", type=float, default=HIT_TOLERANCE)
    p.add_argument("--input-mode", choices=["EFFORT", "VELOCITY"], default="EFFORT",
                   help="VELOCITY samples launch speed directly (no force model, no cap)")
    p.add_argument("--throws", type=int, default=32,
                   help="throws per board; higher amortises board setup but "
                        "adds per-board correlation (default: 32)")
    p.add_argument("--workers", type=int, default=8,
                   help="concurrent DOSBox instances -- performance only, does "
                        "not change the resulting data (default: 8)")
    p.add_argument("--boards-per-chunk", type=int, default=25,
                   help="boards per DOSBox session; changes the data (unlike "
                        "--workers), so it's part of the dataset's identity")
    p.add_argument("--animate", action="store_true",
                   help="watch it play; far slower, for demos")
    p.add_argument("--no-effort-jitter", dest="effort_jitter", action="store_false")
    p.add_argument("--no-banana-var", dest="banana_var", action="store_false")
    p.add_argument("--no-outliers", dest="with_outliers", action="store_false",
                   help="emit is_outlier='none' only (leaves clean:no_outlier stages a no-op)")
    p.add_argument("--dt", type=float, default=0.1,
                   help="physics timestep (0.1 is the game's own); raising it "
                        "changes the trajectories, not just the speed")
    p.add_argument("--max-ticks", type=int, default=0)
    p.add_argument("--angle-min", type=float, default=10.0)
    p.add_argument("--angle-max", type=float, default=88.0)
    p.add_argument("--effort-min", type=float, default=20.0)
    p.add_argument("--effort-max", type=float, default=100.0)
    p.add_argument("--vel-min", type=float, default=10.0)
    p.add_argument("--vel-max", type=float, default=80.0)
    p.add_argument("--dosbox", type=str, default=None)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--raw-out", type=str, default=None,
                   help="also write the unmapped throw log as parquet, joinable "
                        "to the main output on group_id")
    p.add_argument("--allow-partial", action="store_true",
                   help="accept a run where workers timed out or `n` was not "
                        "reached. Off by default -- partial output makes the "
                        "dataset depend on machine speed, not just --seed")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="run and validate, but write no parquet")
    args = p.parse_args()

    out_path = resolve_output(args.out)
    if out_path.exists() and not args.overwrite and not args.dry_run:
        sys.exit(f"Refusing to overwrite {out_path}\nPass --overwrite to replace it, "
                 f"or --dry-run to generate and inspect without writing.")

    t0 = time.time()
    df = generate(
        args.n, seed=args.seed, throws=args.throws, workers=args.workers,
        boards_per_chunk=args.boards_per_chunk,
        input_mode=args.input_mode, hit_tolerance=args.hit_tolerance,
        animate=args.animate, effort_jitter=args.effort_jitter,
        banana_var=args.banana_var, dt=args.dt, max_ticks=args.max_ticks,
        angle_min=args.angle_min, angle_max=args.angle_max,
        effort_min=args.effort_min, effort_max=args.effort_max,
        vel_min=args.vel_min, vel_max=args.vel_max, dosbox=args.dosbox,
        timeout=args.timeout, keep_raw=args.raw_out,
        allow_partial=args.allow_partial, with_outliers=args.with_outliers,
    )
    elapsed = time.time() - t0
    print(f"  Elapsed : {elapsed:.1f}s ({len(df) / elapsed:.1f} rows/s)")
    # generate() already validated the contract on the way out.

    if args.dry_run:
        print("\n--dry-run: nothing written. Summary of what would be saved:\n")
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(df.describe().T[["count", "mean", "min", "max"]])
    else:
        write_parquet(df, out_path)
