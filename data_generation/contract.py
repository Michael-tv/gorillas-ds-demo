"""The 14-column dataset contract, asserted on write and on read.

Both producers -- data_generation/generate.py (RK4 via physics.py) and
data_generation/gorillas.py (the real 1990 game via DOSBox) -- emit the same
columns so either can fill the training slot that params.yaml `training_data`
selects. This module is the one statement of what "the same columns" means, and
it runs at both ends of the pipeline:

  * on write, in each producer, so a bad frame never reaches the parquet;
  * on read, in the shared loaders, so a parquet written by an older revision
    (or by hand, or half-overwritten) cannot quietly train a model.

Checking on read matters as much as on write: `data/` is DVC-cached rather than
in Git, so a pool on disk can predate the current code with nothing in the
working tree revealing it. This started life as `_check_contract` inside
gorillas.py, where it only ran under `__main__` and so validated one producer,
at one moment (AUDIT.md task 38; it is also most of task 28).

Plain asserts and explicit range literals, deliberately -- a schema DSL would
be a new dependency and a new thing for a reader of this repo to learn.
"""
import data_generation.generate as gen

# Physical bounds every clean row must satisfy. `None` means unbounded on that
# side. Mirrors models/regression/common/loader.py's FEATURE_RANGES, which is
# what the clean:"range" strategy filters on -- the point of asserting them here
# is that a *clean* row violating them is a producer bug, whereas a data_error
# row violating them is the whole purpose of that row.
CLEAN_NON_NEGATIVE = [
    "wind_speed_ms", "mass_kg", "radius_m", "drag_coeff",
    "launch_height_m", "landing_height_m",
    "initial_velocity_ms",
]

# landing_distance_m is deliberately NOT in that list: it can legitimately be
# negative. A high-angle shot into a strong headwind lands behind the launch
# point -- measured over 20,000 shots at the configured distribution, 1.44% do,
# every one of them having actually landed (flight times 1-10 s against
# max_time=60), all with wind_x < 0. Example: speed 73 m/s at 79 deg into 17.4
# m/s of headwind lands at -65.0 m, where the same shot windless lands at +30.8.
#
# AUDIT.md §1.2 attributed these rows to physics.simulate returning a
# non-terminating trajectory's endpoint as if it were a landing. That is not
# what they are: zero shots out of 20,000 failed to terminate. simulate_landing
# now reports termination explicitly so the distinction is checkable rather than
# assumed, and sampling.py raises if a shot ever really fails to land.


class ContractViolation(AssertionError):
    pass


def check_contract(df, source="dataset", verbose=True):
    """Assert the frame satisfies the contract. Returns the frame, so it can
    wrap a read: `df = check_contract(pd.read_parquet(path), source=path)`.

    Raises ContractViolation on a structural or physical violation. Conditions
    that are degenerate but legitimate (a single group, a single class) warn
    instead, since a deliberately small or one-sided run is a valid thing to
    generate -- it just must not be silent.
    """
    def require(cond, msg):
        if not cond:
            raise ContractViolation(f"{source}: {msg}")

    missing = [c for c in gen.COLUMNS if c not in df.columns]
    stale_note = ""
    if missing == ["group_id"]:
        # By far the most likely way this fires: a pool generated before task 32
        # added group_id. data/ is DVC-cached rather than in Git, so an existing
        # pool on disk silently predates the code that reads it.
        stale_note = ("\n  This pool predates the group_id column (AUDIT.md task 32), so it "
                      "cannot\n  support a group-aware split. Regenerate it:\n"
                      "    dvc repro generate                  # the Python pool\n"
                      "    dvc repro generate_gorillas         # the game pools (needs DOSBox)\n"
                      "  Nothing is lost -- both producers are deterministic for a fixed seed.")
    require(list(df.columns) == gen.COLUMNS,
            f"column set/order does not match generate.COLUMNS\n"
            f"  expected: {gen.COLUMNS}\n  got:      {list(df.columns)}" + stale_note)
    require(df.columns[gen.MASS_COLUMN_INDEX] == "mass_kg",
            "MASS_COLUMN_INDEX no longer points at mass_kg")
    require(len(df) > 0, "no rows")
    require(df["mass_kg"].ne(0).all(),
            "mass_kg contains zeros (feature engineering divides by it)")
    require(set(df["hit_target"].unique()) <= {0, 1}, "hit_target is not 0/1")
    require(set(df["is_outlier"].unique()) <= {"none", "gravity", "data_error"},
            "unexpected is_outlier value")
    require(df["group_id"].notna().all() and (df["group_id"].astype(str) != "").all(),
            "group_id missing or empty -- a group-aware split cannot be built from this pool")

    # Range checks apply to clean rows only. data_error rows are corrupted on
    # purpose -- scaled, sign-flipped or zeroed -- and violating these bounds is
    # exactly what makes them useful for demonstrating clean:"range".
    clean = df[df["is_outlier"] != "data_error"]
    require(set(clean["wind_direction_norm"].unique()) <= {-1.0, 1.0},
            "wind_direction_norm is not +/-1 on clean rows")
    require(clean["launch_angle_deg"].between(0, 90).all(),
            "launch_angle_deg outside 0-90 on clean rows")
    require((clean[CLEAN_NON_NEGATIVE] >= 0).all().all(),
            "negative value on a clean row in " + ", ".join(CLEAN_NON_NEGATIVE))

    warnings = []
    if df["group_id"].nunique() < 2:
        warnings.append("group_id has fewer than 2 distinct values -- a "
                        "group-aware split has nothing to split across")
    if set(df["hit_target"].unique()) != {0, 1}:
        warnings.append("hit_target has a single class -- stratified splits will fail")
    for w in warnings:
        print(f"  WARNING: {source}: {w}")
    if verbose and not warnings:
        print(f"  Contract check: OK ({len(df):,} rows, {df['group_id'].nunique():,} groups)")
    return df
