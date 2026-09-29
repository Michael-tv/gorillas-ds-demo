"""The 14-column dataset contract, asserted on write and on read.

Both producers (generate.py's RK4 integrator and gorillas.py's real game)
emit the same columns so either can fill the training slot. Checked on read
too, not just on write: `data/` is DVC-cached rather than in Git, so a pool
on disk can predate the current code with nothing in the working tree
revealing it.

Plain asserts and explicit range literals, deliberately -- no schema DSL.
"""
import data_generation.generate as gen

# Physical bounds every clean row must satisfy; mirrors loader.py's FEATURE_RANGES.
# A clean row violating these is a producer bug; a data_error row violating them
# is the point of that row.
CLEAN_NON_NEGATIVE = [
    "wind_speed_ms", "mass_kg", "radius_m", "drag_coeff",
    "launch_height_m", "landing_height_m",
    "initial_velocity_ms",
]

# landing_distance_m is deliberately NOT in that list: a high-angle shot into
# a strong headwind can legitimately land behind the launch point (negative
# value). sampling.py raises if a shot fails to land at all, so a negative
# value here is always a genuine landing, not an integration artifact.


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
        # data/ is DVC-cached rather than in Git, so an existing pool on disk
        # can silently predate the code that reads it.
        stale_note = ("\n  This pool predates the group_id column, so it cannot\n"
                      "  support a group-aware split. Regenerate it:\n"
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
    # A zero-variance feature carries no signal and looks like a broken model in
    # an importance plot (e.g. Gorillas' drag_coeff, held constant by the game).
    # Judged on clean rows only -- a data_error row's corruption could mask it.
    constant = [c for c in gen.COLUMNS[:11] if clean[c].nunique() == 1]
    if constant:
        warnings.append("zero variance (one distinct value) in: " + ", ".join(constant)
                        + " -- these carry no signal and will rank at ~0 in any "
                          "feature-importance plot")
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
