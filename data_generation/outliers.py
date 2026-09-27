"""Row-corruption logic for the "data_error" outlier category.

Extracted from the copy-pasted corruption loop in every old generate_data.py.
"""


def corrupt_row(rng, values, no_zero_indices=frozenset()):
    """Return a copy of `values` with 1-2 entries corrupted (scale, sign-flip, offset, or zeroed).

    `rng` is an explicit random.Random instance -- the caller owns seeding, so
    two calls with the same rng state produce the same corruption regardless
    of what else has drawn from the global `random` module.

    `no_zero_indices` marks columns that must never be corrupted to exactly
    zero (e.g. mass_kg -- engineered features divide by it downstream, so a
    zeroed mass would produce inf/nan instead of a merely-wrong value).
    """
    values = list(values)
    for idx in rng.sample(range(len(values)), rng.randint(1, 2)):
        v = values[idx]
        c = rng.randint(0, 4)
        if c == 0:
            v *= 10
        elif c == 1:
            v *= 0.1
        elif c == 2:
            v = -v
        elif c == 3:
            v += abs(v) * rng.uniform(5, 20)
        elif idx in no_zero_indices:
            v *= 0.01
        else:
            v = 0.0
        values[idx] = v
    return values
