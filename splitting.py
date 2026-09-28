"""Group-aware train/test splitting and cross-validation.

The Gorillas producer throws 32 bananas at each board, and a board's wind and
skyline are shared by all of them. A plain `train_test_split` therefore puts
throws from the same board on both sides of the split, letting a model memorise
board-specific structure and score against rows it has effectively already seen
-- textbook group leakage, and the largest one in this repo whenever a Gorillas
pool is active (AUDIT.md §5.1, task 36).

`group_id` (generate.COLUMNS' 14th column, task 32) is the key. This module is
the single place that knows what to do with it, so every training script gets
the same behaviour from one implementation:

  * When the active pool genuinely has groups, splits are group-aware
    (GroupShuffleSplit, or StratifiedGroupKFold when the labels must also be
    balanced) and cross-validation uses GroupKFold.
  * When it does not -- the Python producer gives every row its own id, because
    sample_shot draws each row independently -- a group-aware split degrades
    to an ordinary random split, which is exactly right. That is why the same
    code path serves both producers with no per-producer branching in the
    training scripts.
"""
import numpy as np
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit,
                                     ShuffleSplit, StratifiedGroupKFold,
                                     train_test_split)


def has_groups(groups):
    """True when `groups` actually groups rows together, i.e. there are fewer
    distinct values than rows. One group per row carries no grouping
    information, so treating it as grouped would only cost accuracy."""
    if groups is None:
        return False
    groups = np.asarray(groups)
    return len(groups) > 0 and len(np.unique(groups)) < len(groups)


def _take(data, idx):
    """Positional selection that works for both a DataFrame (the regression
    loader returns one, so the Pipeline's engineering step keeps column names)
    and a plain ndarray (the classification loader returns one)."""
    return data.iloc[idx] if hasattr(data, "iloc") else data[idx]


def split(X, y, groups=None, test_size=0.2, random_state=42, stratify=False):
    """Split into train/test, keeping each group wholly on one side.

    Returns `(X_train, X_test, y_train, y_test, groups_train)`. The trailing
    `groups_train` is what cross-validation needs, so a caller that splits and
    then cross-validates does not have to index the groups itself and risk
    getting it subtly wrong.

    `stratify=True` asks for balanced labels as well -- which matters here
    because the Gorillas hit rate is 5-7% (§5.4). With groups that means
    StratifiedGroupKFold, since GroupShuffleSplit cannot stratify; the fold
    count is derived from `test_size`, so test_size=0.2 takes one fifth.
    """
    y = np.asarray(y)
    if not has_groups(groups):
        # No group structure: an ordinary random split IS the correct split.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if stratify else None)
        return X_train, X_test, y_train, y_test, None

    groups = np.asarray(groups)
    if stratify:
        n_splits = max(2, round(1.0 / test_size))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                        random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                     random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    n_groups = len(np.unique(groups))
    print(f"  Group-aware split: {n_groups:,} groups -> "
          f"{len(np.unique(groups[train_idx])):,} train / "
          f"{len(np.unique(groups[test_idx])):,} test, no group on both sides")
    return (_take(X, train_idx), _take(X, test_idx),
            y[train_idx], y[test_idx], groups[train_idx])


def cv_for(cv, X, y, groups_train, stratify=False):
    """Translate a plain fold count into a group-aware cross-validation scheme.

    Returns something to pass straight as `cv=` -- the fold count unchanged when
    no grouping applies, otherwise a materialised list of `(train_idx,
    test_idx)` pairs built with the groups already applied.

    Materialised pairs rather than a splitter object on purpose: a splitter
    needs `groups=` supplied again at fit time, which `RandomizedSearchCV.fit`
    accepts but `RidgeCV` and `LassoCV` do not -- they would raise "the 'groups'
    parameter should not be None". An explicit list of index pairs is accepted by
    every sklearn estimator that takes `cv`, so the call site stays one argument
    either way and no training script has to know which kind of estimator it is
    holding.

    Folds are capped at the number of groups, since GroupKFold cannot make more
    folds than there are groups and a small run may legitimately have few.
    """
    if cv is None or not has_groups(groups_train):
        return cv
    groups_train = np.asarray(groups_train)
    n_groups = len(np.unique(groups_train))
    n_splits = max(2, min(cv, n_groups))
    splitter = (StratifiedGroupKFold(n_splits=n_splits) if stratify
                else GroupKFold(n_splits=n_splits))
    if n_splits != cv:
        print(f"  CV: {cv} folds requested, using {n_splits} -- only "
              f"{n_groups:,} groups available")
    else:
        print(f"  CV: {n_splits}-fold, group-aware ({n_groups:,} groups)")
    return list(splitter.split(X, y, groups=groups_train))


def single_split_cv(test_size=0.2, random_state=42):
    """A single train/validation split to use as `cv=` in place of k-fold CV.

    For pools with no group structure to protect (see `cv_for`), fitting each
    search candidate 5x buys nothing over fitting it once -- the fold count
    only matters when groups need spreading across folds without leaking.
    """
    return ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
