"""Group-aware train/test splitting and cross-validation.

The Gorillas producer throws 32 bananas per board, sharing wind/skyline across
throws -- a plain train_test_split would leak board structure across the split.
`group_id` marks this; when a pool has no real groups (e.g. the Python
producer, one id per row), splits degrade to an ordinary random split.
"""
import numpy as np
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit,
                                     ShuffleSplit, StratifiedGroupKFold,
                                     train_test_split)


def has_groups(groups):
    """True when `groups` has fewer distinct values than rows."""
    if groups is None:
        return False
    groups = np.asarray(groups)
    return len(groups) > 0 and len(np.unique(groups)) < len(groups)


def _take(data, idx):
    """Positional selection that works for both a DataFrame and an ndarray."""
    return data.iloc[idx] if hasattr(data, "iloc") else data[idx]


def split(X, y, groups=None, test_size=0.2, random_state=42, stratify=False):
    """Split into train/test, keeping each group wholly on one side.

    Returns `(X_train, X_test, y_train, y_test, groups_train)`.

    `stratify=True` also balances labels -- useful since the Gorillas hit rate
    is only 5-7%. With groups this uses StratifiedGroupKFold (GroupShuffleSplit
    can't stratify); fold count is derived from test_size.
    """
    y = np.asarray(y)
    if not has_groups(groups):
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

    Returns the fold count unchanged when no grouping applies, otherwise a
    materialised list of (train_idx, test_idx) pairs -- needed because RidgeCV/
    LassoCV don't accept a splitter object requiring `groups=` at fit time.

    Folds are capped at the number of groups (GroupKFold can't exceed that).
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

    For ungrouped pools, fitting each search candidate 5x buys nothing over
    fitting it once.
    """
    return ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
