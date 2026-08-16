"""Listener-level cross-validation and leakage guards.

A trial-level random split places the same listener in both training and evaluation. The
model can then memorise the listener rather than learn a transferable relationship, which
inflates every metric. AUDIRE therefore supports **only** grouped splits for headline
results, and :func:`assert_no_listener_leakage` is called on every fold that any
evaluation runner produces.

:class:`LeakySplitter` exists solely so that the leakage guard itself can be tested
against a split that is known to leak; it is rejected by the runner.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

IntArray = npt.NDArray[np.int64]
StrArray = npt.NDArray[np.str_]
BoolArray = npt.NDArray[np.bool_]


class LeakageError(AssertionError):
    """Raised when a split places the same listener on both sides of a fold."""


@dataclass(frozen=True, slots=True)
class Fold:
    """One cross-validation fold."""

    index: int
    train_idx: IntArray
    test_idx: IntArray
    train_listeners: tuple[str, ...]
    test_listeners: tuple[str, ...]

    @property
    def n_train(self) -> int:
        return int(self.train_idx.size)

    @property
    def n_test(self) -> int:
        return int(self.test_idx.size)


def assert_no_listener_leakage(groups: StrArray, train_idx: IntArray, test_idx: IntArray) -> None:
    """Raise :class:`LeakageError` if any listener appears in both index sets."""
    train_listeners = set(np.asarray(groups)[train_idx].tolist())
    test_listeners = set(np.asarray(groups)[test_idx].tolist())
    shared = train_listeners & test_listeners
    if shared:
        raise LeakageError(
            f"{len(shared)} listener(s) appear in both train and test: "
            f"{sorted(shared)[:5]}{'...' if len(shared) > 5 else ''}"
        )


def listener_folds(
    groups: StrArray,
    y: IntArray | None = None,
    *,
    n_splits: int = 5,
    stratify: bool = True,
    seed: int = 0,
) -> list[Fold]:
    """Build listener-level folds.

    ``stratify=True`` uses :class:`sklearn.model_selection.StratifiedGroupKFold` so that
    the outcome base rate stays comparable across folds while whole listeners are kept
    together; it needs ``y``. Every fold is checked for leakage before being returned.

    Raises
    ------
    ValueError
        If there are fewer listeners than folds — silently reducing the fold count would
        change the experiment without saying so.
    """
    groups = np.asarray(groups)
    n_listeners = int(np.unique(groups).size)
    if n_listeners < n_splits:
        raise ValueError(
            f"cannot build {n_splits} listener-level folds from {n_listeners} listeners"
        )

    idx = np.arange(groups.size)
    if stratify:
        if y is None:
            raise ValueError("stratified listener folds need labels")
        splitter: Any = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        pairs = splitter.split(idx.reshape(-1, 1), np.asarray(y), groups)
    else:
        pairs = GroupKFold(n_splits=n_splits).split(idx.reshape(-1, 1), None, groups)

    folds: list[Fold] = []
    for i, (train, test) in enumerate(pairs):
        train_idx = np.asarray(train, dtype=np.int64)
        test_idx = np.asarray(test, dtype=np.int64)
        assert_no_listener_leakage(groups, train_idx, test_idx)
        folds.append(
            Fold(
                index=i,
                train_idx=train_idx,
                test_idx=test_idx,
                train_listeners=tuple(sorted(set(groups[train_idx].tolist()))),
                test_listeners=tuple(sorted(set(groups[test_idx].tolist()))),
            )
        )
    return folds


def leave_one_listener_out(groups: StrArray) -> list[Fold]:
    """One fold per listener. Appropriate for very small cohorts."""
    groups = np.asarray(groups)
    folds: list[Fold] = []
    for i, listener in enumerate(sorted(set(groups.tolist()))):
        test_mask = groups == listener
        train_idx = np.flatnonzero(~test_mask).astype(np.int64)
        test_idx = np.flatnonzero(test_mask).astype(np.int64)
        assert_no_listener_leakage(groups, train_idx, test_idx)
        folds.append(
            Fold(
                index=i,
                train_idx=train_idx,
                test_idx=test_idx,
                train_listeners=tuple(sorted(set(groups[train_idx].tolist()))),
                test_listeners=(str(listener),),
            )
        )
    return folds


class LeakySplitter:
    """A deliberately leaky trial-level splitter.

    **Test fixture only.** It exists so the leakage guard can be shown to fire; the
    evaluation runner does not accept it.
    """

    def __init__(self, n_splits: int = 5, seed: int = 0) -> None:
        self.n_splits = n_splits
        self.seed = seed

    def split(self, groups: StrArray) -> Iterator[tuple[IntArray, IntArray]]:
        n = np.asarray(groups).size
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(n)
        for k in range(self.n_splits):
            test = order[k :: self.n_splits]
            train = np.setdiff1d(order, test)
            yield train.astype(np.int64), test.astype(np.int64)
