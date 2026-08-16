"""Listener-level bootstrap confidence intervals and paired comparisons.

Resampling is over **listeners**, not trials. Trials from one listener are correlated, so
a trial-level bootstrap would produce intervals that are far too narrow and would make
two arms look reliably different when they are not.

Paired comparisons resample the *same* listeners for both arms, which removes
between-listener variance from the contrast and is the right test for "does adding the
confusion profile help?".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
StrArray = npt.NDArray[np.str_]

#: Default number of bootstrap resamples. Preregistered in experiment configs.
DEFAULT_N_BOOTSTRAP = 1000


@dataclass(frozen=True, slots=True)
class Interval:
    """A point estimate with a percentile confidence interval."""

    point: float
    lo: float
    hi: float
    level: float
    n_resamples: int
    n_valid: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.lo:.4f}, {self.hi:.4f}]"

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval lies entirely on one side of zero."""
        return (self.lo > 0.0) or (self.hi < 0.0)


def _listener_resample_indices(groups: StrArray, rng: np.random.Generator) -> IntArray:
    """Draw a bootstrap sample of listeners and return the row indices they contribute."""
    unique = np.unique(groups)
    drawn = rng.choice(unique, size=unique.size, replace=True)
    by_listener: dict[str, IntArray] = {
        str(g): np.flatnonzero(groups == g).astype(np.int64) for g in unique
    }
    return np.concatenate([by_listener[str(g)] for g in drawn])


def bootstrap_metric(
    groups: StrArray,
    statistic: Callable[[IntArray], float],
    *,
    n_resamples: int = DEFAULT_N_BOOTSTRAP,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Bootstrap ``statistic`` by resampling listeners with replacement.

    ``statistic`` receives row indices and returns a scalar. Resamples for which the
    statistic is undefined (``NaN`` — e.g. a resample containing only one class) are
    dropped and counted, rather than propagating ``NaN`` into the interval.
    """
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    point = statistic(np.arange(groups.size, dtype=np.int64))

    values: list[float] = []
    for _ in range(n_resamples):
        idx = _listener_resample_indices(groups, rng)
        v = statistic(idx)
        if np.isfinite(v):
            values.append(float(v))

    if not values:
        return Interval(
            point=point,
            lo=float("nan"),
            hi=float("nan"),
            level=level,
            n_resamples=n_resamples,
            n_valid=0,
        )
    alpha = (1.0 - level) / 2.0
    arr = np.asarray(values, dtype=np.float64)
    return Interval(
        point=float(point),
        lo=float(np.quantile(arr, alpha)),
        hi=float(np.quantile(arr, 1.0 - alpha)),
        level=level,
        n_resamples=n_resamples,
        n_valid=len(values),
    )


def paired_bootstrap_difference(
    groups: StrArray,
    statistic_a: Callable[[IntArray], float],
    statistic_b: Callable[[IntArray], float],
    *,
    n_resamples: int = DEFAULT_N_BOOTSTRAP,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Bootstrap ``statistic_a - statistic_b`` on the same resampled listeners.

    The pairing is what makes this the right comparison between two arms evaluated on the
    identical listeners and trials.
    """
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    all_idx = np.arange(groups.size, dtype=np.int64)
    point = statistic_a(all_idx) - statistic_b(all_idx)

    diffs: list[float] = []
    for _ in range(n_resamples):
        idx = _listener_resample_indices(groups, rng)
        a, b = statistic_a(idx), statistic_b(idx)
        if np.isfinite(a) and np.isfinite(b):
            diffs.append(float(a - b))

    if not diffs:
        return Interval(
            point=point,
            lo=float("nan"),
            hi=float("nan"),
            level=level,
            n_resamples=n_resamples,
            n_valid=0,
        )
    alpha = (1.0 - level) / 2.0
    arr = np.asarray(diffs, dtype=np.float64)
    return Interval(
        point=float(point),
        lo=float(np.quantile(arr, alpha)),
        hi=float(np.quantile(arr, 1.0 - alpha)),
        level=level,
        n_resamples=n_resamples,
        n_valid=len(diffs),
    )


def metric_statistic(
    y: IntArray, p: FloatArray, metric: Callable[[IntArray, FloatArray], float]
) -> Callable[[IntArray], float]:
    """Adapt a ``(y, p) -> float`` metric into an index-taking statistic for the bootstrap."""

    def statistic(idx: IntArray) -> float:
        try:
            return float(metric(y[idx], p[idx]))
        except ValueError:
            # A resample containing a single class makes some metrics undefined; report
            # NaN so the caller drops the resample rather than crashing the run.
            return float("nan")

    return statistic
