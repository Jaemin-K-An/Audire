"""Discrimination and calibration metrics.

Discrimination (can the model *rank* misheard words above correctly-heard ones?) and
calibration (are the probabilities *right*?) are reported together everywhere, because a
threshold caption policy needs both and a good AUC with bad calibration would silently
show the wrong amount of text.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

#: Number of equal-width bins for expected calibration error. Preregistered in configs so
#: that a bin count cannot be chosen after seeing the result.
DEFAULT_ECE_BINS = 10


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Metrics at one operating threshold, plus threshold-free discrimination."""

    n: int
    n_positive: int
    base_rate: float
    threshold: float
    # Threshold-free discrimination
    pr_auc: float
    roc_auc: float
    # Probability quality
    brier: float
    log_loss: float
    ece: float
    mce: float
    # Threshold-dependent
    recall: float
    precision: float
    specificity: float
    f1: float
    accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate(y_true: IntArray, y_prob: FloatArray) -> tuple[IntArray, FloatArray]:
    y = np.asarray(y_true, dtype=np.int64).ravel()
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y_true {y.shape} vs y_prob {p.shape}")
    if y.size == 0:
        raise ValueError("cannot compute metrics on an empty sample")
    if not np.all(np.isin(y, (0, 1))):
        raise ValueError("y_true must contain only 0 and 1")
    if np.any(~np.isfinite(p)):
        raise ValueError("y_prob contains non-finite values")
    if p.min() < 0.0 or p.max() > 1.0:
        raise ValueError(f"y_prob must lie in [0, 1]; got [{p.min()}, {p.max()}]")
    return y, p


def expected_calibration_error(
    y_true: IntArray, y_prob: FloatArray, n_bins: int = DEFAULT_ECE_BINS
) -> tuple[float, float]:
    """Return ``(ECE, MCE)`` using equal-width bins on ``[0, 1]``.

    ECE is the sample-weighted mean absolute gap between predicted confidence and observed
    frequency; MCE is the largest such gap over non-empty bins. Empty bins contribute
    nothing. Both are bin-count dependent, which is why the bin count is preregistered.
    """
    y, p = _validate(y_true, y_prob)
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-closed bins so that p == 1.0 lands in the last bin rather than out of range.
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)

    total = 0.0
    worst = 0.0
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        gap = abs(float(p[mask].mean()) - float(y[mask].mean()))
        total += count * gap
        worst = max(worst, gap)
    return total / y.size, worst


def reliability_curve(
    y_true: IntArray, y_prob: FloatArray, n_bins: int = DEFAULT_ECE_BINS
) -> dict[str, list[float]]:
    """Binned reliability data for a calibration plot.

    Returns bin centres, mean predicted probability, observed frequency and bin counts.
    Empty bins are omitted rather than plotted as zero.
    """
    y, p = _validate(y_true, y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    centres: list[float] = []
    predicted: list[float] = []
    observed: list[float] = []
    counts: list[float] = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        centres.append(float((edges[b] + edges[b + 1]) / 2))
        predicted.append(float(p[mask].mean()))
        observed.append(float(y[mask].mean()))
        counts.append(float(mask.sum()))
    return {
        "bin_centre": centres,
        "mean_predicted": predicted,
        "observed_frequency": observed,
        "count": counts,
    }


def compute_metrics(
    y_true: IntArray,
    y_prob: FloatArray,
    *,
    threshold: float = 0.5,
    n_bins: int = DEFAULT_ECE_BINS,
) -> ClassificationMetrics:
    """Compute the full metric set.

    Degenerate cases are handled explicitly rather than by exception: when the sample has
    only one class, the threshold-free discrimination metrics are undefined and returned
    as ``NaN`` — a single-class evaluation fold must be visible, not silently scored.
    """
    y, p = _validate(y_true, y_prob)
    n_pos = int(y.sum())
    single_class = n_pos == 0 or n_pos == y.size

    pr_auc = float("nan") if single_class else float(average_precision_score(y, p))
    roc_auc = float("nan") if single_class else float(roc_auc_score(y, p))

    ece, mce = expected_calibration_error(y, p, n_bins)

    pred = (p >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())

    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision == precision and recall == recall and (precision + recall) > 0
        else float("nan")
    )

    return ClassificationMetrics(
        n=int(y.size),
        n_positive=n_pos,
        base_rate=float(y.mean()),
        threshold=threshold,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        brier=float(brier_score_loss(y, p)),
        log_loss=float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])),
        ece=ece,
        mce=mce,
        recall=recall,
        precision=precision,
        specificity=specificity,
        f1=f1,
        accuracy=(tp + tn) / y.size,
    )


def prevalence_baseline_metrics(y_true: IntArray) -> ClassificationMetrics:
    """Metrics for a model that always predicts the base rate.

    Reported next to every arm so that "PR-AUC 0.62" can be read against the floor that
    the class balance alone provides.
    """
    y = np.asarray(y_true, dtype=np.int64).ravel()
    return compute_metrics(y, np.full(y.shape, float(y.mean()), dtype=np.float64))
