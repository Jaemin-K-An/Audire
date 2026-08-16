"""RQ1 — the listener-level ablation across feature arms and model families.

The runner executes, for every ``(arm, model)`` combination and every seed in the config:

1. build the design matrix for that arm from a cohort;
2. split by **listener** (stratified group folds), asserting no leakage on every fold;
3. fit on the training listeners and predict the held-out listeners;
4. pool the out-of-fold predictions and compute discrimination + calibration metrics;
5. bootstrap listener-level confidence intervals;
6. run paired bootstrap contrasts against the preregistered reference arms.

Nothing is selected after the fact: the arms, models, seeds, fold count, ECE bin count and
contrasts all come from the config.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from audire.config.logging import get_logger
from audire.confusion.profile import ConfusionProfile
from audire.eval.bootstrap import (
    DEFAULT_N_BOOTSTRAP,
    Interval,
    bootstrap_metric,
    metric_statistic,
    paired_bootstrap_difference,
)
from audire.eval.metrics import (
    DEFAULT_ECE_BINS,
    ClassificationMetrics,
    compute_metrics,
    prevalence_baseline_metrics,
    reliability_curve,
)
from audire.eval.splits import (
    assert_no_listener_leakage,
    assert_prior_fitted_on_train_only,
    listener_folds,
)
from audire.risk.calibration import CalibratedRiskModel, CalibrationMethod
from audire.risk.features import (
    ABLATION_ARMS,
    FeatureMatrix,
    FeatureSpec,
    WordContext,
    build_matrix,
)
from audire.risk.hierarchical import DEFAULT_GROUP_ALPHA, apply_group_prior, fit_group_prior
from audire.risk.models import make_model
from audire.sim.cohort import Cohort

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

log = get_logger(__name__)

#: Preregistered contrasts. Each entry is ``(arm, reference_arm)`` and answers a question
#: from the research plan directly.
DEFAULT_CONTRASTS: tuple[tuple[str, str], ...] = (
    # RQ1: does the confusion profile add anything beyond the clinical measures?
    ("clinical_plus_confusion", "clinical"),
    # RQ1: and beyond the audiogram alone?
    ("clinical_plus_confusion", "pta_only"),
    # Is the confusion profile alone already enough?
    ("confusion_only", "clinical"),
    # Does any listener personalization help at all?
    ("clinical_plus_confusion", "word_context_only"),
)


@dataclass(frozen=True, slots=True)
class ArmResult:
    """Out-of-fold results for one ``(arm, model)`` combination at one seed."""

    arm: str
    model: str
    seed: int
    calibration: str
    n_listeners: int
    n_trials: int
    n_features: int
    metrics: ClassificationMetrics
    prevalence_floor: ClassificationMetrics
    intervals: dict[str, Interval]
    reliability: dict[str, list[float]]
    model_description: dict[str, Any]
    fold_sizes: list[tuple[int, int]]
    #: Out-of-fold predictions, retained so that paired contrasts and the caption study
    #: reuse exactly the predictions that produced these metrics.
    y_true: IntArray = field(repr=False, default_factory=lambda: np.zeros(0, dtype=np.int64))
    y_prob: FloatArray = field(repr=False, default_factory=lambda: np.zeros(0, dtype=np.float64))
    groups: npt.NDArray[np.str_] = field(
        repr=False, default_factory=lambda: np.zeros(0, dtype=np.str_)
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "model": self.model,
            "seed": self.seed,
            "calibration": self.calibration,
            "n_listeners": self.n_listeners,
            "n_trials": self.n_trials,
            "n_features": self.n_features,
            "metrics": self.metrics.to_dict(),
            "prevalence_floor": self.prevalence_floor.to_dict(),
            "intervals": {k: v.to_dict() for k, v in self.intervals.items()},
            "reliability": self.reliability,
            "model_description": self.model_description,
            "fold_sizes": self.fold_sizes,
        }


@dataclass(frozen=True, slots=True)
class ContrastResult:
    """A paired listener-level contrast between two arms."""

    metric: str
    arm: str
    reference: str
    model: str
    seed: int
    interval: Interval

    @property
    def is_significant(self) -> bool:
        """Whether the paired interval excludes zero. Not a p-value; report the interval."""
        return self.interval.excludes_zero

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "arm": self.arm,
            "reference": self.reference,
            "model": self.model,
            "seed": self.seed,
            "interval": self.interval.to_dict(),
            "excludes_zero": self.is_significant,
        }


# --------------------------------------------------------------------------- matrix building


def cohort_matrix(
    cohort: Cohort,
    spec: FeatureSpec,
    *,
    profiles: Mapping[str, ConfusionProfile] | None = None,
) -> FeatureMatrix:
    """Build the design matrix and labels for a whole cohort under one arm.

    ``profiles`` replaces each listener's confusion profile by listener id. It exists for
    fold-safe hierarchical shrinkage: the caller passes profiles shrunk toward a prior
    fitted on that fold's training listeners only. Row order does not depend on it, so
    matrices built with different profiles remain row-aligned and paired contrasts stay
    valid.
    """
    rows = []
    labels: list[int] = []
    for record in cohort.records:
        confusion = record.estimated_confusion if profiles is None else profiles[record.listener_id]
        for trial in record.word_trials:
            rows.append(
                (
                    record.listener_id,
                    trial.word,
                    WordContext(snr_db=trial.snr_db, speaker=trial.speaker),
                    record.hearing,
                    confusion,
                )
            )
            labels.append(int(trial.misheard))
    return build_matrix(spec, rows, labels)


# --------------------------------------------------------------------------- evaluation


def _safe(fn: Any) -> Any:
    def wrapped(y: IntArray, p: FloatArray) -> float:
        if np.unique(y).size < 2:
            return float("nan")
        return float(fn(y, p))

    return wrapped


_BOOTSTRAP_METRICS: dict[str, Any] = {
    "pr_auc": _safe(average_precision_score),
    "roc_auc": _safe(roc_auc_score),
    "brier": lambda y, p: float(brier_score_loss(y, p)),
}


def evaluate_arm(
    cohort: Cohort,
    arm: str,
    model_name: str,
    *,
    seed: int,
    n_splits: int = 5,
    calibration: CalibrationMethod = "none",
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ece_bins: int = DEFAULT_ECE_BINS,
    threshold: float = 0.5,
    model_kwargs: dict[str, Any] | None = None,
    group_shrinkage: bool = False,
    group_alpha: float = DEFAULT_GROUP_ALPHA,
) -> ArmResult:
    """Run listener-level cross-validation for one ``(arm, model)`` combination.

    ``group_shrinkage`` enables Phase-D hierarchical shrinkage of the listener confusion
    profiles toward a population prior. The prior is **refitted inside every fold from the
    training listeners only**, which is why the design matrix is rebuilt per fold in that
    mode: a prior fitted once over the whole cohort would carry held-out listeners'
    responses into the training representation and inflate every metric reported here.
    """
    spec = FeatureSpec.arm(arm, speakers=tuple(sorted({*cohort.config.speakers, "unknown"})))
    matrix = cohort_matrix(cohort, spec)
    assert matrix.y is not None  # build_matrix was given labels

    folds = listener_folds(matrix.groups, matrix.y, n_splits=n_splits, stratify=True, seed=seed)

    oof = np.full(matrix.y.shape, np.nan, dtype=np.float64)
    description: dict[str, Any] = {}
    fold_sizes: list[tuple[int, int]] = []
    shrinkage_log: list[dict[str, Any]] = []

    for fold in folds:
        # Re-assert on every fold: the guard is cheap and this is the failure mode that
        # would silently inflate every number in the report.
        assert_no_listener_leakage(matrix.groups, fold.train_idx, fold.test_idx)

        fold_matrix = matrix
        if group_shrinkage:
            train_ids = set(fold.train_listeners)
            prior = fit_group_prior(
                [r.estimated_confusion for r in cohort.records if r.listener_id in train_ids]
            )
            # The prior itself must be provably free of held-out listeners; the group split
            # alone does not establish that.
            assert_prior_fitted_on_train_only(prior.fitted_from, matrix.groups, fold.test_idx)
            # Every listener is shrunk toward the *same*, training-only prior. A held-out
            # listener keeps their own counts and gains no information about themselves.
            shrunk = {
                r.listener_id: apply_group_prior(r.estimated_confusion, prior, alpha=group_alpha)
                for r in cohort.records
            }
            fold_matrix = cohort_matrix(cohort, spec, profiles=shrunk)
            if not np.array_equal(fold_matrix.groups, matrix.groups):
                raise RuntimeError("shrunk matrix is not row-aligned with the reference matrix")
            shrinkage_log.append({"fold": fold.index, "alpha": group_alpha, **prior.describe()})

        base = make_model(model_name, **(model_kwargs or {}))
        model = (
            base
            if calibration == "none"
            else CalibratedRiskModel(base=base, method=calibration, seed=seed)
        )
        model.fit(_subset(fold_matrix, fold.train_idx))
        oof[fold.test_idx] = model.predict_proba(_subset(fold_matrix, fold.test_idx))
        description = model.describe()
        fold_sizes.append((fold.n_train, fold.n_test))

    if shrinkage_log:
        description = {**description, "group_shrinkage": shrinkage_log}

    if np.any(~np.isfinite(oof)):
        raise RuntimeError(
            "some rows received no out-of-fold prediction; the fold partition is incomplete"
        )

    y = matrix.y
    metrics = compute_metrics(y, oof, threshold=threshold, n_bins=ece_bins)

    intervals = {
        name: bootstrap_metric(
            matrix.groups,
            metric_statistic(y, oof, fn),
            n_resamples=n_bootstrap,
            seed=seed,
        )
        for name, fn in _BOOTSTRAP_METRICS.items()
    }

    log.info(
        "ablation.arm_done",
        arm=arm,
        model=model_name,
        seed=seed,
        calibration=calibration,
        pr_auc=round(metrics.pr_auc, 4),
        brier=round(metrics.brier, 4),
        ece=round(metrics.ece, 4),
    )

    return ArmResult(
        arm=arm,
        model=model_name,
        seed=seed,
        calibration=calibration,
        n_listeners=int(np.unique(matrix.groups).size),
        n_trials=len(matrix),
        n_features=len(matrix.feature_names),
        metrics=metrics,
        prevalence_floor=prevalence_baseline_metrics(y),
        intervals=intervals,
        reliability=reliability_curve(y, oof, ece_bins),
        model_description=description,
        fold_sizes=fold_sizes,
        y_true=y,
        y_prob=oof,
        groups=matrix.groups,
    )


def contrast(
    a: ArmResult,
    b: ArmResult,
    *,
    metric: str = "pr_auc",
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 0,
) -> ContrastResult:
    """Paired listener-level contrast ``a - b`` on the identical evaluation rows."""
    if not np.array_equal(a.y_true, b.y_true) or not np.array_equal(a.groups, b.groups):
        raise ValueError(
            "paired contrast requires the two arms to have been evaluated on identical "
            "rows in identical order"
        )
    fn = _BOOTSTRAP_METRICS[metric]
    return ContrastResult(
        metric=metric,
        arm=a.arm,
        reference=b.arm,
        model=a.model,
        seed=a.seed,
        interval=paired_bootstrap_difference(
            a.groups,
            metric_statistic(a.y_true, a.y_prob, fn),
            metric_statistic(b.y_true, b.y_prob, fn),
            n_resamples=n_bootstrap,
            seed=seed,
        ),
    )


def run_ablation(
    cohort: Cohort,
    *,
    arms: tuple[str, ...] = tuple(ABLATION_ARMS),
    models: tuple[str, ...] = ("logistic",),
    seed: int = 0,
    n_splits: int = 5,
    calibration: CalibrationMethod = "none",
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ece_bins: int = DEFAULT_ECE_BINS,
    contrasts: tuple[tuple[str, str], ...] = DEFAULT_CONTRASTS,
    contrast_metrics: tuple[str, ...] = ("pr_auc", "brier"),
) -> tuple[list[ArmResult], list[ContrastResult]]:
    """Run every ``(arm, model)`` combination and the preregistered contrasts."""
    results: dict[tuple[str, str], ArmResult] = {}
    for model_name in models:
        for arm in arms:
            if model_name == "phoneme_independence" and "confusion" not in ABLATION_ARMS[arm]:
                # The deterministic baseline is only defined where C_u is available.
                continue
            results[(arm, model_name)] = evaluate_arm(
                cohort,
                arm,
                model_name,
                seed=seed,
                n_splits=n_splits,
                calibration=calibration,
                n_bootstrap=n_bootstrap,
                ece_bins=ece_bins,
            )

    contrast_results: list[ContrastResult] = []
    for model_name in models:
        for arm, reference in contrasts:
            key_a, key_b = (arm, model_name), (reference, model_name)
            if key_a not in results or key_b not in results:
                continue
            for metric in contrast_metrics:
                contrast_results.append(
                    contrast(
                        results[key_a],
                        results[key_b],
                        metric=metric,
                        n_bootstrap=n_bootstrap,
                        seed=seed,
                    )
                )

    return list(results.values()), contrast_results


def _subset(matrix: FeatureMatrix, idx: IntArray) -> FeatureMatrix:
    return FeatureMatrix(
        X=matrix.X[idx],
        feature_names=matrix.feature_names,
        groups=matrix.groups[idx],
        y=None if matrix.y is None else matrix.y[idx],
        meta=matrix.meta,
    )
