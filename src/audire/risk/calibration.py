"""Probability calibration for risk models.

The threshold caption policy consumes probabilities directly, so a model that ranks well
but is miscalibrated will show the wrong *amount* of text. Calibration is therefore an
explicit, evaluated arm rather than something applied silently.

Listener-group integrity is preserved: the calibrator is fitted on a held-out slice of
**listeners**, never on a random slice of trials, so a listener never contributes to both
the base model's training and its own calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from audire.risk.features import FeatureMatrix
from audire.risk.models import RiskModel

FloatArray = npt.NDArray[np.float64]

CalibrationMethod = Literal["none", "platt", "isotonic"]


def _split_by_listener(
    groups: npt.NDArray[np.str_], holdout_fraction: float, seed: int
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Split row indices into (base-fit, calibration-fit) masks by whole listener."""
    unique = np.unique(groups)
    if unique.size < 2:
        raise ValueError(
            "calibration needs at least two listeners so that the calibrator is fitted on "
            "listeners the base model did not see"
        )
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    n_hold = max(1, round(holdout_fraction * unique.size))
    n_hold = min(n_hold, unique.size - 1)
    hold = set(shuffled[:n_hold].tolist())
    cal_mask = np.array([g in hold for g in groups], dtype=bool)
    return ~cal_mask, cal_mask


@dataclass
class CalibratedRiskModel(RiskModel):
    """Wraps a base model with a post-hoc probability calibrator.

    Parameters
    ----------
    base:
        The model to calibrate. Refitted on the base slice during :meth:`fit`.
    method:
        ``"platt"`` fits a one-dimensional logistic regression on the base model's
        log-odds; ``"isotonic"`` fits a monotone step function; ``"none"`` passes the base
        probabilities through unchanged (useful as an explicit control arm).
    holdout_fraction:
        Fraction of *listeners* reserved for fitting the calibrator.
    """

    base: RiskModel = field(default_factory=lambda: _require_base())
    #: What the experiment configuration asked for. Never mutated, so a run record can
    #: always answer "what was requested?" independently of what was achievable.
    method: CalibrationMethod = "platt"
    holdout_fraction: float = 0.25
    seed: int = 0
    name: str = "calibrated"
    _platt: LogisticRegression | None = None
    _isotonic: IsotonicRegression | None = None
    _fitted: bool = False
    n_calibration_listeners: int = 0
    #: What actually ran. Differs from :attr:`method` only after a recorded fallback.
    effective_method: CalibrationMethod = "platt"
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.holdout_fraction < 1.0:
            raise ValueError("holdout_fraction must be in (0, 1)")
        self.name = f"{self.base.name}+{self.method}"
        self.effective_method = self.method

    def _fall_back(self, matrix: FeatureMatrix, reason: str) -> CalibratedRiskModel:
        """Refit uncalibrated, recording why.

        An earlier version overwrote ``self.method`` here. That destroyed the only record
        of what had been requested: ``describe()`` then reported ``method: "none"`` while
        ``name`` still read ``"...+platt"``, so a run that deliberately used no calibration
        was indistinguishable from one whose calibration silently failed. Comparing
        calibration arms across folds is exactly the question E22 asks, and it cannot be
        answered from an artifact that has forgotten the question.
        """
        self.effective_method = "none"
        self.fallback_reason = reason
        self.base.fit(matrix)
        self._fitted = True
        return self

    def fit(self, matrix: FeatureMatrix) -> CalibratedRiskModel:
        if matrix.y is None:
            raise ValueError("cannot fit a calibrator without labels")

        self.effective_method = self.method
        self.fallback_reason = None
        self._platt = self._isotonic = None

        if self.method == "none":
            self.base.fit(matrix)
            self._fitted = True
            return self

        base_mask, cal_mask = _split_by_listener(matrix.groups, self.holdout_fraction, self.seed)
        self.n_calibration_listeners = int(np.unique(matrix.groups[cal_mask]).size)

        self.base.fit(_subset(matrix, base_mask))
        raw = self.base.predict_proba(_subset(matrix, cal_mask))
        y_cal = matrix.y[cal_mask]

        if np.unique(y_cal).size < 2:
            # Not enough label variety on the calibration slice to fit anything sensible.
            # Fall back to the uncalibrated base and say so, rather than fitting noise.
            return self._fall_back(
                matrix,
                f"교정 슬라이스({self.n_calibration_listeners}명)의 라벨이 한 종류뿐이라 "
                f"{self.method} 교정기를 적합할 수 없었습니다",
            )

        if self.method == "platt":
            self._platt = LogisticRegression(max_iter=1000)
            self._platt.fit(_logit(raw).reshape(-1, 1), y_cal)
        else:
            self._isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self._isotonic.fit(raw, y_cal)

        self._fitted = True
        return self

    def predict_proba(self, matrix: FeatureMatrix) -> FloatArray:
        raw = self.base.predict_proba(matrix)
        if self.effective_method == "none":
            return raw
        if self._platt is not None:
            out: FloatArray = self._platt.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]
            return out.astype(np.float64)
        if self._isotonic is not None:
            iso: FloatArray = np.clip(
                np.asarray(self._isotonic.predict(raw), dtype=np.float64), 0.0, 1.0
            )
            return iso
        raise ValueError("calibrator is not fitted")  # pragma: no cover

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": "calibrated",
            # `method` is what was asked for and `effective_method` is what ran; reporting
            # both is what makes a fallback visible instead of merely absent.
            "requested_method": self.method,
            "effective_method": self.effective_method,
            "fell_back": self.effective_method != self.method,
            "fallback_reason": self.fallback_reason,
            "holdout_fraction": self.holdout_fraction,
            "n_calibration_listeners": self.n_calibration_listeners,
            "base": self.base.describe(),
        }


def _require_base() -> RiskModel:  # pragma: no cover - guards a programming error
    raise TypeError("CalibratedRiskModel requires an explicit `base` model")


def _logit(p: FloatArray, eps: float = 1e-6) -> FloatArray:
    q = np.clip(p, eps, 1 - eps)
    out: FloatArray = np.log(q / (1 - q))
    return out


def _subset(matrix: FeatureMatrix, mask: npt.NDArray[np.bool_]) -> FeatureMatrix:
    return FeatureMatrix(
        X=matrix.X[mask],
        feature_names=matrix.feature_names,
        groups=matrix.groups[mask],
        y=None if matrix.y is None else matrix.y[mask],
        meta=matrix.meta,
    )
