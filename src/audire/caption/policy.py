"""Selective-caption policies.

Two policies are supported, as required by the research plan:

``ThresholdPolicy``
    Show word *w* iff ``R(w, u) > tau_u``. Consumes probabilities directly, so it depends
    on the model being calibrated; the amount of text shown varies with the material.

``BudgetPolicy``
    Show the highest-risk ``B`` fraction of words. Calibration-free — it only needs the
    *ranking* — and the amount of text is fixed in advance, which is what makes matched
    caption-ratio comparisons possible.

Both can additionally surface words the recogniser itself was unsure about. That decision
is recorded as :data:`~audire.caption.word.CaptionDecision.SHOWN_LOW_ASR_CONFIDENCE` so it
never counts as a personalization hit when the caption study is scored (ADR-0010).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import numpy.typing as npt

from audire.caption.word import CaptionDecision, WordRisk

FloatArray = npt.NDArray[np.float64]

#: Caption budgets evaluated by the RQ2 study.
STANDARD_BUDGETS: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50)


class CaptionPolicy(ABC):
    """Decides which words are displayed."""

    @abstractmethod
    def apply(self, words: list[WordRisk]) -> list[WordRisk]:
        """Return the words with their :attr:`WordRisk.decision` and policy label set."""

    @property
    @abstractmethod
    def label(self) -> str: ...

    def describe(self) -> dict[str, Any]:
        return {"policy": self.label}


def caption_ratio(words: list[WordRisk]) -> float:
    """Fraction of words displayed."""
    return sum(w.is_shown for w in words) / len(words) if words else 0.0


def caption_reduction_ratio(words: list[WordRisk]) -> float:
    """``CRR = 1 - captioned/all``: how much less text than full captions is shown."""
    return 1.0 - caption_ratio(words)


# --------------------------------------------------------------------------- policies


@dataclass(frozen=True, slots=True)
class FullCaptionPolicy(CaptionPolicy):
    """Show everything. The accessibility baseline and the B0 comparison arm."""

    def apply(self, words: list[WordRisk]) -> list[WordRisk]:
        return [
            replace(w, decision=CaptionDecision.SHOWN_FULL_MODE, policy=self.label) for w in words
        ]

    @property
    def label(self) -> str:
        return "full"


@dataclass(frozen=True, slots=True)
class ThresholdPolicy(CaptionPolicy):
    """Show word *w* iff ``listener_risk > tau``.

    ``tau`` may be personalized per listener; :func:`personalized_threshold` derives one
    from a validation set, which is what RQ3 tests against a single global threshold.
    """

    tau: float = 0.5
    #: Words whose ASR confidence falls below this are shown regardless of listener risk.
    asr_confidence_floor: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.tau <= 1.0:
            raise ValueError(f"tau must be in [0, 1], got {self.tau}")
        if self.asr_confidence_floor is not None and not 0.0 <= self.asr_confidence_floor <= 1.0:
            raise ValueError("asr_confidence_floor must be in [0, 1]")

    def apply(self, words: list[WordRisk]) -> list[WordRisk]:
        out: list[WordRisk] = []
        for w in words:
            if w.listener_risk > self.tau:
                decision = CaptionDecision.SHOWN_HIGH_RISK
            elif _asr_uncertain(w, self.asr_confidence_floor):
                decision = CaptionDecision.SHOWN_LOW_ASR_CONFIDENCE
            else:
                decision = CaptionDecision.HIDDEN
            out.append(replace(w, decision=decision, policy=self.label))
        return out

    @property
    def label(self) -> str:
        base = f"threshold(tau={self.tau:.4f})"
        if self.asr_confidence_floor is not None:
            base += f"+asr_floor({self.asr_confidence_floor:.2f})"
        return base

    def describe(self) -> dict[str, Any]:
        return {
            "policy": "threshold",
            "tau": self.tau,
            "asr_confidence_floor": self.asr_confidence_floor,
            "label": self.label,
        }


@dataclass(frozen=True, slots=True)
class BudgetPolicy(CaptionPolicy):
    """Show the highest-risk ``budget`` fraction of words.

    Ties are broken by earlier start time, then by original order, so the policy is
    deterministic. Calibration-free: only the ranking matters.
    """

    budget: float = 0.20
    asr_confidence_floor: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.budget <= 1.0:
            raise ValueError(f"budget must be in [0, 1], got {self.budget}")
        if self.asr_confidence_floor is not None and not 0.0 <= self.asr_confidence_floor <= 1.0:
            raise ValueError("asr_confidence_floor must be in [0, 1]")

    def apply(self, words: list[WordRisk]) -> list[WordRisk]:
        if not words:
            return []
        n_show = round(self.budget * len(words))
        order = sorted(
            range(len(words)),
            key=lambda i: (-words[i].listener_risk, words[i].start_s, i),
        )
        chosen = set(order[:n_show])
        out: list[WordRisk] = []
        for i, w in enumerate(words):
            if i in chosen:
                decision = CaptionDecision.SHOWN_HIGH_RISK
            elif _asr_uncertain(w, self.asr_confidence_floor):
                decision = CaptionDecision.SHOWN_LOW_ASR_CONFIDENCE
            else:
                decision = CaptionDecision.HIDDEN
            out.append(replace(w, decision=decision, policy=self.label))
        return out

    @property
    def label(self) -> str:
        base = f"budget(B={self.budget:.2f})"
        if self.asr_confidence_floor is not None:
            base += f"+asr_floor({self.asr_confidence_floor:.2f})"
        return base

    def describe(self) -> dict[str, Any]:
        return {
            "policy": "budget",
            "budget": self.budget,
            "asr_confidence_floor": self.asr_confidence_floor,
            "label": self.label,
        }


def _asr_uncertain(word: WordRisk, floor: float | None) -> bool:
    return floor is not None and word.asr_confidence is not None and word.asr_confidence < floor


# --------------------------------------------------------------------------- thresholds


def personalized_threshold(risks: FloatArray, target_ratio: float) -> float:
    """Return the threshold that shows ``target_ratio`` of ``risks`` for this listener.

    This is the RQ3 mechanism: instead of one global ``tau`` for everyone, each listener
    gets the ``tau`` that hits their own caption budget given their own risk distribution.
    A listener whose risks are uniformly high would otherwise see almost everything
    captioned under a global threshold.
    """
    if not 0.0 <= target_ratio <= 1.0:
        raise ValueError(f"target_ratio must be in [0, 1], got {target_ratio}")
    if risks.size == 0:
        return 1.0
    if target_ratio == 0.0:
        return 1.0
    if target_ratio == 1.0:
        return 0.0
    # The quantile at 1 - ratio is the smallest value that leaves `ratio` above it.
    q = float(np.quantile(risks, 1.0 - target_ratio))
    # Nudge below the quantile so that a word exactly at it is included.
    return float(np.nextafter(q, -np.inf))


def global_threshold(risks_by_listener: dict[str, FloatArray], target_ratio: float) -> float:
    """One threshold for everyone, chosen to hit ``target_ratio`` over the pooled words.

    The RQ3 comparator. Pooling means listeners with high overall risk get more captions
    than their share of the budget and low-risk listeners get fewer.
    """
    pooled = (
        np.concatenate(list(risks_by_listener.values()))
        if risks_by_listener
        else np.zeros(0, dtype=np.float64)
    )
    return personalized_threshold(pooled, target_ratio)
