"""``WordRisk`` — the record that carries a word from ASR through risk to caption.

The single most important structural rule here is that **ASR uncertainty and listener
mishearing risk are separate fields** (ADR-0010). A word the recogniser was unsure about
is not a word the listener would mishear; conflating them would let ASR failure
masquerade as a personalization result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from audire.risk.features import PhonemeRisk


class CaptionDecision(StrEnum):
    """Why a word is or is not shown."""

    #: Selected by the active policy because its listener risk was high enough.
    SHOWN_HIGH_RISK = "shown_high_risk"
    #: Shown because the caption mode displays everything.
    SHOWN_FULL_MODE = "shown_full_mode"
    #: Shown because the ASR hypothesis itself is unreliable and the listener should see
    #: the uncertainty. Recorded distinctly so it never counts as a personalization hit.
    SHOWN_LOW_ASR_CONFIDENCE = "shown_low_asr_confidence"
    HIDDEN = "hidden"


@dataclass(frozen=True, slots=True)
class WordRisk:
    """One recognised word with its personalized risk and caption decision."""

    #: Surface token exactly as produced by the recogniser.
    text: str
    start_s: float
    end_s: float
    #: ``P(this listener mishears this word)``. The only quantity the policies rank on.
    listener_risk: float
    #: Recogniser confidence for this token, or ``None`` when the backend does not
    #: expose one. **Never** merged into ``listener_risk``.
    asr_confidence: float | None
    model_version: str
    #: Which ablation arm / feature set produced ``listener_risk``.
    model_arm: str
    decision: CaptionDecision
    #: The policy that made the decision, e.g. ``"threshold(tau=0.42)"``.
    policy: str
    #: Per-phoneme evidence behind the score, for the explanation panel.
    contributions: tuple[PhonemeRisk, ...] = ()
    #: Free-form extra provenance (listener id, calibration size, ...).
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.end_s < self.start_s:
            raise ValueError(
                f"word {self.text!r} ends ({self.end_s}) before it starts ({self.start_s})"
            )
        if not 0.0 <= self.listener_risk <= 1.0:
            raise ValueError(f"listener_risk must be in [0, 1], got {self.listener_risk}")
        if self.asr_confidence is not None and not 0.0 <= self.asr_confidence <= 1.0:
            raise ValueError(f"asr_confidence must be in [0, 1], got {self.asr_confidence}")

    @property
    def is_shown(self) -> bool:
        return self.decision is not CaptionDecision.HIDDEN

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def explanation(self, top_k: int = 3) -> dict[str, Any]:
        """Structured explanation of why this word received its score.

        Reports the phonemes that contributed most risk, how much calibration evidence
        each rests on, and the ASR confidence *as a separate line*.
        """
        ranked = sorted(self.contributions, key=lambda c: c.p_correct)[:top_k]
        return {
            "word": self.text,
            "listener_risk": self.listener_risk,
            "asr_confidence": self.asr_confidence,
            "asr_note": (
                "ASR confidence is reported separately and does not contribute to the "
                "listener risk score."
            ),
            "model_version": self.model_version,
            "model_arm": self.model_arm,
            "decision": self.decision.value,
            "policy": self.policy,
            "weakest_phonemes": [
                {
                    "position": c.position.value,
                    "phoneme": c.target,
                    "p_correct": c.p_correct,
                    "n_calibration_observations": c.n_observations,
                    "estimate_from_prior_only": c.n_observations == 0,
                    "backed_off_to_surface_form": c.backed_off,
                    "likely_confusions": [
                        {"perceived": label, "probability": prob, "count": count}
                        for label, prob, count in c.top_confusions
                    ],
                }
                for c in ranked
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "listener_risk": self.listener_risk,
            "asr_confidence": self.asr_confidence,
            "model_version": self.model_version,
            "model_arm": self.model_arm,
            "decision": self.decision.value,
            "policy": self.policy,
            "shown": self.is_shown,
            "explanation": self.explanation(),
            "meta": self.meta,
        }
