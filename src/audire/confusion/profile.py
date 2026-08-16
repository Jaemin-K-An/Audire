"""The per-listener confusion profile: one matrix per syllable position.

``ConfusionProfile`` is the ``C_u`` of the research plan. It is deliberately kept
separate from :class:`~audire.profile.schema.HearingProfile`: WRS is a *global*
speech-recognition factor and the confusion profile is the *local* error-structure
factor, and the project's central hypothesis is that these carry different information.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from audire.confusion.errors import ParsedTrial, parse_response
from audire.confusion.matrix import ConfusionMatrix, SmoothingSpec
from audire.hangul.inventory import Position
from audire.identity import validate_listener_id

POSITIONS: tuple[Position, ...] = (Position.ONSET, Position.NUCLEUS, Position.CODA)


@dataclass(slots=True)
class CalibrationTrial:
    """One presented stimulus and the listener's raw answer, before scoring."""

    stimulus_id: str
    target: str
    response: str
    #: Free-form condition label (e.g. ``"quiet"``, ``"snr_5"``, speaker id). Retained so
    #: that a profile can be rebuilt for a subset of conditions.
    condition: str = "default"

    def parse(self) -> ParsedTrial:
        return parse_response(self.target, self.response)


@dataclass(slots=True)
class ConfusionProfile:
    """Onset / nucleus / coda confusion matrices for one listener.

    Attributes
    ----------
    listener_id:
        Opaque identifier. Must never be a name or other direct identifier.
    matrices:
        One :class:`~audire.confusion.matrix.ConfusionMatrix` per position.
    is_synthetic:
        ``True`` for simulator output. Enforced end to end so that synthetic evidence
        can never be silently reported as observed data.
    n_trials:
        Number of calibration trials that produced this profile.
    """

    listener_id: str
    matrices: dict[Position, ConfusionMatrix]
    is_synthetic: bool
    n_trials: int = 0
    n_unusable_responses: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # The same identifier rule as the hearing profile and the store. Aggregates such
        # as ``__pooled__`` are permitted here because a group prior is a legitimate kind
        # of confusion profile; it is not permitted as a scoring subject.
        validate_listener_id(self.listener_id, allow_aggregate=True)
        missing = [p for p in POSITIONS if p not in self.matrices]
        if missing:
            raise ValueError(f"confusion profile is missing matrices for: {missing}")

    # ---------------------------------------------------------------- construction

    @classmethod
    def empty(
        cls,
        listener_id: str,
        *,
        is_synthetic: bool,
        smoothing: SmoothingSpec | None = None,
    ) -> Self:
        spec = smoothing or SmoothingSpec()
        return cls(
            listener_id=listener_id,
            matrices={p: ConfusionMatrix.empty(p, spec) for p in POSITIONS},
            is_synthetic=is_synthetic,
        )

    @classmethod
    def from_trials(
        cls,
        listener_id: str,
        trials: Iterable[CalibrationTrial | ParsedTrial],
        *,
        is_synthetic: bool,
        smoothing: SmoothingSpec | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> Self:
        """Build a profile by accumulating scored calibration trials.

        Unusable responses (blank, non-Hangul) still contribute: they populate the
        ``NO_RESPONSE`` column, so a listener who frequently fails to answer is not
        silently scored as if those trials had never happened.
        """
        profile = cls.empty(listener_id, is_synthetic=is_synthetic, smoothing=smoothing)
        profile.provenance = dict(provenance or {})
        for trial in trials:
            parsed = trial.parse() if isinstance(trial, CalibrationTrial) else trial
            profile.add_parsed(parsed)
        return profile

    def add_parsed(self, parsed: ParsedTrial) -> None:
        """Accumulate one scored trial into the matrices."""
        for obs in parsed.observations:
            self.matrices[obs.position].observe(obs.target, obs.perceived)
        self.n_trials += 1
        if parsed.response_syllable is None:
            self.n_unusable_responses += 1

    # ---------------------------------------------------------------- queries

    def matrix(self, position: Position) -> ConfusionMatrix:
        return self.matrices[position]

    def p_correct(self, position: Position, target: str) -> float:
        """Smoothed probability that ``target`` is perceived correctly at ``position``."""
        return self.matrices[position].p_correct(target)

    def evidence(self, position: Position, target: str) -> int:
        """Raw number of trials supporting the estimate for ``(position, target)``."""
        return self.matrices[position].n_observations(target)

    @property
    def total_observations(self) -> int:
        """Total position-level observations (3 per trial)."""
        return sum(m.total_observations for m in self.matrices.values())

    @property
    def coverage(self) -> dict[str, float]:
        """Fraction of each position's target alphabet that has at least one observation.

        Low coverage is the honest signal that a short calibration cannot support
        phoneme-specific claims about the unobserved categories.
        """
        return {
            p.value: len(self.matrices[p].observed_targets) / len(self.matrices[p].target_labels)
            for p in POSITIONS
        }

    def overall_accuracy(self) -> float | None:
        """Empirical position-level accuracy across all observed trials.

        Returns ``None`` when there is no evidence at all. This is *not* a WRS: it is a
        phoneme-level accuracy over AUDIRE's own calibration list, not a standardised
        word recognition score on KS-MWL-A.
        """
        total = self.total_observations
        if total == 0:
            return None
        correct = 0
        for p in POSITIONS:
            m = self.matrices[p]
            n_t = len(m.target_labels)
            correct += int(sum(m.counts[i, i] for i in range(n_t)))
        return correct / total

    def with_smoothing(self, smoothing: SmoothingSpec) -> ConfusionProfile:
        """Return a copy using a different smoothing specification (counts are shared)."""
        return ConfusionProfile(
            listener_id=self.listener_id,
            matrices={p: m.with_smoothing(smoothing) for p, m in self.matrices.items()},
            is_synthetic=self.is_synthetic,
            n_trials=self.n_trials,
            n_unusable_responses=self.n_unusable_responses,
            created_at=self.created_at,
            provenance=dict(self.provenance),
        )

    # ---------------------------------------------------------------- serialisation

    def to_dict(self) -> dict[str, Any]:
        return {
            "listener_id": self.listener_id,
            "is_synthetic": self.is_synthetic,
            "n_trials": self.n_trials,
            "n_unusable_responses": self.n_unusable_responses,
            "created_at": self.created_at,
            "coverage": self.coverage,
            "provenance": self.provenance,
            "matrices": {p.value: self.matrices[p].to_dict() for p in POSITIONS},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any], smoothing: SmoothingSpec | None = None) -> Self:
        return cls(
            listener_id=payload["listener_id"],
            matrices={
                Position(k): ConfusionMatrix.from_dict(v, smoothing)
                for k, v in payload["matrices"].items()
            },
            is_synthetic=bool(payload["is_synthetic"]),
            n_trials=int(payload.get("n_trials", 0)),
            n_unusable_responses=int(payload.get("n_unusable_responses", 0)),
            created_at=payload.get("created_at", datetime.now(UTC).isoformat()),
            provenance=payload.get("provenance", {}),
        )

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path, smoothing: SmoothingSpec | None = None) -> Self:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")), smoothing)


def pool_profiles(
    profiles: Sequence[ConfusionProfile],
    *,
    listener_id: str = "__pooled__",
    smoothing: SmoothingSpec | None = None,
) -> ConfusionProfile:
    """Pool counts across listeners into a group profile.

    The result is used as the *prior* for hierarchical shrinkage of an individual with
    few calibration trials. Pooling is only meaningful across listeners drawn from the
    same population and condition; the caller is responsible for that.

    Raises
    ------
    ValueError
        If ``profiles`` is empty, or if it mixes synthetic and non-synthetic listeners
        (which would let simulated evidence leak into a real listener's prior).
    """
    if not profiles:
        raise ValueError("cannot pool an empty sequence of profiles")
    synth = {p.is_synthetic for p in profiles}
    if len(synth) > 1:
        raise ValueError(
            "refusing to pool synthetic and non-synthetic profiles into one prior; "
            "synthetic provenance would be lost"
        )
    spec = smoothing or profiles[0].matrices[Position.ONSET].smoothing
    pooled = ConfusionProfile.empty(listener_id, is_synthetic=synth.pop(), smoothing=spec)
    for prof in profiles:
        for p in POSITIONS:
            pooled.matrices[p].counts += prof.matrices[p].counts
        pooled.n_trials += prof.n_trials
        pooled.n_unusable_responses += prof.n_unusable_responses
    pooled.provenance = {
        "pooled_from": [p.listener_id for p in profiles],
        "n_listeners": len(profiles),
    }
    return pooled
