"""ASR adapter contract.

The recogniser is deliberately behind a narrow interface so that the backend can be
swapped without touching risk or captioning. Two rules are structural, not stylistic:

1. **Per-token confidence is a separate output.** It is carried through to
   :class:`~audire.caption.word.WordRisk.asr_confidence` and never folded into listener
   risk (ADR-0010). A word the recogniser was unsure about is not a word the listener
   would mishear.
2. **Every transcript records the backend identity.** Backend name, model id, revision,
   device and compute type all go into the result provenance, so a caption export can
   always be traced to the exact recogniser that produced it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from audire.hangul.syllable import is_hangul_syllable


@dataclass(frozen=True, slots=True)
class Token:
    """One recognised word with its timing and the recogniser's own confidence."""

    text: str
    start_s: float
    end_s: float
    #: Backend-reported probability in ``[0, 1]``, or ``None`` when the backend does not
    #: expose one. Never a placeholder value — absence is represented as absence.
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.end_s < self.start_s:
            raise ValueError(
                f"token {self.text!r} ends ({self.end_s}) before it starts ({self.start_s})"
            )
        if self.start_s < 0:
            raise ValueError(f"token {self.text!r} has a negative start time")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def hangul_text(self) -> str:
        """The token's Hangul syllables only, which is what the risk model scores."""
        return "".join(ch for ch in self.text if is_hangul_syllable(ch))

    @property
    def has_hangul(self) -> bool:
        return bool(self.hangul_text)


@dataclass(frozen=True, slots=True)
class Transcript:
    """A full recognition result plus the provenance needed to reproduce it."""

    tokens: tuple[Token, ...]
    language: str
    #: Backend's own confidence that it identified the language correctly.
    language_probability: float | None
    duration_s: float
    backend: str
    model_id: str
    #: Everything needed to reproduce: revision, device, compute type, decode options.
    provenance: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def text(self) -> str:
        return " ".join(t.text for t in self.tokens)

    @property
    def hangul_tokens(self) -> tuple[Token, ...]:
        """Tokens containing at least one Hangul syllable.

        Non-Hangul tokens (numerals, Latin words, punctuation) cannot be scored by a
        Korean phoneme confusion profile. They are excluded from *risk scoring* but are
        still present in :attr:`tokens`, so a caption renderer can decide what to do with
        them rather than having them silently deleted.
        """
        return tuple(t for t in self.tokens if t.has_hangul)

    def timing_problems(self) -> list[str]:
        """Report timing defects instead of silently accepting them.

        Word timestamps are estimated, so overlap and non-monotonicity do occur. They must
        be visible: a caption built on bad timing is a usability failure, and E7 tracks
        timestamp quality separately from recognition accuracy.
        """
        problems: list[str] = []
        for i, t in enumerate(self.tokens):
            if t.end_s > self.duration_s + 0.5:
                problems.append(f"token {i} ({t.text!r}) ends past the media duration")
            if i and t.start_s < self.tokens[i - 1].start_s:
                problems.append(f"token {i} ({t.text!r}) starts before the previous token")
            if i and t.start_s < self.tokens[i - 1].end_s - 1e-6:
                problems.append(f"token {i} ({t.text!r}) overlaps the previous token")
        return problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model_id": self.model_id,
            "language": self.language,
            "language_probability": self.language_probability,
            "duration_s": self.duration_s,
            "n_tokens": len(self.tokens),
            "n_hangul_tokens": len(self.hangul_tokens),
            "timing_problems": self.timing_problems(),
            "provenance": self.provenance,
            "tokens": [asdict(t) for t in self.tokens],
        }


class ASRBackend(ABC):
    """A replaceable Korean-capable recogniser with word-level timestamps."""

    #: Stable identifier written into every transcript's provenance.
    name: str = "abstract"

    @abstractmethod
    def transcribe(self, media: Path, *, language: str = "ko") -> Transcript:
        """Recognise ``media`` and return tokens with word-level timings."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Backend identity and configuration, for provenance."""

    def is_available(self) -> bool:
        """Whether this backend can actually run here (dependencies, weights, device)."""
        return True


class ASRUnavailable(RuntimeError):
    """Raised when a backend's dependencies or model weights are not present.

    Carries an actionable message: the application surfaces this to the user rather than
    silently degrading to a different recogniser, which would corrupt provenance.
    """
