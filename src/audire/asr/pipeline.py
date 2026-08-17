"""End-to-end pipeline: media → transcript → personalized word risk → selective captions.

This is the single production path. The web application and the research evaluation both
call :func:`caption_media`, so a caption a user sees is produced by exactly the code that
the reported numbers were produced by.

The ASR-versus-listener-risk separation (ADR-0010) is enforced structurally here: the
recogniser's confidence is attached to :attr:`~audire.caption.word.WordRisk.asr_confidence`
and the risk model never sees it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from audire.asr.base import ASRBackend, Transcript
from audire.caption.policy import CaptionPolicy, FullCaptionPolicy, caption_ratio
from audire.caption.word import CaptionDecision, WordRisk
from audire.config.logging import get_logger
from audire.confusion.profile import ConfusionProfile
from audire.identity import validate_listener_id
from audire.profile.schema import HearingProfile
from audire.risk.features import WordContext, phoneme_risks
from audire.risk.models import MODEL_VERSION, WordScorer

log = get_logger(__name__)


@dataclass(slots=True)
class CaptionResult:
    """Everything one media file produced."""

    listener_id: str
    words: list[WordRisk]
    transcript: Transcript
    policy: dict[str, Any]
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def caption_ratio(self) -> float:
        return caption_ratio(self.words)

    @property
    def n_shown(self) -> int:
        return sum(w.is_shown for w in self.words)

    def summary(self) -> dict[str, Any]:
        return {
            "listener_id": self.listener_id,
            "n_words": len(self.words),
            "n_shown": self.n_shown,
            "caption_ratio": self.caption_ratio,
            "caption_reduction_ratio": 1.0 - self.caption_ratio,
            "policy": self.policy,
            "asr": {
                "backend": self.transcript.backend,
                "model_id": self.transcript.model_id,
                "language": self.transcript.language,
                "language_probability": self.transcript.language_probability,
                "n_tokens": len(self.transcript),
                "n_scoreable_tokens": len(self.transcript.hangul_tokens),
                "timing_problems": self.transcript.timing_problems(),
            },
            "provenance": self.provenance,
        }


class IncompleteProfile(ValueError):
    """Raised when the listener's profile cannot support the requested scoring arm."""


def check_identity(
    listener_id: str,
    hearing: HearingProfile | None,
    confusion: ConfusionProfile | None,
) -> list[str]:
    """Return the reasons these inputs do not describe one and the same listener.

    Two invariants, both of which used to be unchecked:

    1. **The three identifiers must agree.** Scoring listener A with listener B's profile
       produces a confident, plausible-looking number for the wrong person. In an
       accessibility tool that means captioning tuned to someone else.
    2. **Synthetic provenance must agree.** Combining a real hearing profile with a
       synthetic confusion profile launders simulated evidence into a result about a real
       person, which ``docs/RISK_REGISTER.md`` S1 forbids.
    """
    problems: list[str] = []
    try:
        validate_listener_id(listener_id)
    except ValueError as exc:
        problems.append(str(exc))

    for label, profile in (("hearing", hearing), ("confusion", confusion)):
        if profile is None:
            continue
        if profile.listener_id != listener_id:
            problems.append(
                f"{label} profile belongs to listener {profile.listener_id!r} but "
                f"{listener_id!r} was requested; refusing to score one listener with "
                f"another listener's profile"
            )

    if (
        hearing is not None
        and confusion is not None
        and hearing.is_synthetic != confusion.is_synthetic
    ):
        problems.append(
            f"provenance mismatch: hearing profile is_synthetic="
            f"{hearing.is_synthetic} but confusion profile is_synthetic="
            f"{confusion.is_synthetic}. 합성(synthetic) 근거와 실측 근거를 섞을 수 "
            f"없습니다."
        )
    return problems


def check_ready(
    scorer: WordScorer,
    hearing: HearingProfile | None,
    confusion: ConfusionProfile | None,
    *,
    listener_id: str | None = None,
    min_calibration_trials: int = 10,
) -> list[str]:
    """Return the reasons this listener cannot be scored yet. Empty means ready.

    Fails loudly and specifically rather than producing a plausible-looking score from a
    profile that has no evidence behind it. When ``listener_id`` is supplied the identity
    invariants in :func:`check_identity` are enforced as well.
    """
    problems: list[str] = []
    if listener_id is not None:
        problems.extend(check_identity(listener_id, hearing, confusion))
    blocks = set(scorer.spec.blocks)
    if ({"pta", "clinical"} & blocks) and hearing is None:
        problems.append(f"arm {scorer.spec.name!r} needs a hearing profile; none was supplied")
    if "confusion" in blocks:
        if confusion is None:
            problems.append(
                f"arm {scorer.spec.name!r} needs a confusion profile; run a calibration first"
            )
        elif confusion.n_trials < min_calibration_trials:
            problems.append(
                f"only {confusion.n_trials} calibration trials; at least "
                f"{min_calibration_trials} are needed before per-phoneme estimates mean "
                f"anything"
            )
    if not scorer.model.is_fitted:
        problems.append("the risk model is not fitted")
    return problems


def score_transcript(
    transcript: Transcript,
    scorer: WordScorer,
    *,
    listener_id: str,
    hearing: HearingProfile | None,
    confusion: ConfusionProfile | None,
    snr_db: float = 20.0,
    speaker: str = "unknown",
) -> list[WordRisk]:
    """Attach a personalized mishearing risk to every token in ``transcript``.

    Tokens with no Hangul (numerals, Latin words) cannot be scored by a Korean phoneme
    confusion profile. They are kept with ``listener_risk = 0.0`` and a ``meta`` flag
    rather than being dropped, so the caption renderer decides what to do with them.
    """
    problems = check_ready(scorer, hearing, confusion, listener_id=listener_id)
    if problems:
        raise IncompleteProfile("; ".join(problems))

    scoreable = [t for t in transcript.tokens if t.has_hangul]
    by_token: dict[int, float] = {}
    if scoreable:
        contexts = [WordContext(snr_db=snr_db, speaker=speaker)] * len(scoreable)
        risks = scorer.score(
            listener_id, [t.hangul_text for t in scoreable], contexts, hearing, confusion
        )
        by_token = {id(t): float(r) for t, r in zip(scoreable, risks, strict=True)}

    out: list[WordRisk] = []
    for token in transcript.tokens:
        risk = by_token.get(id(token), 0.0)
        contributions = (
            tuple(phoneme_risks(token.hangul_text, confusion))
            if (token.has_hangul and confusion is not None)
            else ()
        )
        out.append(
            WordRisk(
                text=token.text,
                start_s=token.start_s,
                end_s=token.end_s,
                listener_risk=risk,
                # Kept strictly separate from listener_risk (ADR-0010).
                asr_confidence=token.confidence,
                model_version=MODEL_VERSION,
                model_arm=scorer.spec.name,
                decision=CaptionDecision.HIDDEN,
                policy="unapplied",
                contributions=contributions,
                meta={
                    "scoreable": token.has_hangul,
                    "reason_not_scored": None if token.has_hangul else "no Hangul syllables",
                    "asr_backend": transcript.backend,
                },
            )
        )
    return out


def media_digest(media: Path, *, chunk: int = 1 << 20) -> str:
    """SHA-256 of a media file, streamed so a large video does not enter memory.

    Serves two purposes. It identifies exactly which file produced a caption, which the
    filename does not — files get renamed, and two different recordings can share a name.
    And it gives the logs something to say about the input that is not the filename:
    ``상담_김철수_2026.mp4`` is itself personal data, and a captioning system handles
    precisely the material people do not expect to be shipped to a log aggregator.
    """
    h = hashlib.sha256()
    with media.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def caption_media(
    media: Path,
    backend: ASRBackend,
    scorer: WordScorer,
    *,
    listener_id: str,
    hearing: HearingProfile | None,
    confusion: ConfusionProfile | None,
    policy: CaptionPolicy | None = None,
    language: str = "ko",
    snr_db: float = 20.0,
    speaker: str = "unknown",
) -> CaptionResult:
    """Run the complete pipeline on one media file."""
    active_policy = policy or FullCaptionPolicy()
    # After transcription, not before: the backend validates the media and reports a
    # specific reason when it is unusable. Hashing first replaced that with a bare
    # FileNotFoundError from this function.
    transcript = backend.transcribe(media, language=language)
    digest = media_digest(media)
    scored = score_transcript(
        transcript,
        scorer,
        listener_id=listener_id,
        hearing=hearing,
        confusion=confusion,
        snr_db=snr_db,
        speaker=speaker,
    )
    words = active_policy.apply(scored)

    result = CaptionResult(
        listener_id=listener_id,
        words=words,
        transcript=transcript,
        policy=active_policy.describe(),
        provenance={
            "media": media.name,
            # The digest, not the name, is what identifies the input reproducibly.
            "media_sha256": digest,
            "media_bytes": media.stat().st_size,
            "asr": backend.describe(),
            "model": scorer.describe(),
            "listener": {
                "has_hearing_profile": hearing is not None,
                "calibration_trials": None if confusion is None else confusion.n_trials,
                "calibration_coverage": None if confusion is None else confusion.coverage,
                "is_synthetic": None if confusion is None else confusion.is_synthetic,
            },
            "context": {"snr_db": snr_db, "speaker": speaker},
        },
    )
    log.info(
        "pipeline.done",
        # Digest rather than filename: log sinks travel further than the media does.
        media_sha256=digest[:16],
        n_words=len(words),
        n_shown=result.n_shown,
        caption_ratio=round(result.caption_ratio, 3),
        policy=active_policy.label,
    )
    return result
