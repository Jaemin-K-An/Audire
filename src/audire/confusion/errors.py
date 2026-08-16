"""Calibration-response parsing and Korean monosyllabic error classification.

A calibration trial presents one Korean monosyllable and records what the listener
reported. This module turns a ``(target, response)`` pair into

* a per-position ``(target_category, perceived_category)`` observation triple that the
  confusion matrices consume, and
* an interpretable error label.

Error taxonomy
--------------
AUDIRE's taxonomy is *structurally* comparable to the substitution / addition /
omission / compound scheme used by Joo et al. (2026, DOI 10.21848/asr.250214) but is
defined independently here from the orthographic decomposition. It is **not** a
reimplementation of that paper's 10-category coding scheme and results are not
interchangeable with it. See docs/DECISIONS.md ADR-0005.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from audire.hangul.inventory import NO_CODA, NO_RESPONSE, NULL_ONSET, Position
from audire.hangul.syllable import Syllable, decompose_syllable, is_hangul_syllable


class PositionErrorType(StrEnum):
    """Error type at a single syllable position."""

    CORRECT = "correct"
    SUBSTITUTION = "substitution"
    #: A segment present in the target was not reported (coda X -> none, onset X -> ㅇ).
    OMISSION = "omission"
    #: A segment absent from the target was reported (none -> coda X, ㅇ -> onset X).
    ADDITION = "addition"
    #: No usable response for this position.
    NO_RESPONSE = "no_response"


class TrialErrorType(StrEnum):
    """Whole-syllable outcome of one calibration trial."""

    CORRECT = "correct"
    ONSET_ERROR = "onset_error"
    NUCLEUS_ERROR = "nucleus_error"
    CODA_ERROR = "coda_error"
    #: Two or more positions wrong. Reported separately because compound errors rise
    #: sharply with hearing-loss severity in the reference literature.
    COMPOUND = "compound"
    NO_RESPONSE = "no_response"


class ResponseQuality(StrEnum):
    """How usable the raw response string was."""

    #: Exactly one Hangul syllable.
    OK = "ok"
    #: Blank, "?", or otherwise explicitly declined.
    BLANK = "blank"
    #: Contained no Hangul syllable (e.g. Latin text, digits).
    NON_HANGUL = "non_hangul"
    #: More than one Hangul syllable; the first is scored and this flag is retained.
    MULTI_SYLLABLE = "multi_syllable"


#: Tokens a listener may enter to explicitly decline / report "I did not hear it".
DECLINE_TOKENS: frozenset[str] = frozenset(
    {"", "?", "??", "-", "x", "X", "모름", "못들음", "무응답"}
)


@dataclass(frozen=True, slots=True)
class PositionObservation:
    """One target -> perceived observation at one syllable position."""

    position: Position
    target: str
    perceived: str
    error_type: PositionErrorType

    @property
    def is_correct(self) -> bool:
        return self.error_type is PositionErrorType.CORRECT


@dataclass(frozen=True, slots=True)
class ParsedTrial:
    """The fully scored result of one calibration trial.

    Attributes
    ----------
    target:
        The presented Hangul syllable.
    raw_response:
        The listener's response exactly as entered, retained for auditability.
    response_syllable:
        The scored syllable, or ``None`` when the response was unusable.
    quality:
        Why ``response_syllable`` is or is not present.
    observations:
        Exactly three :class:`PositionObservation` values, one per position, always in
        onset / nucleus / coda order. Never truncated -- an unusable response produces
        three ``NO_RESPONSE`` observations rather than zero observations.
    trial_error:
        Whole-syllable label.
    """

    target: str
    raw_response: str
    response_syllable: Syllable | None
    quality: ResponseQuality
    observations: tuple[PositionObservation, PositionObservation, PositionObservation]
    trial_error: TrialErrorType

    @property
    def is_correct(self) -> bool:
        """Whether the listener reported the syllable exactly."""
        return self.trial_error is TrialErrorType.CORRECT

    @property
    def target_syllable(self) -> Syllable:
        return decompose_syllable(self.target)


def _classify_position(position: Position, target: str, perceived: str) -> PositionErrorType:
    """Classify one position's ``target -> perceived`` transition."""
    if perceived == NO_RESPONSE:
        return PositionErrorType.NO_RESPONSE
    if target == perceived:
        return PositionErrorType.CORRECT
    if position is Position.CODA:
        if target != NO_CODA and perceived == NO_CODA:
            return PositionErrorType.OMISSION
        if target == NO_CODA and perceived != NO_CODA:
            return PositionErrorType.ADDITION
        return PositionErrorType.SUBSTITUTION
    if position is Position.ONSET:
        # ㅇ is the phonologically null onset: ㄱ -> ㅇ drops a consonant, ㅇ -> ㄱ adds one.
        if target != NULL_ONSET and perceived == NULL_ONSET:
            return PositionErrorType.OMISSION
        if target == NULL_ONSET and perceived != NULL_ONSET:
            return PositionErrorType.ADDITION
        return PositionErrorType.SUBSTITUTION
    return PositionErrorType.SUBSTITUTION


def _first_hangul_syllable(text: str) -> tuple[str | None, int]:
    """Return the first Hangul syllable in ``text`` and the total Hangul syllable count."""
    found = [ch for ch in text if is_hangul_syllable(ch)]
    return (found[0] if found else None), len(found)


def parse_response(target: str, response: str) -> ParsedTrial:
    """Score one calibration trial.

    Parameters
    ----------
    target:
        The presented syllable. Must be a single precomposed modern Hangul syllable.
    response:
        The listener's raw answer. Leading/trailing whitespace is stripped; the rest is
        preserved verbatim in :attr:`ParsedTrial.raw_response`.

    Returns
    -------
    ParsedTrial
        Always with exactly three position observations, so that no trial silently
        contributes zero evidence.

    Raises
    ------
    ValueError
        If ``target`` is not a single modern Hangul syllable. Stimuli are curated, so a
        malformed target is a programming error rather than a listener behaviour.
    """
    if not is_hangul_syllable(target):
        raise ValueError(f"calibration target must be one Hangul syllable, got {target!r}")

    tgt = decompose_syllable(target)
    stripped = response.strip()

    if stripped in DECLINE_TOKENS:
        quality = ResponseQuality.BLANK
        resp: Syllable | None = None
    else:
        first, n_syl = _first_hangul_syllable(stripped)
        if first is None:
            quality = ResponseQuality.NON_HANGUL
            resp = None
        else:
            quality = ResponseQuality.MULTI_SYLLABLE if n_syl > 1 else ResponseQuality.OK
            resp = decompose_syllable(first)

    observations = tuple(
        PositionObservation(
            position=pos,
            target=tgt.get(pos),
            perceived=NO_RESPONSE if resp is None else resp.get(pos),
            error_type=(
                PositionErrorType.NO_RESPONSE
                if resp is None
                else _classify_position(pos, tgt.get(pos), resp.get(pos))
            ),
        )
        for pos in (Position.ONSET, Position.NUCLEUS, Position.CODA)
    )
    assert len(observations) == 3

    return ParsedTrial(
        target=target,
        raw_response=response,
        response_syllable=resp,
        quality=quality,
        observations=observations,
        trial_error=_classify_trial(observations),
    )


def _classify_trial(
    observations: tuple[PositionObservation, PositionObservation, PositionObservation],
) -> TrialErrorType:
    if all(o.error_type is PositionErrorType.NO_RESPONSE for o in observations):
        return TrialErrorType.NO_RESPONSE
    wrong = [o for o in observations if not o.is_correct]
    if not wrong:
        return TrialErrorType.CORRECT
    if len(wrong) > 1:
        return TrialErrorType.COMPOUND
    match wrong[0].position:
        case Position.ONSET:
            return TrialErrorType.ONSET_ERROR
        case Position.NUCLEUS:
            return TrialErrorType.NUCLEUS_ERROR
        case Position.CODA:
            return TrialErrorType.CODA_ERROR
