"""Decomposition and recomposition of modern precomposed Hangul syllables.

The Unicode Hangul Syllables block (U+AC00..U+D7A3) is algorithmically composed:

    code = 0xAC00 + (onset_index * 21 + nucleus_index) * 28 + coda_index

so decomposition is exact and total for every code point in the block. This module
exposes that algorithm with AUDIRE's explicit :data:`~audire.hangul.inventory.NO_CODA`
category instead of an empty string, and passes non-Hangul characters through
unchanged so that mixed Korean/Latin/numeric caption text survives a round trip.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

from audire.hangul.inventory import (
    CODA_JAMO,
    NO_CODA,
    NUCLEUS_JAMO,
    ONSET_JAMO,
    Position,
)

#: First code point of the Unicode Hangul Syllables block (가).
HANGUL_SYLLABLE_START: Final[int] = 0xAC00
#: Last code point of the Unicode Hangul Syllables block (힣).
HANGUL_SYLLABLE_END: Final[int] = 0xD7A3

_N_NUCLEUS: Final[int] = len(NUCLEUS_JAMO)  # 21
_N_CODA_SLOTS: Final[int] = len(CODA_JAMO) + 1  # 28, slot 0 == no coda

_ONSET_INDEX: Final[dict[str, int]] = {j: i for i, j in enumerate(ONSET_JAMO)}
_NUCLEUS_INDEX: Final[dict[str, int]] = {j: i for i, j in enumerate(NUCLEUS_JAMO)}
#: Coda symbol -> Unicode coda slot. ``NO_CODA`` maps to slot 0.
_CODA_INDEX: Final[dict[str, int]] = {NO_CODA: 0} | {j: i + 1 for i, j in enumerate(CODA_JAMO)}


@dataclass(frozen=True, slots=True)
class Syllable:
    """One decomposed modern Hangul syllable.

    Attributes
    ----------
    onset:
        One of :data:`~audire.hangul.inventory.ONSET_JAMO`. ``ㅇ`` is the null onset.
    nucleus:
        One of :data:`~audire.hangul.inventory.NUCLEUS_JAMO`.
    coda:
        One of :data:`~audire.hangul.inventory.CODA_JAMO`, or
        :data:`~audire.hangul.inventory.NO_CODA` when the syllable has no coda.
        Never ``None`` and never ``""`` -- "no coda" is an explicit category.
    """

    onset: str
    nucleus: str
    coda: str

    def __post_init__(self) -> None:
        if self.onset not in _ONSET_INDEX:
            raise ValueError(f"invalid onset jamo: {self.onset!r}")
        if self.nucleus not in _NUCLEUS_INDEX:
            raise ValueError(f"invalid nucleus jamo: {self.nucleus!r}")
        if self.coda not in _CODA_INDEX:
            raise ValueError(f"invalid coda jamo: {self.coda!r}")

    @property
    def has_coda(self) -> bool:
        """Whether this syllable carries a coda consonant."""
        return self.coda != NO_CODA

    @property
    def structure(self) -> str:
        """CV / CVC / V / VC structure label, matching the primary dataset's convention.

        The null onset ``ㅇ`` is treated as absence of an initial consonant, which is how
        the Korean Monosyllabic Speech Perception Test Dataset labels its ``V`` and ``VC``
        items.
        """
        head = "C" if self.onset != "ㅇ" else ""
        tail = "C" if self.has_coda else ""
        return f"{head}V{tail}"

    def as_dict(self) -> dict[str, str]:
        """Return a position-keyed mapping, the form consumed by the confusion engine."""
        return {
            Position.ONSET.value: self.onset,
            Position.NUCLEUS.value: self.nucleus,
            Position.CODA.value: self.coda,
        }

    def get(self, position: Position) -> str:
        """Return the jamo occupying ``position``."""
        match position:
            case Position.ONSET:
                return self.onset
            case Position.NUCLEUS:
                return self.nucleus
            case Position.CODA:
                return self.coda

    def compose(self) -> str:
        """Recompose this syllable into its precomposed Hangul character."""
        return compose_syllable(self.onset, self.nucleus, self.coda)

    def __str__(self) -> str:
        return self.compose()


def is_hangul_syllable(ch: str) -> bool:
    """Return whether ``ch`` is a single precomposed modern Hangul syllable."""
    return len(ch) == 1 and HANGUL_SYLLABLE_START <= ord(ch) <= HANGUL_SYLLABLE_END


def decompose_syllable(ch: str) -> Syllable:
    """Decompose one precomposed Hangul syllable into onset / nucleus / coda.

    Raises
    ------
    ValueError
        If ``ch`` is not exactly one character in U+AC00..U+D7A3. Callers that need to
        tolerate non-Hangul text should use :func:`decompose_text` instead.
    """
    if not is_hangul_syllable(ch):
        raise ValueError(f"not a precomposed modern Hangul syllable: {ch!r}")
    offset = ord(ch) - HANGUL_SYLLABLE_START
    coda_slot = offset % _N_CODA_SLOTS
    nucleus_idx = (offset // _N_CODA_SLOTS) % _N_NUCLEUS
    onset_idx = offset // (_N_CODA_SLOTS * _N_NUCLEUS)
    return Syllable(
        onset=ONSET_JAMO[onset_idx],
        nucleus=NUCLEUS_JAMO[nucleus_idx],
        coda=NO_CODA if coda_slot == 0 else CODA_JAMO[coda_slot - 1],
    )


def compose_syllable(onset: str, nucleus: str, coda: str = NO_CODA) -> str:
    """Compose onset / nucleus / coda jamo into a precomposed Hangul syllable.

    ``coda`` accepts :data:`~audire.hangul.inventory.NO_CODA`. Empty string and ``None``
    are rejected on purpose: "no coda" must be stated explicitly so that it cannot be
    confused with a missing observation.

    Raises
    ------
    ValueError
        If any argument is not a member of its position's inventory.
    """
    try:
        onset_idx = _ONSET_INDEX[onset]
        nucleus_idx = _NUCLEUS_INDEX[nucleus]
        coda_slot = _CODA_INDEX[coda]
    except KeyError as exc:
        raise ValueError(
            f"cannot compose syllable from ({onset!r}, {nucleus!r}, {coda!r}): "
            f"{exc.args[0]!r} is not in its position inventory"
        ) from exc
    code = (
        HANGUL_SYLLABLE_START + (onset_idx * _N_NUCLEUS + nucleus_idx) * _N_CODA_SLOTS + coda_slot
    )
    return chr(code)


def syllable_structure(ch: str) -> str:
    """Return the CV/CVC/V/VC structure label of one Hangul syllable."""
    return decompose_syllable(ch).structure


def iter_syllables(text: str) -> Iterator[tuple[int, str, Syllable | None]]:
    """Iterate over ``text`` yielding ``(index, character, decomposition_or_None)``.

    Non-Hangul characters yield ``None`` for the decomposition rather than raising, so
    that mixed-script caption text can be processed in one pass.
    """
    for i, ch in enumerate(text):
        yield i, ch, decompose_syllable(ch) if is_hangul_syllable(ch) else None


def decompose_text(text: str) -> list[Syllable | str]:
    """Decompose every Hangul syllable in ``text``, passing other characters through.

    The result is a mixed list of :class:`Syllable` objects and single-character strings.
    :func:`recompose_text` is its exact inverse for all inputs.
    """
    return [syl if syl is not None else ch for _, ch, syl in iter_syllables(text)]


def recompose_text(parts: list[Syllable | str]) -> str:
    """Inverse of :func:`decompose_text`."""
    return "".join(p.compose() if isinstance(p, Syllable) else p for p in parts)


def jamo_sequence(text: str) -> list[tuple[Position, str]]:
    """Flatten ``text`` into an ordered ``(position, jamo)`` sequence.

    Non-Hangul characters contribute nothing. ``NO_CODA`` slots are *included*, because
    the absence of a coda is an observable perceptual category (a listener can
    incorrectly add one).
    """
    out: list[tuple[Position, str]] = []
    for _, _, syl in iter_syllables(text):
        if syl is None:
            continue
        out.append((Position.ONSET, syl.onset))
        out.append((Position.NUCLEUS, syl.nucleus))
        out.append((Position.CODA, syl.coda))
    return out
