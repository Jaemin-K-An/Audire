"""Phoneme (jamo) inventories and the response-category alphabets used by AUDIRE.

Design notes
------------
AUDIRE builds one confusion matrix per *syllable position*. A confusion matrix needs a
closed alphabet of categories for both the target axis and the perceived axis. This
module defines those alphabets and the two special categories that make the alphabets
closed without silently discarding data:

``NO_CODA`` (``"-"``)
    A genuine linguistic category, not a missing value. Korean syllables legitimately
    have no coda. Encoding it as a real category means the coda matrix represents
    *omission* (``target coda X`` -> ``NO_CODA``) and *addition* (``NO_CODA`` ->
    ``perceived coda X``) as ordinary cells rather than as dropped observations.

``NO_RESPONSE`` (``"?"``)
    The listener produced nothing usable for that position (no answer, unintelligible
    answer, or a non-Hangul answer). It is a perceived-axis-only category: a *target*
    is never ``NO_RESPONSE``.

Null onset
----------
``ㅇ`` in onset position is the phonologically null onset. It is kept as an ordinary
onset category so that ``ㅇ -> ㄱ`` (an addition-type error) and ``ㄱ -> ㅇ`` (an
omission-type error) are representable as ordinary substitutions in a square matrix.
``audire.confusion.errors`` classifies those cells into error *types*; the matrix
itself stays a plain square stochastic matrix.

Relation to the literature
--------------------------
Ma et al. (2026), DOI 10.21848/asr.250216, analyse 18 onsets, 16 nuclei and 8 codas.
AUDIRE deliberately uses the *complete orthographic* inventory instead (19 / 21 / 27 +
``NO_CODA``) so that any modern Hangul syllable can be scored without a lossy mapping.
The reduced literature inventories are reachable through ``audire.confusion.grouping``
when a comparison to published tables is required. See docs/DECISIONS.md ADR-0004.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

#: 19 onset (초성) jamo in Unicode composition order. Index 11 (``ㅇ``) is the null onset.
ONSET_JAMO: Final[tuple[str, ...]] = (
    "ㄱ",
    "ㄲ",
    "ㄴ",
    "ㄷ",
    "ㄸ",
    "ㄹ",
    "ㅁ",
    "ㅂ",
    "ㅃ",
    "ㅅ",
    "ㅆ",
    "ㅇ",
    "ㅈ",
    "ㅉ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
)

#: 21 nucleus (중성) jamo in Unicode composition order.
NUCLEUS_JAMO: Final[tuple[str, ...]] = (
    "ㅏ",
    "ㅐ",
    "ㅑ",
    "ㅒ",
    "ㅓ",
    "ㅔ",
    "ㅕ",
    "ㅖ",
    "ㅗ",
    "ㅘ",
    "ㅙ",
    "ㅚ",
    "ㅛ",
    "ㅜ",
    "ㅝ",
    "ㅞ",
    "ㅟ",
    "ㅠ",
    "ㅡ",
    "ㅢ",
    "ㅣ",
)

#: 27 coda (종성) jamo in Unicode composition order. Unicode index 0 is "no coda" and is
#: represented by :data:`NO_CODA` rather than by an empty string.
CODA_JAMO: Final[tuple[str, ...]] = (
    "ㄱ",
    "ㄲ",
    "ㄳ",
    "ㄴ",
    "ㄵ",
    "ㄶ",
    "ㄷ",
    "ㄹ",
    "ㄺ",
    "ㄻ",
    "ㄼ",
    "ㄽ",
    "ㄾ",
    "ㄿ",
    "ㅀ",
    "ㅁ",
    "ㅂ",
    "ㅄ",
    "ㅅ",
    "ㅆ",
    "ㅇ",
    "ㅈ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
)

#: Explicit "this syllable has no coda" category. A real category, never a missing value.
NO_CODA: Final[str] = "-"

#: Explicit "listener produced no usable response for this position" category.
#: Valid on the perceived axis only.
NO_RESPONSE: Final[str] = "?"

#: The null onset. Kept as an ordinary onset category; see the module docstring.
NULL_ONSET: Final[str] = "ㅇ"


class Position(StrEnum):
    """Syllable position. The unit of a confusion matrix."""

    ONSET = "onset"
    NUCLEUS = "nucleus"
    CODA = "coda"


_TARGET_CATEGORIES: Final[dict[Position, tuple[str, ...]]] = {
    Position.ONSET: ONSET_JAMO,
    Position.NUCLEUS: NUCLEUS_JAMO,
    Position.CODA: (NO_CODA, *CODA_JAMO),
}

_PERCEIVED_CATEGORIES: Final[dict[Position, tuple[str, ...]]] = {
    pos: (*cats, NO_RESPONSE) for pos, cats in _TARGET_CATEGORIES.items()
}


def categories_for(position: Position, *, axis: str = "target") -> tuple[str, ...]:
    """Return the ordered category alphabet for ``position``.

    Parameters
    ----------
    position:
        Syllable position.
    axis:
        ``"target"`` for the row alphabet (what was presented) or ``"perceived"`` for
        the column alphabet (what the listener reported). The perceived alphabet adds
        :data:`NO_RESPONSE`.

    Raises
    ------
    ValueError
        If ``axis`` is not ``"target"`` or ``"perceived"``.
    """
    if axis == "target":
        return _TARGET_CATEGORIES[position]
    if axis == "perceived":
        return _PERCEIVED_CATEGORIES[position]
    raise ValueError(f"axis must be 'target' or 'perceived', got {axis!r}")


def is_valid_category(position: Position, symbol: str, *, axis: str = "target") -> bool:
    """Return whether ``symbol`` is a legal category for ``position`` on ``axis``."""
    return symbol in categories_for(position, axis=axis)


def n_categories(position: Position, *, axis: str = "target") -> int:
    """Return the alphabet size for ``position`` on ``axis``."""
    return len(categories_for(position, axis=axis))
