"""Modern-Hangul syllable decomposition, recomposition and phoneme inventories.

This package is the foundation of every downstream AUDIRE component: the confusion
engine, the risk model and the caption engine all operate on the onset / nucleus /
coda decomposition produced here.
"""

from audire.hangul.inventory import (
    CODA_JAMO,
    NO_CODA,
    NO_RESPONSE,
    NUCLEUS_JAMO,
    ONSET_JAMO,
    Position,
    categories_for,
    is_valid_category,
)
from audire.hangul.syllable import (
    HANGUL_SYLLABLE_END,
    HANGUL_SYLLABLE_START,
    Syllable,
    compose_syllable,
    decompose_syllable,
    decompose_text,
    is_hangul_syllable,
    iter_syllables,
    recompose_text,
    syllable_structure,
)

__all__ = [
    "CODA_JAMO",
    "HANGUL_SYLLABLE_END",
    "HANGUL_SYLLABLE_START",
    "NO_CODA",
    "NO_RESPONSE",
    "NUCLEUS_JAMO",
    "ONSET_JAMO",
    "Position",
    "Syllable",
    "categories_for",
    "compose_syllable",
    "decompose_syllable",
    "decompose_text",
    "is_hangul_syllable",
    "is_valid_category",
    "iter_syllables",
    "recompose_text",
    "syllable_structure",
]
