"""Korean phonological feature classes used for features, back-off and explanations.

Everything in this module is standard descriptive Korean phonology (three-way laryngeal
contrast in obstruents; place and manner classes; coda neutralisation to seven
consonants -- 음절의 끝소리 규칙). It encodes no numeric claim and no result from any
particular study; it is a linguistic index used to build features, to back a sparse
confusion row off onto a broader class, and to phrase explanations.

Consistency check against the literature
----------------------------------------
Ma et al. (2026, DOI 10.21848/asr.250216) report analysing 18 onsets, 16 nuclei and 8
codas. AUDIRE's onset inventory minus the null onset ``ㅇ`` is 18, and the neutralised
coda inventory plus "no coda" is 8, which is consistent with that report. AUDIRE's
nucleus inventory is the full orthographic 21; the reason that paper analyses 16 is not
stated in the material available to this project and is recorded as an open question in
docs/RESULTS.md rather than guessed at.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from audire.hangul.inventory import CODA_JAMO, NO_CODA, NO_RESPONSE, NUCLEUS_JAMO, ONSET_JAMO


class Phonation(StrEnum):
    """Korean three-way laryngeal contrast, plus sonorant categories."""

    LAX = "lax"  # 평음
    TENSE = "tense"  # 경음
    ASPIRATED = "aspirated"  # 격음
    NASAL = "nasal"
    LIQUID = "liquid"
    NONE = "none"  # null onset ㅇ


class Place(StrEnum):
    BILABIAL = "bilabial"
    ALVEOLAR = "alveolar"
    ALVEOLO_PALATAL = "alveolo_palatal"
    VELAR = "velar"
    GLOTTAL = "glottal"
    NONE = "none"


class Manner(StrEnum):
    STOP = "stop"
    AFFRICATE = "affricate"
    FRICATIVE = "fricative"
    NASAL = "nasal"
    LIQUID = "liquid"
    NONE = "none"


class VowelShape(StrEnum):
    MONOPHTHONG = "monophthong"
    Y_GLIDE = "y_glide"  # 이중모음 with /j/ onglide
    W_GLIDE = "w_glide"  # 이중모음 with /w/ onglide
    UI = "ui"  # ㅢ, conventionally treated separately


ONSET_PHONATION: Final[dict[str, Phonation]] = {
    "ㄱ": Phonation.LAX,
    "ㄷ": Phonation.LAX,
    "ㅂ": Phonation.LAX,
    "ㅅ": Phonation.LAX,
    "ㅈ": Phonation.LAX,
    "ㅎ": Phonation.LAX,
    "ㄲ": Phonation.TENSE,
    "ㄸ": Phonation.TENSE,
    "ㅃ": Phonation.TENSE,
    "ㅆ": Phonation.TENSE,
    "ㅉ": Phonation.TENSE,
    "ㅋ": Phonation.ASPIRATED,
    "ㅌ": Phonation.ASPIRATED,
    "ㅍ": Phonation.ASPIRATED,
    "ㅊ": Phonation.ASPIRATED,
    "ㄴ": Phonation.NASAL,
    "ㅁ": Phonation.NASAL,
    "ㄹ": Phonation.LIQUID,
    "ㅇ": Phonation.NONE,
}

ONSET_PLACE: Final[dict[str, Place]] = {
    "ㅂ": Place.BILABIAL,
    "ㅃ": Place.BILABIAL,
    "ㅍ": Place.BILABIAL,
    "ㅁ": Place.BILABIAL,
    "ㄷ": Place.ALVEOLAR,
    "ㄸ": Place.ALVEOLAR,
    "ㅌ": Place.ALVEOLAR,
    "ㄴ": Place.ALVEOLAR,
    "ㅅ": Place.ALVEOLAR,
    "ㅆ": Place.ALVEOLAR,
    "ㄹ": Place.ALVEOLAR,
    "ㅈ": Place.ALVEOLO_PALATAL,
    "ㅉ": Place.ALVEOLO_PALATAL,
    "ㅊ": Place.ALVEOLO_PALATAL,
    "ㄱ": Place.VELAR,
    "ㄲ": Place.VELAR,
    "ㅋ": Place.VELAR,
    "ㅎ": Place.GLOTTAL,
    "ㅇ": Place.NONE,
}

ONSET_MANNER: Final[dict[str, Manner]] = {
    "ㄱ": Manner.STOP,
    "ㄲ": Manner.STOP,
    "ㅋ": Manner.STOP,
    "ㄷ": Manner.STOP,
    "ㄸ": Manner.STOP,
    "ㅌ": Manner.STOP,
    "ㅂ": Manner.STOP,
    "ㅃ": Manner.STOP,
    "ㅍ": Manner.STOP,
    "ㅈ": Manner.AFFRICATE,
    "ㅉ": Manner.AFFRICATE,
    "ㅊ": Manner.AFFRICATE,
    "ㅅ": Manner.FRICATIVE,
    "ㅆ": Manner.FRICATIVE,
    "ㅎ": Manner.FRICATIVE,
    "ㄴ": Manner.NASAL,
    "ㅁ": Manner.NASAL,
    "ㄹ": Manner.LIQUID,
    "ㅇ": Manner.NONE,
}

NUCLEUS_SHAPE: Final[dict[str, VowelShape]] = {
    "ㅏ": VowelShape.MONOPHTHONG,
    "ㅐ": VowelShape.MONOPHTHONG,
    "ㅓ": VowelShape.MONOPHTHONG,
    "ㅔ": VowelShape.MONOPHTHONG,
    "ㅗ": VowelShape.MONOPHTHONG,
    "ㅚ": VowelShape.MONOPHTHONG,
    "ㅜ": VowelShape.MONOPHTHONG,
    "ㅟ": VowelShape.MONOPHTHONG,
    "ㅡ": VowelShape.MONOPHTHONG,
    "ㅣ": VowelShape.MONOPHTHONG,
    "ㅑ": VowelShape.Y_GLIDE,
    "ㅒ": VowelShape.Y_GLIDE,
    "ㅕ": VowelShape.Y_GLIDE,
    "ㅖ": VowelShape.Y_GLIDE,
    "ㅛ": VowelShape.Y_GLIDE,
    "ㅠ": VowelShape.Y_GLIDE,
    "ㅘ": VowelShape.W_GLIDE,
    "ㅙ": VowelShape.W_GLIDE,
    "ㅝ": VowelShape.W_GLIDE,
    "ㅞ": VowelShape.W_GLIDE,
    "ㅢ": VowelShape.UI,
}

#: Rounded vowels (lip rounding present in the nucleus or its onglide).
ROUNDED_NUCLEI: Final[frozenset[str]] = frozenset(
    {"ㅗ", "ㅜ", "ㅛ", "ㅠ", "ㅚ", "ㅟ", "ㅘ", "ㅙ", "ㅝ", "ㅞ"}
)

#: Korean coda neutralisation (음절의 끝소리 규칙): all 27 orthographic codas surface as
#: one of seven consonants. Clusters resolve to the consonant that is actually released.
CODA_NEUTRALISATION: Final[dict[str, str]] = {
    "ㄱ": "ㄱ",
    "ㄲ": "ㄱ",
    "ㅋ": "ㄱ",
    "ㄳ": "ㄱ",
    "ㄺ": "ㄱ",
    "ㄴ": "ㄴ",
    "ㄵ": "ㄴ",
    "ㄶ": "ㄴ",
    "ㄷ": "ㄷ",
    "ㅅ": "ㄷ",
    "ㅆ": "ㄷ",
    "ㅈ": "ㄷ",
    "ㅊ": "ㄷ",
    "ㅌ": "ㄷ",
    "ㅎ": "ㄷ",
    "ㄹ": "ㄹ",
    "ㄼ": "ㄹ",
    "ㄽ": "ㄹ",
    "ㄾ": "ㄹ",
    "ㅀ": "ㄹ",
    "ㅁ": "ㅁ",
    "ㄻ": "ㅁ",
    "ㅂ": "ㅂ",
    "ㅍ": "ㅂ",
    "ㅄ": "ㅂ",
    "ㄿ": "ㅂ",
    "ㅇ": "ㅇ",
}

#: The seven surface codas plus the explicit "no coda" category: eight categories.
NEUTRALISED_CODA_CATEGORIES: Final[tuple[str, ...]] = (
    NO_CODA,
    "ㄱ",
    "ㄴ",
    "ㄷ",
    "ㄹ",
    "ㅁ",
    "ㅂ",
    "ㅇ",
)


def neutralise_coda(coda: str) -> str:
    """Map an orthographic coda to its surface (neutralised) form.

    :data:`~audire.hangul.inventory.NO_CODA` and
    :data:`~audire.hangul.inventory.NO_RESPONSE` pass through unchanged.
    """
    if coda in (NO_CODA, NO_RESPONSE):
        return coda
    try:
        return CODA_NEUTRALISATION[coda]
    except KeyError as exc:
        raise KeyError(f"{coda!r} is not a coda jamo") from exc


def onset_features(onset: str) -> dict[str, str]:
    """Phonological feature bundle for an onset jamo."""
    return {
        "phonation": ONSET_PHONATION[onset].value,
        "place": ONSET_PLACE[onset].value,
        "manner": ONSET_MANNER[onset].value,
    }


def nucleus_features(nucleus: str) -> dict[str, str | bool]:
    """Phonological feature bundle for a nucleus jamo."""
    return {
        "shape": NUCLEUS_SHAPE[nucleus].value,
        "rounded": nucleus in ROUNDED_NUCLEI,
    }


def coda_features(coda: str) -> dict[str, str | bool]:
    """Phonological feature bundle for a coda category."""
    if coda == NO_CODA:
        return {"surface": NO_CODA, "is_cluster": False, "present": False}
    return {
        "surface": neutralise_coda(coda),
        "is_cluster": len(_CLUSTER_MEMBERS.get(coda, "")) > 0,
        "present": True,
    }


#: Orthographic coda clusters (겹받침) and their constituent consonants.
_CLUSTER_MEMBERS: Final[dict[str, str]] = {
    "ㄳ": "ㄱㅅ",
    "ㄵ": "ㄴㅈ",
    "ㄶ": "ㄴㅎ",
    "ㄺ": "ㄹㄱ",
    "ㄻ": "ㄹㅁ",
    "ㄼ": "ㄹㅂ",
    "ㄽ": "ㄹㅅ",
    "ㄾ": "ㄹㅌ",
    "ㄿ": "ㄹㅍ",
    "ㅀ": "ㄹㅎ",
    "ㅄ": "ㅂㅅ",
}


def _self_check() -> None:
    """Fail fast at import time if a feature table drifts out of sync with the inventory."""
    missing_onset = set(ONSET_JAMO) - set(ONSET_PHONATION)
    missing_nucleus = set(NUCLEUS_JAMO) - set(NUCLEUS_SHAPE)
    missing_coda = set(CODA_JAMO) - set(CODA_NEUTRALISATION)
    if missing_onset or missing_nucleus or missing_coda:
        raise RuntimeError(
            "phonological feature tables are incomplete: "
            f"onset={sorted(missing_onset)} nucleus={sorted(missing_nucleus)} "
            f"coda={sorted(missing_coda)}"
        )


_self_check()
