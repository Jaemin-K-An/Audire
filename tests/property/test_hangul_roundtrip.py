"""E1 — exhaustive and property-based Hangul decomposition/recomposition tests.

The exhaustive test covers every one of the 11,172 precomposed modern Hangul syllables,
so the round-trip property is *proved by enumeration* for the whole block, not merely
sampled. The Hypothesis tests additionally cover mixed-script text and the composition
inverse direction.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from audire.hangul import (
    CODA_JAMO,
    HANGUL_SYLLABLE_END,
    HANGUL_SYLLABLE_START,
    NO_CODA,
    NUCLEUS_JAMO,
    ONSET_JAMO,
    Position,
    Syllable,
    compose_syllable,
    decompose_syllable,
    decompose_text,
    is_hangul_syllable,
    recompose_text,
)
from audire.hangul.syllable import jamo_sequence

ALL_SYLLABLES = [chr(c) for c in range(HANGUL_SYLLABLE_START, HANGUL_SYLLABLE_END + 1)]

hangul_syllables = st.sampled_from(ALL_SYLLABLES)
mixed_text = st.text(
    alphabet=st.one_of(
        hangul_syllables,
        st.sampled_from(list(" .,!?0123456789abcXYZ가나다\n\t")),
        st.characters(min_codepoint=0x20, max_codepoint=0x2FF),
    ),
    max_size=60,
)


def test_block_size_is_exactly_11172() -> None:
    """19 onsets x 21 nuclei x 28 coda slots == 11172 precomposed syllables."""
    assert len(ALL_SYLLABLES) == 19 * 21 * 28 == 11172


def test_inventory_sizes() -> None:
    assert len(ONSET_JAMO) == 19
    assert len(NUCLEUS_JAMO) == 21
    assert len(CODA_JAMO) == 27
    assert len(set(ONSET_JAMO)) == 19
    assert len(set(NUCLEUS_JAMO)) == 21
    assert len(set(CODA_JAMO)) == 27


def test_exhaustive_roundtrip_every_modern_syllable() -> None:
    """recompose(decompose(x)) == x for ALL 11,172 modern Hangul syllables."""
    for ch in ALL_SYLLABLES:
        assert decompose_syllable(ch).compose() == ch, ch


def test_exhaustive_decomposition_is_injective() -> None:
    """Distinct syllables must never decompose to the same (onset, nucleus, coda)."""
    seen: dict[tuple[str, str, str], str] = {}
    for ch in ALL_SYLLABLES:
        s = decompose_syllable(ch)
        key = (s.onset, s.nucleus, s.coda)
        assert key not in seen, f"{ch} collides with {seen.get(key)}"
        seen[key] = ch
    assert len(seen) == 11172


def test_exhaustive_composition_covers_the_whole_block() -> None:
    """The forward direction: every (onset, nucleus, coda) triple yields a distinct syllable."""
    produced = {
        compose_syllable(o, n, c)
        for o in ONSET_JAMO
        for n in NUCLEUS_JAMO
        for c in (NO_CODA, *CODA_JAMO)
    }
    assert produced == set(ALL_SYLLABLES)


def test_exhaustive_coda_flag_consistency() -> None:
    n_with_coda = sum(decompose_syllable(ch).has_coda for ch in ALL_SYLLABLES)
    # 27 of the 28 coda slots carry a consonant.
    assert n_with_coda == 19 * 21 * 27


@given(hangul_syllables)
def test_property_roundtrip(ch: str) -> None:
    assert decompose_syllable(ch).compose() == ch


@given(mixed_text)
def test_property_text_roundtrip_preserves_non_hangul(text: str) -> None:
    """recompose_text(decompose_text(x)) == x for arbitrary mixed-script text."""
    assert recompose_text(decompose_text(text)) == text


@given(
    st.sampled_from(ONSET_JAMO),
    st.sampled_from(NUCLEUS_JAMO),
    st.sampled_from((NO_CODA, *CODA_JAMO)),
)
def test_property_compose_then_decompose(onset: str, nucleus: str, coda: str) -> None:
    syl = decompose_syllable(compose_syllable(onset, nucleus, coda))
    assert (syl.onset, syl.nucleus, syl.coda) == (onset, nucleus, coda)


# --------------------------------------------------------------------------- known values


@pytest.mark.parametrize(
    ("ch", "expected"),
    [
        ("가", ("ㄱ", "ㅏ", NO_CODA)),
        ("각", ("ㄱ", "ㅏ", "ㄱ")),
        ("힣", ("ㅎ", "ㅣ", "ㅎ")),
        ("아", ("ㅇ", "ㅏ", NO_CODA)),
        ("앙", ("ㅇ", "ㅏ", "ㅇ")),
        ("값", ("ㄱ", "ㅏ", "ㅄ")),
        ("돐", ("ㄷ", "ㅗ", "ㄽ")),
        ("왜", ("ㅇ", "ㅙ", NO_CODA)),
        ("의", ("ㅇ", "ㅢ", NO_CODA)),
        ("쌍", ("ㅆ", "ㅏ", "ㅇ")),
    ],
)
def test_known_decompositions(ch: str, expected: tuple[str, str, str]) -> None:
    s = decompose_syllable(ch)
    assert (s.onset, s.nucleus, s.coda) == expected


@pytest.mark.parametrize(
    ("ch", "structure"),
    [("가", "CV"), ("각", "CVC"), ("아", "V"), ("악", "VC"), ("의", "V"), ("옹", "VC")],
)
def test_structure_labels_match_dataset_convention(ch: str, structure: str) -> None:
    """The primary dataset labels items V / VC / CV / CVC with ㅇ as 'no initial consonant'."""
    assert decompose_syllable(ch).structure == structure


# --------------------------------------------------------------------------- failure paths


@pytest.mark.parametrize("bad", ["", "가나", "A", "ㄱ", "ㅏ", " ", "1", "漢", "ᄀ"])
def test_decompose_rejects_non_syllables(bad: str) -> None:
    assert not is_hangul_syllable(bad)
    with pytest.raises(ValueError, match="not a precomposed modern Hangul syllable"):
        decompose_syllable(bad)


@pytest.mark.parametrize(
    ("onset", "nucleus", "coda"),
    [
        ("ㅏ", "ㅏ", NO_CODA),  # vowel in onset slot
        ("ㄱ", "ㄱ", NO_CODA),  # consonant in nucleus slot
        ("ㄱ", "ㅏ", ""),  # empty string is NOT accepted; NO_CODA must be explicit
        ("ㄱ", "ㅏ", "ㄸ"),  # ㄸ is a valid onset but never a coda
        ("ㄱ", "ㅏ", "ㅃ"),
        ("ㄳ", "ㅏ", NO_CODA),  # cluster is coda-only
    ],
)
def test_compose_rejects_out_of_inventory_jamo(onset: str, nucleus: str, coda: str) -> None:
    with pytest.raises(ValueError, match="not in its position inventory"):
        compose_syllable(onset, nucleus, coda)


def test_syllable_dataclass_validates_on_construction() -> None:
    with pytest.raises(ValueError, match="invalid onset"):
        Syllable(onset="ㅏ", nucleus="ㅏ", coda=NO_CODA)
    with pytest.raises(ValueError, match="invalid nucleus"):
        Syllable(onset="ㄱ", nucleus="ㄱ", coda=NO_CODA)
    with pytest.raises(ValueError, match="invalid coda"):
        Syllable(onset="ㄱ", nucleus="ㅏ", coda="ㄸ")


def test_no_coda_is_not_the_empty_string() -> None:
    """Regression guard: 'no coda' must stay a distinguishable explicit category."""
    assert NO_CODA != ""
    assert NO_CODA not in CODA_JAMO
    assert decompose_syllable("가").coda == NO_CODA


# --------------------------------------------------------------------------- jamo sequence


def test_jamo_sequence_includes_no_coda_slots() -> None:
    """Absence of a coda is observable (a listener can add one), so it must be emitted."""
    seq = jamo_sequence("가족")
    assert seq == [
        (Position.ONSET, "ㄱ"),
        (Position.NUCLEUS, "ㅏ"),
        (Position.CODA, NO_CODA),
        (Position.ONSET, "ㅈ"),
        (Position.NUCLEUS, "ㅗ"),
        (Position.CODA, "ㄱ"),
    ]


def test_jamo_sequence_skips_non_hangul() -> None:
    assert jamo_sequence("a 가 1") == [
        (Position.ONSET, "ㄱ"),
        (Position.NUCLEUS, "ㅏ"),
        (Position.CODA, NO_CODA),
    ]


def test_get_by_position() -> None:
    s = decompose_syllable("각")
    assert s.get(Position.ONSET) == "ㄱ"
    assert s.get(Position.NUCLEUS) == "ㅏ"
    assert s.get(Position.CODA) == "ㄱ"
    assert s.as_dict() == {"onset": "ㄱ", "nucleus": "ㅏ", "coda": "ㄱ"}
    assert str(s) == "각"
