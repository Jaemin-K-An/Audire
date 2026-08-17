"""§9 정리 — 유니코드 정규화와 설명 항목 개수.

둘 다 "조용히 틀리는" 부류의 결함입니다. 어느 쪽도 예외를 던지지 않고, 집계 지표로도
드러나지 않으며, 특정 입력 경로에서만 나타납니다.
"""

from __future__ import annotations

import unicodedata

import pytest

from audire.confusion import CalibrationTrial, ConfusionProfile
from audire.confusion.errors import ResponseQuality, parse_response, to_nfc
from audire.hangul.inventory import Position


def _nfd(text: str) -> str:
    return unicodedata.normalize("NFD", text)


# ------------------------------------------------------------------- 유니코드 정규화


def test_nfd_correct_answer_is_scored_correct():
    """회귀 테스트.

    정규화 이전에는 NFD 로 입력된 **정답**이 ``non_hangul`` 로 분류되고 오답으로
    채점됐습니다. macOS 는 파일명과 여러 복사/IME 경로에서 NFD 를 만들어내므로 가정이
    아니라 실제로 들어옵니다. 그런 청취자는 모든 시행이 사용 불가가 되어 혼동 프로파일이
    통째로 비고, 집계 검사로는 절대 드러나지 않습니다.
    """
    assert _nfd("각") != "각", "이 테스트의 전제: 두 표현은 문자열로 다릅니다"

    parsed = parse_response("각", _nfd("각"))
    assert parsed.quality is ResponseQuality.OK
    assert parsed.is_correct


def test_nfd_wrong_answer_is_scored_as_that_wrong_answer():
    """오답도 제대로 해석되어야 합니다 — 그냥 '사용 불가' 로 버려지면 증거를 잃습니다."""
    parsed = parse_response("각", _nfd("닥"))
    assert parsed.quality is ResponseQuality.OK
    assert not parsed.is_correct
    assert parsed.response_syllable is not None
    assert parsed.response_syllable.onset == "ㄷ"


def test_nfd_target_is_accepted():
    """자극 쪽이 NFD 여도 프로그래밍 오류로 거부되어서는 안 됩니다."""
    assert parse_response(_nfd("각"), "각").is_correct


def test_raw_response_keeps_the_listener_input_untouched():
    """정규화는 채점에만 적용되고, 감사용 원본은 입력 그대로 남아야 합니다."""
    raw = _nfd("각")
    parsed = parse_response("각", raw)
    assert parsed.raw_response == raw
    assert parsed.raw_response != parsed.target


def test_nfd_and_nfc_produce_identical_confusion_evidence():
    """같은 응답의 두 표현이 같은 행렬을 만들어야 합니다."""
    pairs = [("각", "각"), ("각", "닥"), ("간", "간")]
    nfc = ConfusionProfile.from_trials(
        "L",
        [
            CalibrationTrial(stimulus_id=f"s{i}", target=t, response=r)
            for i, (t, r) in enumerate(pairs)
        ],
        is_synthetic=True,
    )
    nfd = ConfusionProfile.from_trials(
        "L",
        [
            CalibrationTrial(stimulus_id=f"s{i}", target=_nfd(t), response=_nfd(r))
            for i, (t, r) in enumerate(pairs)
        ],
        is_synthetic=True,
    )
    for position in (Position.ONSET, Position.NUCLEUS, Position.CODA):
        assert (nfc.matrix(position).counts == nfd.matrix(position).counts).all()
    assert nfc.n_unusable_responses == nfd.n_unusable_responses == 0


def test_to_nfc_is_idempotent_and_leaves_non_hangul_alone():
    assert to_nfc(to_nfc(_nfd("한국어"))) == to_nfc(_nfd("한국어"))
    assert to_nfc("abc 123") == "abc 123"


def test_genuinely_non_hangul_input_is_still_rejected():
    """정규화가 '무엇이든 통과시키는' 것으로 번지면 안 됩니다."""
    assert parse_response("각", "abc").quality is ResponseQuality.NON_HANGUL


# --------------------------------------------------------------------- 설명 항목 개수


@pytest.fixture
def matrix():
    responses = ["각"] * 5 + ["닥"] * 3 + ["박"] * 2
    profile = ConfusionProfile.from_trials(
        "L",
        [
            CalibrationTrial(stimulus_id=f"s{i}", target="각", response=r)
            for i, r in enumerate(responses)
        ],
        is_synthetic=True,
    )
    return profile.matrix(Position.ONSET)


@pytest.mark.parametrize(("k", "expected"), [(3, 3), (2, 2), (1, 1), (0, 0), (-1, 0)])
def test_top_confusions_returns_exactly_k_entries(matrix, k, expected):
    """회귀 테스트.

    한계 검사가 append **뒤에** 있어서 ``k=0`` 이 1개를, ``k=-1`` 도 1개를 돌려줬습니다.
    "설명을 붙이지 말라" 고 요청한 호출자가 조용히 설명을 받았습니다.
    """
    assert len(matrix.top_confusions("ㄱ", k=k)) == expected


def test_top_confusions_never_includes_the_target_itself(matrix):
    assert all(label != "ㄱ" for label, _p, _n in matrix.top_confusions("ㄱ", k=5))


def test_top_confusions_are_ordered_by_probability(matrix):
    probs = [p for _label, p, _n in matrix.top_confusions("ㄱ", k=5)]
    assert probs == sorted(probs, reverse=True)
