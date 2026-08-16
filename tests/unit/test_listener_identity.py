"""P0.5 — 청취자 신원 불변식.

한 청취자를 **다른 청취자의 프로파일로 채점하는 일이 절대 없어야** 합니다.
그런 일이 조용히 벌어지면 개인화 시스템의 출력 전체가 무의미해지고, 접근성 도구로서는
잘못된 사람에게 맞춘 자막을 보여 주게 됩니다.

수정 전 결함:
- `check_ready()`는 `listener_id`를 세 입력 사이에서 전혀 비교하지 않았다.
  청취자 A의 id로 B의 청력 프로파일과 C의 혼동 프로파일을 넘겨도 그대로 채점했다.
- 합성/실측 출처 호환성 검사가 없었다. 실측 청력 프로파일에 합성 혼동 프로파일을
  섞어도 통과했다.
- id 검증이 계층마다 달랐다. `ProfileStore`는 안전한 알파벳을 강제했지만
  `HearingProfile` 스키마는 길이만 봤고 `ConfusionProfile`은 아무 검증도 없었다.
  즉 스키마가 받아들인 id를 저장소가 거부할 수 있었다.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from audire.confusion import CalibrationTrial, ConfusionProfile
from audire.profile import (
    Audiogram,
    AudiogramPoint,
    Ear,
    EarProfile,
    HearingProfile,
    ProfileSource,
    ProfileStore,
    SpeechScores,
    validate_listener_id,
)


def _hearing(listener_id: str, *, synthetic: bool = False) -> HearingProfile:
    ear = EarProfile(
        ear=Ear.RIGHT,
        audiogram=Audiogram(
            ear=Ear.RIGHT,
            thresholds={f: AudiogramPoint(db_hl=40.0) for f in (500, 1000, 2000, 4000)},
        ),
        speech=SpeechScores(
            ear=Ear.RIGHT, srt_db_hl=40.0, wrs_percent=70.0, wrs_presentation_level_db_hl=70.0
        ),
    )
    return HearingProfile(
        listener_id=listener_id,
        source=ProfileSource.SYNTHETIC if synthetic else ProfileSource.MANUAL,
        is_synthetic=synthetic,
        right=ear,
    )


def _confusion(listener_id: str, *, synthetic: bool = False, n: int = 30) -> ConfusionProfile:
    return ConfusionProfile.from_trials(
        listener_id,
        [CalibrationTrial(stimulus_id=f"s{i}", target="각", response="각") for i in range(n)],
        is_synthetic=synthetic,
    )


# =========================================================== 공유 ListenerId 검증


@pytest.mark.parametrize("good", ["L001", "SYN0042", "a", "p.2-x_9", "A" * 64])
def test_safe_ids_are_accepted(good: str) -> None:
    assert validate_listener_id(good) == good


@pytest.mark.parametrize(
    "bad",
    [
        "",  # 비어 있음
        "김철수",  # 사람 이름 — 불투명 식별자가 아니다
        "Kim Chul-Su",  # 공백 + 이름
        "../escape",  # 경로 탈출
        "a/b",  # 경로 구분자
        ".hidden",  # 영숫자로 시작하지 않음
        "A" * 65,  # 너무 김
        "id\x00null",  # 널 바이트
        "id\nnewline",  # 로그 위조 가능
    ],
)
def test_unsafe_ids_are_rejected_everywhere(bad: str) -> None:
    """스키마·프로파일·저장소가 **같은** 규칙을 써야 한다."""
    with pytest.raises((ValueError, ValidationError)):
        validate_listener_id(bad)
    with pytest.raises((ValueError, ValidationError)):
        _hearing(bad)
    with pytest.raises((ValueError, ValidationError)):
        ConfusionProfile.empty(bad, is_synthetic=True)


def test_schema_and_store_agree_on_what_is_valid(tmp_path) -> None:
    """스키마가 받아들인 id 를 저장소가 거부하면 안 된다(과거 불일치 회귀)."""
    store = ProfileStore(tmp_path)
    profile = _hearing("L-001.a_2")
    store.save_hearing(profile)
    assert store.load("L-001.a_2").hearing.listener_id == "L-001.a_2"


def test_aggregate_ids_are_reserved_and_documented() -> None:
    """집단 프로파일은 청취자가 아니므로 예약된 형태만 허용한다."""
    a = _confusion("A", synthetic=True)
    b = _confusion("B", synthetic=True)
    from audire.confusion import pool_profiles

    pooled = pool_profiles([a, b])
    assert pooled.listener_id == "__pooled__"
    # 예약 형태는 통과하되 일반 청취자 id 로는 쓸 수 없음이 드러나야 한다.
    assert validate_listener_id("__pooled__", allow_aggregate=True) == "__pooled__"
    with pytest.raises(ValueError, match=r"예약|reserved|집단|aggregate"):
        validate_listener_id("__pooled__")


# =========================================================== 채점 경계


def test_mismatched_hearing_listener_id_is_rejected(fitted_scorer) -> None:
    """A 의 id 로 B 의 청력 프로파일을 채점하려는 시도는 실패해야 한다."""
    from audire.asr.pipeline import check_ready

    scorer = fitted_scorer
    problems = check_ready(
        scorer, _hearing("LISTENER_B"), _confusion("LISTENER_A"), listener_id="LISTENER_A"
    )
    assert any("LISTENER_B" in p for p in problems), problems


def test_mismatched_confusion_listener_id_is_rejected(fitted_scorer) -> None:
    from audire.asr.pipeline import check_ready

    problems = check_ready(
        fitted_scorer, _hearing("LISTENER_A"), _confusion("LISTENER_C"), listener_id="LISTENER_A"
    )
    assert any("LISTENER_C" in p for p in problems), problems


def test_matching_ids_pass(fitted_scorer) -> None:
    from audire.asr.pipeline import check_ready

    assert (
        check_ready(
            fitted_scorer,
            _hearing("LISTENER_A"),
            _confusion("LISTENER_A"),
            listener_id="LISTENER_A",
        )
        == []
    )


def test_real_hearing_profile_with_synthetic_confusion_is_rejected(fitted_scorer) -> None:
    """합성 근거가 실측 청취자의 결과로 세탁되어서는 안 된다."""
    from audire.asr.pipeline import check_ready

    problems = check_ready(
        fitted_scorer,
        _hearing("L1", synthetic=False),
        _confusion("L1", synthetic=True),
        listener_id="L1",
    )
    assert any("synthetic" in p or "합성" in p for p in problems), problems


def test_synthetic_hearing_profile_with_real_confusion_is_rejected(fitted_scorer) -> None:
    from audire.asr.pipeline import check_ready

    problems = check_ready(
        fitted_scorer,
        _hearing("L1", synthetic=True),
        _confusion("L1", synthetic=False),
        listener_id="L1",
    )
    assert any("synthetic" in p or "합성" in p for p in problems), problems


def test_consistent_synthetic_provenance_passes(fitted_scorer) -> None:
    from audire.asr.pipeline import check_ready

    assert (
        check_ready(
            fitted_scorer,
            _hearing("SYN1", synthetic=True),
            _confusion("SYN1", synthetic=True),
            listener_id="SYN1",
        )
        == []
    )


# =========================================================== 파이프라인 통합


def test_pipeline_refuses_to_score_a_mismatched_listener(fitted_scorer) -> None:
    """가장 중요한 경로: 실제 채점 함수가 신원 불일치를 거부해야 한다."""
    from audire.asr.pipeline import IncompleteProfile, score_transcript
    from audire.asr.replay import transcript_from_dict

    transcript = transcript_from_dict(
        {
            "backend": "test",
            "model_id": "test",
            "language": "ko",
            "duration_s": 1.0,
            "tokens": [{"text": "가족", "start_s": 0.0, "end_s": 0.5, "confidence": 0.9}],
        }
    )
    with pytest.raises(IncompleteProfile, match=r"LISTENER_B|listener"):
        score_transcript(
            transcript,
            fitted_scorer,
            listener_id="LISTENER_A",
            hearing=_hearing("LISTENER_B", synthetic=True),
            confusion=_confusion("LISTENER_A", synthetic=True),
        )


def test_word_scorer_rejects_a_mismatched_confusion_profile(fitted_scorer) -> None:
    """채점기 자체도 방어해야 한다. 파이프라인만 검사하면 우회 경로가 남는다."""
    from audire.risk.features import WordContext

    with pytest.raises(ValueError, match=r"LISTENER_X|listener"):
        fitted_scorer.score(
            "LISTENER_A",
            ["가족"],
            [WordContext()],
            _hearing("LISTENER_A", synthetic=True),
            _confusion("LISTENER_X", synthetic=True),
        )
