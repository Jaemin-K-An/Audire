"""Phase D17 — 상호작용 기반 혼동 특징 블록.

이 블록의 존재 이유는 단 하나입니다: "이 청취자가 약한 음운 부류" x "이 단어가 그 부류를
포함하는 정도". 집계 평균으로는 표현할 수 없는 이 신호가 실제로 살아 있는지, 그리고
차원 수가 통제된 채로 유지되는지를 검증합니다.
"""

from __future__ import annotations

import numpy as np
import pytest

from audire.confusion import CalibrationTrial, ConfusionProfile
from audire.hangul.inventory import NO_CODA, Position, categories_for
from audire.hangul.syllable import HANGUL_SYLLABLE_START, compose_syllable
from audire.risk import (
    ABLATION_ARMS,
    BLOCK_PREFIXES,
    FeatureSpec,
    WordContext,
    confusion_rich_features,
    listener_class_weakness,
    listener_global_error,
    n_rich_features,
    word_class_share,
)
from audire.risk.confusion_features import (
    _CODA_SURFACES,
    _NUCLEUS_SHAPES,
    _ONSET_MANNERS,
    _ONSET_PHONATIONS,
)


def _profile(pairs: list[tuple[str, str]], listener_id: str = "L1") -> ConfusionProfile:
    return ConfusionProfile.from_trials(
        listener_id,
        [
            CalibrationTrial(stimulus_id=f"s{i}", target=t, response=r)
            for i, (t, r) in enumerate(pairs)
        ],
        is_synthetic=True,
    )


@pytest.fixture(scope="module")
def flat() -> ConfusionProfile:
    return _profile([("각", "각")])


@pytest.fixture(scope="module")
def hearing():
    from audire.profile.schema import (
        Audiogram,
        AudiogramPoint,
        Ear,
        EarProfile,
        HearingProfile,
        ProfileSource,
        SpeechScores,
    )

    ear = EarProfile(
        ear=Ear.RIGHT,
        audiogram=Audiogram(
            ear=Ear.RIGHT,
            thresholds={f: AudiogramPoint(db_hl=40.0) for f in (500, 1000, 2000, 4000)},
        ),
        speech=SpeechScores(
            ear=Ear.RIGHT,
            srt_db_hl=40.0,
            wrs_percent=70.0,
            wrs_presentation_level_db_hl=70.0,
        ),
    )
    return HearingProfile(
        listener_id="L1", source=ProfileSource.SYNTHETIC, is_synthetic=True, right=ear
    )


# ------------------------------------------------------------------------- 부류 목록 완전성


def test_class_inventories_match_the_grouping_tables():
    """부류 목록은 음운 분류표에서 파생되어야 하며, 손으로 다시 적어서는 안 됩니다."""
    from audire.confusion.grouping import NUCLEUS_SHAPE, ONSET_MANNER, ONSET_PHONATION

    assert set(_ONSET_MANNERS) == {m.value for m in ONSET_MANNER.values()}
    assert set(_ONSET_PHONATIONS) == {p.value for p in ONSET_PHONATION.values()}
    assert set(_NUCLEUS_SHAPES) == {s.value for s in NUCLEUS_SHAPE.values()}
    # 종성은 7종 대표음 + 무종성 (음절의 끝소리 규칙).
    assert len(_CODA_SURFACES) == 8


def test_every_hangul_syllable_produces_a_full_feature_vector(flat):
    """회귀 테스트.

    부류 목록을 손으로 적었던 초기 구현은 공명음 발성유형(유음/비음/무초성)을 빠뜨렸고,
    ``ㄴ/ㄹ/ㅁ/ㅇ`` 로 시작하는 모든 단어에서 KeyError 로 터졌습니다. 음절 공간 전체를
    훑어 그 부류의 오류가 다시 들어올 수 없게 합니다.
    """
    onsets = list(categories_for(Position.ONSET, axis="target"))
    nuclei = list(categories_for(Position.NUCLEUS, axis="target"))
    codas = list(categories_for(Position.CODA, axis="target"))
    expected = n_rich_features()

    rng = np.random.default_rng(0)
    # 11,172 음절 전수는 느리므로, 모든 초성 x 모든 중성은 전수로, 종성은 무작위 표본으로.
    for onset in onsets:
        for nucleus in nuclei:
            coda = codas[int(rng.integers(len(codas)))]
            syllable = compose_syllable(onset, nucleus, coda)
            assert len(confusion_rich_features(syllable, flat)) == expected


def test_all_onset_classes_are_reachable_from_real_syllables(flat):
    """어떤 발성유형도 '어떤 음절로도 도달할 수 없는' 죽은 열이 되어서는 안 됩니다."""
    seen: set[str] = set()
    for onset in categories_for(Position.ONSET, axis="target"):
        share = word_class_share(compose_syllable(onset, "ㅏ", NO_CODA))
        seen |= {k for k, v in share.items() if k.startswith("sh_onset_phon_") and v > 0}
    assert seen == {f"sh_onset_phon_{p}" for p in _ONSET_PHONATIONS}


# --------------------------------------------------------------------------- 상호작용의 성질


def test_shares_sum_to_one_within_each_class_family():
    share = word_class_share("한국어")
    for family in ("sh_onset_manner_", "sh_onset_phon_", "sh_nucleus_", "sh_coda_"):
        total = sum(v for k, v in share.items() if k.startswith(family))
        assert total == pytest.approx(1.0), family


def test_share_is_zero_for_absent_classes():
    # '아' = 무초성(ㅇ) + ㅏ + 무종성.
    share = word_class_share("아")
    assert share["sh_onset_manner_none"] == pytest.approx(1.0)
    assert share["sh_onset_manner_stop"] == 0.0
    assert share["sh_coda_none"] == pytest.approx(1.0)


def test_interaction_vanishes_when_either_half_is_zero(flat):
    """ix = 약점 x 포함비율. 한쪽이 0 이면 반드시 0 이어야 곱셈항이라 말할 수 있습니다."""
    features = confusion_rich_features("아", flat)
    share = word_class_share("아")
    for key, value in share.items():
        if value == 0.0:
            assert features[f"ix_{key.removeprefix('sh_')}"] == 0.0


def test_interaction_equals_the_product_of_its_halves(flat):
    features = confusion_rich_features("한국어", flat)
    weakness = listener_class_weakness(flat)
    share = word_class_share("한국어")
    for wk_key, wk_value in weakness.items():
        class_key = wk_key.removeprefix("wk_")
        assert features[f"ix_{class_key}"] == pytest.approx(wk_value * share[f"sh_{class_key}"])


def test_a_targeted_weakness_shows_up_only_in_the_matching_interaction():
    """핵심 주장: 부류별 약점이 그 부류를 포함한 단어에서만 위험을 올려야 합니다.

    두 청취자 모두 파열음과 마찰음 **양쪽에** 같은 양의 증거를 갖게 하고, 오류의 분포만
    반대로 둡니다. 한쪽 부류를 미관측으로 남기면 '증거 없음' 처리 규칙을 시험하는 것이지
    상호작용을 시험하는 것이 아니게 됩니다.
    """
    # 파열음(ㅂ)에서 25/30 틀리고 마찰음(ㅅ)에서 5/30 틀리는 청취자.
    weak_stop = _profile(
        [("박", "박")] * 5 + [("박", "각")] * 25 + [("삭", "삭")] * 25 + [("삭", "각")] * 5
    )
    # 정확히 반대. 전체 오답률은 동일합니다.
    weak_fric = _profile(
        [("박", "박")] * 25 + [("박", "각")] * 5 + [("삭", "삭")] * 5 + [("삭", "각")] * 25,
        listener_id="L2",
    )

    stop_word, fric_word = "바보", "사수"
    f_stop_on_stopword = confusion_rich_features(stop_word, weak_stop)
    f_fric_on_stopword = confusion_rich_features(stop_word, weak_fric)

    # 파열음 단어에서는 파열음 약점 청취자의 상호작용이 더 커야 합니다.
    assert f_stop_on_stopword["ix_onset_manner_stop"] > f_fric_on_stopword["ix_onset_manner_stop"]
    # 반대 방향도 성립해야 대칭적인 주장이 됩니다.
    f_stop_on_fricword = confusion_rich_features(fric_word, weak_stop)
    f_fric_on_fricword = confusion_rich_features(fric_word, weak_fric)
    assert (
        f_fric_on_fricword["ix_onset_manner_fricative"]
        > f_stop_on_fricword["ix_onset_manner_fricative"]
    )


def test_absence_of_evidence_is_not_treated_as_evidence_of_weakness():
    """회귀 테스트.

    균일 사전분포에서 미관측 target 의 ``p_correct`` 는 1/20 ≈ 0.05 입니다. 초기 구현은
    부류 약점을 모든 구성원의 평활된 값으로 평균 내면서 미관측 행을 포함시켰고, 그 결과
    **한 번도 제시되지 않은 부류**(약점 0.95)가 **실제로 틀린 부류**(0.84)보다 더 약한
    것으로 나왔습니다. 25회 교정에서는 대부분의 부류가 미관측이므로 이 인공물이 상호작용
    항 전체를 지배했을 것입니다.
    """
    # 파열음에서 25/30 틀림. 마찰음은 한 번도 제시되지 않음.
    listener = _profile([("박", "박")] * 5 + [("박", "각")] * 25)
    weakness = listener_class_weakness(listener)

    measured = weakness["wk_onset_manner_stop"]
    unobserved = weakness["wk_onset_manner_fricative"]
    # 측정된 약점이 미관측 부류보다 커야 합니다.
    assert measured > unobserved
    # 측정값은 관측된 오답률을 따라야 합니다 (평활 때문에 정확히 25/30 은 아님).
    assert measured == pytest.approx(0.84, abs=0.05)
    # 미관측 부류는 청취자의 전체 오답률로 후퇴합니다 — 0.95 같은 인공물이 아니라.
    assert unobserved == pytest.approx(listener_global_error(listener))


def test_a_listener_with_no_evidence_at_all_is_neutral():
    """관측이 전혀 없으면 0.5 — 최대한 무정보이지, 최대한 위험이 아닙니다."""
    empty = ConfusionProfile.empty("L-empty", is_synthetic=True)
    assert listener_global_error(empty) == pytest.approx(0.5)
    assert all(v == pytest.approx(0.5) for v in listener_class_weakness(empty).values())


def test_two_words_with_equal_mean_risk_can_differ_in_the_block():
    """집계 평균이 표현하지 못하는 것을 이 블록이 표현한다는 존재 이유 자체의 검증."""
    weak_stop = _profile([("박", "박")] * 5 + [("박", "각")] * 25)
    a = confusion_rich_features("바보", weak_stop)
    b = confusion_rich_features("사수", weak_stop)
    assert a != b


# ------------------------------------------------------------------------------- 불확실성


def test_thin_evidence_is_reported_not_concealed():
    """관측 2회의 0.9 와 관측 200회의 0.9 는 다른 증거이고, 특징이 이를 구분해야 합니다."""
    thin = _profile([("각", "각")] * 2)
    thick = _profile([("각", "각")] * 200, listener_id="L2")
    f_thin = confusion_rich_features("각", thin)
    f_thick = confusion_rich_features("각", thick)

    assert f_thin["x2_post_sd_mean"] > f_thick["x2_post_sd_mean"]
    assert f_thin["x2_frac_thin_evidence"] > f_thick["x2_frac_thin_evidence"]


def test_concentrated_and_diffuse_error_mass_are_distinguished():
    """같은 오답률이라도 한 이웃에 몰린 오류와 흩어진 오류는 예측 가능성이 다릅니다."""
    concentrated = _profile([("각", "각")] * 20 + [("각", "닥")] * 20)
    diffuse = _profile(
        [("각", "각")] * 20 + [("각", "닥")] * 7 + [("각", "박")] * 7 + [("각", "삭")] * 6,
        listener_id="L2",
    )
    assert (
        confusion_rich_features("각", concentrated)["x2_top1_share"]
        > confusion_rich_features("각", diffuse)["x2_top1_share"]
    )


def test_worst_segment_emphasis_tracks_the_tail_not_the_mean(flat):
    """단어는 한 분절만 실패해도 오청이므로 최악값이 평균보다 중요합니다."""
    features = confusion_rich_features("한국어", flat)
    assert 0.0 <= features["x2_worst_two_mean"] <= 1.0
    assert 0.0 <= features["x2_p_spread"] <= 1.0
    for position in ("onset", "nucleus", "coda"):
        assert 0.0 <= features[f"x2_min_p_{position}"] <= 1.0


def test_empty_word_degrades_to_defined_values(flat):
    """한글이 없는 입력에서도 NaN 없이 정의된 값이 나와야 합니다."""
    features = confusion_rich_features("ABC", flat)
    assert len(features) == n_rich_features()
    assert all(np.isfinite(v) for v in features.values())


# ---------------------------------------------------------------------------- 블록 배선


def test_block_is_additive_and_the_old_arm_survives():
    """기존 arm 이 그대로 남아 있어야 비교가 가능합니다 (의미론을 조용히 바꾸지 않음)."""
    assert ABLATION_ARMS["clinical_plus_confusion"] == ("word", "context", "clinical", "confusion")
    rich = ABLATION_ARMS["clinical_plus_confusion_rich"]
    assert set(ABLATION_ARMS["clinical_plus_confusion"]).issubset(rich)
    assert "confusion_rich" in rich


def test_every_rich_column_matches_the_declared_block_prefixes(flat):
    """열 이름 접두사는 ablation 이 블록을 잘라내는 근거입니다. 어긋나면 누출입니다."""
    prefixes = BLOCK_PREFIXES["confusion_rich"]
    for name in confusion_rich_features("한국어", flat):
        assert name.startswith(prefixes), name


def test_rich_arm_adds_exactly_the_rich_block(flat, hearing):
    speakers = ("male", "female", "unknown")
    context = WordContext(snr_db=5.0, speaker="male")
    base = FeatureSpec.arm("clinical_plus_confusion", speakers=speakers)
    rich = FeatureSpec.arm("clinical_plus_confusion_rich", speakers=speakers)
    n_base = len(base.row("한국어", context, hearing, flat))
    n_rich = len(rich.row("한국어", context, hearing, flat))
    assert n_rich == n_base + n_rich_features()


def test_dimensionality_is_bounded_and_stated():
    """차원 통제는 명시적 약속입니다. 늘어나면 테스트가 먼저 알려야 합니다."""
    assert n_rich_features() == 59


def test_row_is_deterministic(flat):
    a = confusion_rich_features("한국어", flat)
    b = confusion_rich_features("한국어", flat)
    assert a == b


def test_feature_spec_row_produces_finite_values(flat, hearing):
    """FeatureSpec 경로로도 NaN/inf 없이 나와야 합니다 (모델이 결측을 상속하지 않도록)."""
    spec = FeatureSpec.arm("confusion_rich_only", speakers=("male", "unknown"))
    row = spec.row("한국어", WordContext(snr_db=5.0, speaker="male"), hearing, flat)
    assert np.isfinite(np.asarray(list(row.values()), dtype=np.float64)).all()


def test_hangul_syllable_block_is_unchanged():
    """음절 블록이 옮겨가면 이 파일의 전수 검사가 무의미해집니다."""
    assert HANGUL_SYLLABLE_START == 0xAC00
