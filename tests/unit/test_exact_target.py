"""Phase D — 목표 음소를 개별 지목하는 블록의 불변식.

미션이 요구하는 항목: 특징 목록이 결정적인가, 같은 단어/프로파일이 안정적인 열 순서를
내는가, 단어에 있는 음소만 출현 상호작용을 활성화하는가, 미관측 행이 불확실성을 드러내는가,
그리고 홀드아웃 청취자 정보가 몰래 섞이지 않는가.
"""

from __future__ import annotations

import numpy as np
import pytest

from audire.confusion import CalibrationTrial, ConfusionProfile
from audire.hangul.inventory import Position
from audire.risk import ABLATION_ARMS, BLOCK_PREFIXES
from audire.risk.exact_target import (
    exact_target_features,
    exact_target_offdiag_features,
    n_exact_target_features,
    n_exact_target_offdiag_features,
    word_phoneme_counts,
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


# --------------------------------------------------------------------- 차원과 결정성


def test_column_count_is_fixed_by_the_inventory_not_the_data():
    """데이터에 따라 열이 늘면 절제 비교와 모델 저장이 깨집니다."""
    assert n_exact_target_features() == 68  # 초성 19 + 중성 21 + 종성 28
    assert n_exact_target_offdiag_features() == 18


def test_column_set_is_identical_across_words_and_listeners(flat):
    other = _profile([("삭", "박")] * 5, listener_id="L2")
    sets = [
        set(exact_target_features(w, p)) for w in ("가", "한국어", "ABC") for p in (flat, other)
    ]
    assert all(s == sets[0] for s in sets)
    assert len(sets[0]) == n_exact_target_features()


def test_column_order_is_stable(flat):
    a = list(exact_target_features("한국어", flat))
    b = list(exact_target_features("한국어", flat))
    assert a == b


def test_values_are_deterministic(flat):
    assert exact_target_features("한국어", flat) == exact_target_features("한국어", flat)
    assert exact_target_offdiag_features("한국어", flat) == exact_target_offdiag_features(
        "한국어", flat
    )


def test_all_values_are_finite(flat):
    for fn in (exact_target_features, exact_target_offdiag_features):
        values = np.asarray(list(fn("한국어", flat).values()), dtype=np.float64)
        assert np.isfinite(values).all()


# ------------------------------------------------------------------ 출현 상호작용


def test_only_phonemes_present_in_the_word_are_activated(flat):
    """단어에 없는 음소가 0 이 아니면 그 열은 청취자 정보를 단어와 무관하게 흘립니다."""
    counts = word_phoneme_counts("가")
    present = {
        f"{position.value}{i:02d}"
        for position in Position
        for i, jamo in enumerate(
            __import__("audire.risk.exact_target", fromlist=["_INVENTORY"])._INVENTORY[position]
        )
        if jamo in counts[position]
    }
    features = exact_target_features("가", flat)
    for name, value in features.items():
        token = name.removeprefix("et_")
        if token not in present:
            assert value == 0.0, name


def test_occurrence_count_scales_the_feature():
    """같은 음소가 두 번 나오면 두 배가 되어야 '출현 횟수 × 오류 확률' 이라 말할 수 있습니다."""
    listener = _profile([("각", "닥")] * 20)
    once = exact_target_features("가", listener)
    twice = exact_target_features("가가", listener)
    key = next(k for k, v in once.items() if v > 0)
    assert twice[key] == pytest.approx(2 * once[key])


def test_a_listener_specific_weakness_shows_up_on_that_phoneme():
    """부류 평균이 지우는 것을 이 블록이 남기는지 — 존재 이유의 검증."""
    # ㄱ 은 잘 듣고 ㅋ 은 거의 못 듣는 청취자. 둘 다 파열음이라 부류 평균은 같아집니다.
    listener = _profile([("각", "각")] * 30 + [("칵", "탁")] * 30)
    g = exact_target_features("가", listener)
    k = exact_target_features("카", listener)
    g_active = max(v for v in g.values() if v > 0)
    k_active = max(v for v in k.values() if v > 0)
    assert k_active > g_active


def test_a_word_with_no_hangul_activates_nothing(flat):
    features = exact_target_features("ABC123", flat)
    assert set(features) == set(exact_target_features("가", flat))
    assert all(v == 0.0 for v in features.values())


# ------------------------------------------------------------- 오프대각 구조 (D2/D3)


def test_within_and_across_class_mass_sum_to_the_error_mass():
    listener = _profile([("각", "각")] * 10 + [("각", "닥")] * 10)
    features = exact_target_offdiag_features("가", listener)
    total = features["eo_onset_within_class"] + features["eo_onset_across_class"]
    assert total == pytest.approx(features["eo_onset_expected_errors"], rel=1e-9)


def test_expected_errors_is_a_sum_so_word_length_survives():
    """평균으로 만들면 '이 위치에서 오류가 몇 개나 날 것인가' 가 사라집니다."""
    listener = _profile([("각", "닥")] * 20)
    short = exact_target_offdiag_features("가", listener)
    long = exact_target_offdiag_features("가가가", listener)
    assert long["eo_onset_expected_errors"] > short["eo_onset_expected_errors"]


def test_ratio_features_do_not_grow_with_word_length():
    listener = _profile([("각", "닥")] * 20)
    short = exact_target_offdiag_features("가", listener)
    long = exact_target_offdiag_features("가가가", listener)
    for key in ("eo_onset_within_class", "eo_onset_top1_share", "eo_onset_evidence"):
        assert long[key] == pytest.approx(short[key])


def test_positions_are_kept_separate():
    """V2 의 복구 기전이 위치별로 다르므로 위치를 합치면 그 구조가 사라집니다."""
    listener = _profile([("각", "닥")] * 20)
    features = exact_target_offdiag_features("각", listener)
    for position in ("onset", "nucleus", "coda"):
        assert f"eo_{position}_expected_errors" in features


def test_unobserved_targets_expose_their_uncertainty():
    """증거가 없는 음소는 확신처럼 보이면 안 됩니다."""
    thin = _profile([("각", "각")] * 2)
    thick = _profile([("각", "각")] * 200, listener_id="L2")
    assert (
        exact_target_offdiag_features("가", thin)["eo_onset_uncertainty"]
        > exact_target_offdiag_features("가", thick)["eo_onset_uncertainty"]
    )
    assert (
        exact_target_offdiag_features("가", thin)["eo_onset_evidence"]
        < exact_target_offdiag_features("가", thick)["eo_onset_evidence"]
    )


# --------------------------------------------------------------------------- 배선


def test_new_arms_are_additive_and_old_arms_survive():
    """기존 arm 을 지우면 어느 블록이 무엇을 했는지 말할 수 없습니다."""
    rich = set(ABLATION_ARMS["clinical_plus_confusion_rich"])
    assert rich < set(ABLATION_ARMS["exact_target"])
    assert set(ABLATION_ARMS["exact_target"]) < set(ABLATION_ARMS["exact_target_offdiag"])
    # 원래 arm 들이 그대로 남아 있어야 합니다.
    assert ABLATION_ARMS["clinical_plus_confusion"] == ("word", "context", "clinical", "confusion")


def test_every_column_matches_its_declared_block_prefix(flat):
    """접두사는 절제가 블록을 잘라내는 근거입니다. 어긋나면 누출입니다."""
    for name in exact_target_features("한국어", flat):
        assert name.startswith(BLOCK_PREFIXES["exact_target"]), name
    for name in exact_target_offdiag_features("한국어", flat):
        assert name.startswith(BLOCK_PREFIXES["exact_target_offdiag"]), name


def test_the_two_blocks_do_not_share_column_names(flat):
    a = set(exact_target_features("한국어", flat))
    b = set(exact_target_offdiag_features("한국어", flat))
    assert a.isdisjoint(b)


def test_arm_adds_exactly_the_declared_columns():
    from audire.profile.schema import (
        Audiogram,
        AudiogramPoint,
        Ear,
        EarProfile,
        HearingProfile,
        ProfileSource,
        SpeechScores,
    )
    from audire.risk import FeatureSpec, WordContext

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
    hearing = HearingProfile(
        listener_id="L1", source=ProfileSource.SYNTHETIC, is_synthetic=True, right=ear
    )
    confusion = _profile([("각", "각")])
    speakers = ("male", "female", "unknown")
    context = WordContext(snr_db=20.0, speaker="male")

    sizes = {
        arm: len(FeatureSpec.arm(arm, speakers=speakers).row("한국어", context, hearing, confusion))
        for arm in ("clinical_plus_confusion_rich", "exact_target", "exact_target_offdiag")
    }
    assert (
        sizes["exact_target"] == sizes["clinical_plus_confusion_rich"] + n_exact_target_features()
    )
    assert (
        sizes["exact_target_offdiag"] == sizes["exact_target"] + n_exact_target_offdiag_features()
    )


def test_features_never_see_more_than_one_listener(flat):
    """이 블록은 넘겨받은 프로파일 하나만 봅니다. 집단 통계를 몰래 쓰면 누출입니다."""
    a = _profile([("각", "닥")] * 10, listener_id="A")
    b = _profile([("각", "박")] * 10, listener_id="B")
    # A 의 특징은 B 가 존재하든 말든 같아야 합니다.
    before = exact_target_features("가", a)
    _ = exact_target_features("가", b)
    assert exact_target_features("가", a) == before
