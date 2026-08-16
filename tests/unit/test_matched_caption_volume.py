"""P0.6 — RQ3 의 "동일 자막량" 주장이 실제로 참인지.

RQ3 는 전역 임계값과 청취자별 임계값을 **같은 자막량에서** 비교한다고 말합니다.
임계값 방식(`p > tau`)은 동점이 없을 때만 그 주장을 지킵니다. 동점이 많으면 경계에서
한꺼번에 포함되거나 제외되어 실제 자막 개수가 목표에서 크게 벗어납니다.

수정 전 실측: 기록된 5시드 실행에서 두 arm 의 자막량 차이는 최대 0.0011(0.11%p)에
불과했습니다. 즉 **기존에 보고된 RQ3 수치는 실질적으로 영향받지 않았습니다.**
그러나 그것은 로지스틱 확률이 거의 연속적이었던 덕분이지 강제된 성질이 아닙니다.
아래 첫 번째 테스트가 그 취약성을 드러냅니다.

강제하는 불변식:

1. 목표 비율이 주어지면 두 arm 모두 **정확한 개수**를 선택한다.
2. 동점 처리는 결정론적이다 — 같은 입력은 항상 같은 선택을 낸다.
3. 목표/달성 비율과 개수, 경계 동점 수를 모두 보고한다.
4. 총합 재현율과 청취자별 분포(중앙값·최하위)를 분리해 보고한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from audire.eval.ablation import ArmResult
from audire.eval.caption import compare_thresholds, select_exact_count
from audire.eval.metrics import compute_metrics


def _arm(y: np.ndarray, groups: np.ndarray, p: np.ndarray) -> ArmResult:
    m = compute_metrics(y, p)
    return ArmResult(
        arm="t",
        model="m",
        seed=0,
        calibration="none",
        n_listeners=int(np.unique(groups).size),
        n_trials=int(y.size),
        n_features=1,
        metrics=m,
        prevalence_floor=m,
        intervals={},
        reliability={},
        model_description={},
        fold_sizes=[],
        y_true=y,
        y_prob=p,
        groups=groups,
    )


def _tie_heavy(n_listeners: int = 8, n_words: int = 50, n_levels: int = 3):
    """점수가 몇 개 값에만 몰린, 동점이 지배적인 상황.

    `word_context_only` arm 이 실제로 이렇게 동작합니다. 같은 단어는 같은 특징을 갖고
    청취자 정보가 없으므로 정확히 같은 확률을 받습니다.
    """
    rng = np.random.default_rng(0)
    groups = np.array(
        [f"L{i:02d}" for i in range(n_listeners) for _ in range(n_words)], dtype=np.str_
    )
    levels = np.linspace(0.2, 0.8, n_levels)
    p = levels[rng.integers(0, n_levels, size=groups.size)]
    y = (rng.random(groups.size) < p).astype(np.int64)
    return y, groups, p


# =========================================================== 취약성 재현


def test_tie_heavy_scores_break_a_threshold_based_volume_match() -> None:
    """동점이 지배적이면 순수 임계값 선택은 목표 자막량을 크게 벗어난다.

    이것이 수정의 근거다. 로지스틱 확률에서는 우연히 문제가 드러나지 않았을 뿐이다.
    """
    _y, groups, p = _tie_heavy()
    from audire.caption.policy import global_threshold

    tau = global_threshold({str(g): p[groups == g] for g in np.unique(groups)}, 0.20)
    achieved = float((p > tau).mean())
    assert abs(achieved - 0.20) > 0.05, (
        f"동점이 지배적인데도 임계값이 목표를 맞췄다(달성 {achieved:.3f}). "
        f"이 테스트의 전제가 무너졌으니 픽스처를 다시 봐야 한다."
    )


# =========================================================== 정확 개수 선택


def test_select_exact_count_returns_exactly_n() -> None:
    _y, _g, p = _tie_heavy()
    for n in (0, 1, 37, 100, p.size):
        assert int(select_exact_count(p, n).sum()) == n, n


def test_select_exact_count_is_deterministic_under_ties() -> None:
    _y, _g, p = _tie_heavy()
    a = select_exact_count(p, 137)
    b = select_exact_count(p, 137)
    assert np.array_equal(a, b)


def test_select_exact_count_prefers_higher_scores() -> None:
    p = np.array([0.9, 0.1, 0.5, 0.7])
    chosen = select_exact_count(p, 2)
    assert chosen.tolist() == [True, False, False, True]


def test_select_exact_count_rejects_impossible_requests() -> None:
    p = np.zeros(5)
    with pytest.raises(ValueError, match=r"개수|count"):
        select_exact_count(p, 6)
    with pytest.raises(ValueError, match=r"개수|count"):
        select_exact_count(p, -1)


# =========================================================== RQ3 비교


@pytest.mark.parametrize("target", [0.10, 0.20, 0.30, 0.50])
def test_both_arms_achieve_the_same_caption_volume(target: float) -> None:
    """가장 중요한 불변식: '동일 자막량' 주장이 실제로 참이어야 한다."""
    y, groups, p = _tie_heavy()
    cmp = compare_thresholds(_arm(y, groups, p), target)

    g, pe = cmp.global_point.achieved_ratio, cmp.personalized_point.achieved_ratio
    # 청취자별 반올림 때문에 완전 동일할 수는 없으나 1개 단어 수준이어야 한다.
    assert abs(g - pe) <= 1.0 / y.size + 1e-9, f"자막량이 어긋났다: {g:.5f} vs {pe:.5f}"
    assert abs(g - target) <= 1.0 / y.size + 1e-9


def test_comparison_reports_counts_and_tie_behaviour() -> None:
    y, groups, p = _tie_heavy()
    d = compare_thresholds(_arm(y, groups, p), 0.20).to_dict()

    assert d["target_ratio"] == 0.20
    assert d["target_count"] == round(0.20 * y.size)
    assert d["global"]["achieved_count"] == d["target_count"]
    assert d["selection"] == "exact_count"
    # 경계 동점 수를 숨기지 않는다.
    assert "n_ties_at_boundary" in d["global"]
    assert d["global"]["n_ties_at_boundary"] >= 1, "동점 픽스처인데 0 으로 보고됐다"
    assert "tie_break" in d


def test_comparison_separates_aggregate_from_listener_distribution() -> None:
    """총합 재현율만 보고하면 자막을 못 받는 청취자를 숨길 수 있다."""
    y, groups, p = _tie_heavy()
    d = compare_thresholds(_arm(y, groups, p), 0.20).to_dict()
    for arm in ("global", "personalized"):
        assert "misheard_recall" in d[arm]
        assert "recall_median" in d[arm]
        assert "recall_min" in d[arm]
    assert "equity" in d


def test_global_threshold_can_starve_listeners_and_this_is_visible() -> None:
    """전역 선택이 특정 청취자에게 자막을 거의 주지 않는 상황이 지표로 드러나야 한다."""
    groups = np.array(["low"] * 100 + ["high"] * 100, dtype=np.str_)
    p = np.concatenate([np.full(100, 0.05), np.full(100, 0.95)])
    y = np.ones(200, dtype=np.int64)

    d = compare_thresholds(_arm(y, groups, p), 0.50).to_dict()
    assert d["global"]["recall_min"] == pytest.approx(0.0)
    assert d["personalized"]["recall_min"] == pytest.approx(0.5)
    # 총합만 보면 전역이 동점이거나 우세해 보인다.
    assert d["global"]["misheard_recall"] >= d["personalized"]["misheard_recall"] - 1e-9


def test_comparison_is_deterministic() -> None:
    y, groups, p = _tie_heavy()
    a = compare_thresholds(_arm(y, groups, p), 0.20).to_dict()
    b = compare_thresholds(_arm(y, groups, p), 0.20).to_dict()
    assert a["global"] == b["global"]
    assert a["personalized"] == b["personalized"]
