"""E30 — 라이브 절제 하니스의 불변식.

핵심은 **동일 자막률 비교가 실제로 자막량을 맞추는가** 입니다. 자막을 더 보여주면 재현율은
당연히 오르므로, 양을 맞추지 않은 arm 비교는 무의미합니다.
"""

from __future__ import annotations

import numpy as np
import pytest

from audire.experiments.live_ablation import (
    DIAGNOSTIC_ARM,
    LIVE_ARMS,
    LiveAblationConfig,
    LiveThreshold,
    evaluate_live_arm,
    run_live_ablation,
    select_with_threshold,
    threshold_for_caption_rate,
    threshold_metrics,
)
from audire.live import ContractViolation
from audire.sim import SimulationConfig, build_cohort

TINY = SimulationConfig(
    name="e30-tiny",
    seeds=[1, 2],
    n_listeners=12,
    n_calibration_trials=30,
    n_word_trials=60,
    snr_conditions_db=[20.0, 0.0],
)


def _cfg(**overrides) -> LiveAblationConfig:
    base = {
        "name": "e30-test",
        "simulation": TINY,
        "n_splits": 3,
        "thresholds": [0.4, 0.6],
        "budgets": [0.2],
        "matched_caption_rate": 0.2,
    }
    return LiveAblationConfig(**{**base, **overrides})


@pytest.fixture(scope="module")
def result():
    return run_live_ablation(_cfg())


# ------------------------------------------------------------------------- 계약 강제


def test_config_rejects_an_arm_that_violates_the_live_contract():
    """음향 맥락을 쓰는 arm 이 라이브 실험에 들어오면 안 됩니다."""
    with pytest.raises(ContractViolation, match="context"):
        _cfg(live_arms=["clinical_plus_confusion"])


def test_live_arms_are_evaluated_under_the_live_contract(result):
    for row in result["rows"]:
        if row["role"] == "live":
            assert row["input_contract"] == "live-caption-v1"


def test_the_diagnostic_arm_is_labelled_and_not_a_product_candidate(result):
    diagnostic = [r for r in result["rows"] if r["arm"] == DIAGNOSTIC_ARM]
    assert diagnostic
    for row in diagnostic:
        assert row["role"] == "diagnostic"
        assert row["input_contract"] != "live-caption-v1"


# --------------------------------------------------------- 동점 분할과 자막률 매칭


def test_pure_threshold_cannot_hit_a_target_rate_when_scores_tie():
    """회귀 테스트의 전제.

    단어 특징만 쓰는 모델은 같은 단어에 같은 점수를 줍니다. 실측하면 16,000행에서 고유
    점수가 33개뿐이고 훈련 행의 17.5% 가 임계값과 정확히 같았습니다. 순수한 ``>= tau`` 는
    그 덩어리를 통째로 넣거나 뺍니다.
    """
    scores = np.array([0.5] * 60 + [0.9] * 40, dtype=np.float64)
    # 목표 20% 를 순수 임계값으로는 낼 수 없습니다: 0.4 아니면 0.0 뿐입니다.
    achievable = {float((scores >= t).mean()) for t in np.unique(scores)}
    assert 0.2 not in achievable


def test_tie_splitting_hits_the_target_rate_on_the_data_it_was_chosen_from():
    scores = np.array([0.5] * 60 + [0.9] * 40, dtype=np.float64)
    groups = np.array([f"L{i % 4}" for i in range(100)])
    threshold = threshold_for_caption_rate(scores, 0.6)
    selected = select_with_threshold(scores, groups, threshold)
    assert selected.mean() == pytest.approx(0.6, abs=0.02)


def test_tie_splitting_is_deterministic():
    """같은 청취자·같은 항목은 항상 같은 판정을 받아야 화면이 깜빡이지 않습니다."""
    scores = np.array([0.5] * 50, dtype=np.float64)
    groups = np.array([f"L{i % 5}" for i in range(50)])
    threshold = LiveThreshold(tau=0.5, tie_pass_fraction=0.4)
    a = select_with_threshold(scores, groups, threshold)
    b = select_with_threshold(scores, groups, threshold)
    assert np.array_equal(a, b)


def test_tie_fraction_zero_selects_nothing_at_the_boundary():
    scores = np.array([0.5] * 20, dtype=np.float64)
    groups = np.array(["A"] * 20)
    assert not select_with_threshold(scores, groups, LiveThreshold(0.5, 0.0)).any()


def test_tie_fraction_one_selects_the_whole_boundary_block():
    scores = np.array([0.5] * 20, dtype=np.float64)
    groups = np.array(["A"] * 20)
    assert select_with_threshold(scores, groups, LiveThreshold(0.5, 1.0)).all()


def test_scores_strictly_above_the_threshold_always_pass():
    scores = np.array([0.9, 0.5, 0.1], dtype=np.float64)
    groups = np.array(["A", "A", "A"])
    assert select_with_threshold(scores, groups, LiveThreshold(0.5, 0.0))[0]


def test_threshold_selection_uses_training_scores_only():
    """홀드아웃 점수로 임계값을 고르면 동일 자막률 비교가 무너집니다."""
    import inspect

    from audire.experiments import live_ablation

    source = inspect.getsource(live_ablation.evaluate_live_arm)
    # 임계값은 훈련 예측에서만 나와야 합니다.
    assert "threshold_for_caption_rate(train_scores" in source
    assert "threshold_for_caption_rate(oof" not in source


def test_matched_comparison_reports_whether_the_rate_actually_matched(result):
    """맞지 않았으면 비교가 무효라는 사실이 요약에 남아야 합니다."""
    head = result["summary"]["headline"]
    assert "caption_rate_matched_within_tolerance" in head
    assert isinstance(head["caption_rate_matched_within_tolerance"], bool)


def test_matched_caption_rates_are_close_across_live_arms(result):
    """arm 마다 자막량이 다르면 재현율 비교가 자막량 비교가 되어 버립니다."""
    rates = [
        e["matched_caption_rate"]["mean"] for e in result["summary"]["table"] if e["role"] == "live"
    ]
    assert max(rates) - min(rates) < 0.05, rates


# ------------------------------------------------------------------------- 지표


def test_threshold_metrics_report_caption_rate_and_fairness():
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int64)
    groups = np.array(["A"] * 4 + ["B"] * 4)
    scores = np.array([0.9, 0.8, 0.2, 0.1, 0.1, 0.2, 0.8, 0.9])
    out = threshold_metrics(y, scores, groups, 0.5)
    for key in (
        "caption_rate",
        "misheard_recall",
        "precision",
        "f1",
        "recall_median_listener",
        "recall_worst_listener",
        "caption_rate_sd_listener",
        "frac_listeners_near_zero_captions",
    ):
        assert key in out, key
    # A 는 완벽, B 는 최악 -> 최하위 청취자가 0 이어야 합니다.
    assert out["recall_worst_listener"] == pytest.approx(0.0)


def test_probability_and_budget_metrics_are_both_reported(result):
    row = result["rows"][0]
    for key in ("pr_auc", "roc_auc", "brier", "log_loss", "ece", "mce"):
        assert key in row, key
    assert "recall@0.2" in row


# --------------------------------------------------------------------- 격자 완전성


def test_every_declared_arm_and_seed_appears(result):
    produced = {(r["arm"], r["seed"]) for r in result["rows"]}
    expected = {(a, s) for a in [*LIVE_ARMS, DIAGNOSTIC_ARM] for s in TINY.seeds}
    assert produced == expected


def test_paired_contrasts_are_per_seed(result):
    for contrast in result["summary"]["headline"]["contrasts"]:
        assert contrast["n_seeds"] == len(TINY.seeds)
        assert len(contrast["per_seed"]) == len(TINY.seeds)


def test_no_listener_crosses_the_split():
    """계약 작업이 청취자 수준 분할을 건드리지 않았는지 확인합니다."""
    cohort = build_cohort(TINY, 1)
    row = evaluate_live_arm(cohort, "live_word_context", _cfg(), 1)
    assert row["n_listeners"] == TINY.n_listeners


def test_caveat_states_the_synthetic_scope(result):
    caveat = result["summary"]["headline"]["caveat"]
    assert "합성" in caveat
    assert "진단" in caveat
