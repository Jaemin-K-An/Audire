"""생성 과정에 존재하는 단어 수준 신호의 상한 측정.

이 진단은 Phase C(Simulator V2)의 근거입니다. 상한을 모르고 추정기만 늘리면 "왜 이득이
없는가" 를 모델 탓으로 오해하게 됩니다.
"""

from __future__ import annotations

import pytest

from audire.sim import SimulationConfig, build_cohort
from audire.sim.diagnostics import outcome_signal_ceiling, perceived_form_influence

SMALL = SimulationConfig(
    name="diag", seeds=[101], n_listeners=30, n_calibration_trials=60, n_word_trials=120
)


@pytest.fixture(scope="module")
def cohort():
    return build_cohort(SMALL, 101)


def test_zero_segment_errors_is_deterministically_heard_correctly(cohort):
    """V1 의 구조: exact 는 n_segment_errors == 0 과 동치입니다."""
    out = outcome_signal_ceiling(cohort)
    assert out["p_misheard_given_zero_errors"] == pytest.approx(0.0, abs=1e-12)


def test_oracle_ceiling_far_exceeds_the_word_length_heuristic(cohort):
    """오류 개수를 알면 음절 수만 아는 것보다 훨씬 잘 맞힙니다.

    두 값의 간격이 곧 "모델이 추정해야 할 것" 의 크기입니다.
    """
    out = outcome_signal_ceiling(cohort)
    assert out["oracle_pr_auc"] > out["word_length_pr_auc"]
    assert out["oracle_pr_auc"] > out["prevalence"]


def test_ceiling_is_reproducible_for_a_fixed_seed():
    a = outcome_signal_ceiling(build_cohort(SMALL, 101))
    b = outcome_signal_ceiling(build_cohort(SMALL, 101))
    assert a["oracle_pr_auc"] == b["oracle_pr_auc"]


def test_v1_perceived_form_does_not_influence_the_outcome(cohort):
    """V1 의 핵심 한계.

    복구 확률이 지각형을 보지 않으므로, 같은 (오류 수, 음절 수) 안에서는 목표를 '닥' 으로
    듣든 '삭' 으로 듣든 결과 분포가 같습니다. 어휘가 커서 비교 가능한 지각형 쌍이 드물면
    측정 자체가 불가능하며, 그 사실도 함께 보고됩니다.
    """
    out = perceived_form_influence(cohort)
    if out["n_cells_with_comparable_forms"] == 0:
        pytest.skip("어휘가 커서 같은 지각형이 충분히 반복되지 않았습니다")
    # V1 에서는 지각형 간 오청률 차이가 순수 표집 잡음이어야 합니다.
    assert out["mean_rate_sd_across_forms"] < 0.35


def test_ceiling_rejects_an_empty_cohort(cohort):
    """빈 입력에서 조용히 NaN 을 돌려주면 상한이 없다는 사실이 가려집니다."""
    import dataclasses

    empty = dataclasses.replace(cohort, records=())
    with pytest.raises(ValueError, match="빈 코호트"):
        outcome_signal_ceiling(empty)
