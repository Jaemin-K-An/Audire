"""E22 — 모델 비교 하니스와 교정 기록.

여기서 고정하는 것은 결과가 아니라 **결과를 만드는 규칙**입니다. 유리한 시드를 고를 수
없어야 하고, 지는 후보가 표에서 사라지지 않아야 하며, "isotonic 을 평가했다" 는 주장이
실제로 isotonic 이 돌았다는 뜻이어야 합니다.
"""

from __future__ import annotations

import numpy as np
import pytest

from audire.eval.ablation import evaluate_arm
from audire.experiments.model_comparison import (
    ModelComparisonConfig,
    budget_metrics,
    run_model_comparison,
)
from audire.risk import CalibratedRiskModel, FeatureMatrix, LogisticRiskModel
from audire.sim import SimulationConfig, build_cohort

TINY = SimulationConfig(
    name="e22-tiny",
    n_listeners=14,
    n_calibration_trials=35,
    n_word_trials=40,
    seeds=[3, 4],
)


def _cfg(**overrides) -> ModelComparisonConfig:
    base = {
        "name": "e22-test",
        "simulation": TINY,
        "models": ["logistic", "residual"],
        "calibrations": ["none", "platt"],
        "n_splits": 4,
        "n_bootstrap": 0,
        "budgets": [0.1, 0.2],
        "primary_budget": 0.2,
    }
    return ModelComparisonConfig(**{**base, **overrides})


@pytest.fixture(scope="module")
def result():
    return run_model_comparison(_cfg())


# ------------------------------------------------------------------------------ 설정 검증


def test_config_rejects_an_unknown_model():
    with pytest.raises(ValueError, match="알 수 없는 모델"):
        _cfg(models=["logistic", "neural_magic"])


def test_config_rejects_an_unknown_arm():
    with pytest.raises(ValueError, match="알 수 없는 arm"):
        _cfg(arm="not_an_arm")


def test_primary_budget_must_be_one_of_the_reported_budgets():
    """주 판정 지점이 표에 없으면 headline 을 사후에 고를 여지가 생깁니다."""
    with pytest.raises(ValueError, match="주 판정 예산"):
        _cfg(budgets=[0.1, 0.3], primary_budget=0.2)


def test_default_models_include_the_reference_baseline():
    assert "logistic" in ModelComparisonConfig(simulation=TINY).models


# ----------------------------------------------------------------------- 예산 지표의 정의


def test_budget_metrics_respect_the_per_listener_budget():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)
    groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    scores = np.array([0.9, 0.1, 0.8, 0.2, 0.9, 0.1, 0.8, 0.2])

    out = budget_metrics(y, groups, scores, (0.5,), seed=0)["0.5"]
    # 청취자마다 4개 중 2개 = 전체의 50%.
    assert out["achieved_ratio"] == pytest.approx(0.5)
    # 완벽한 점수이므로 두 청취자 모두 오청 2개를 모두 잡습니다.
    assert out["recall"] == pytest.approx(1.0)
    assert out["recall_worst_listener"] == pytest.approx(1.0)


def test_budget_metrics_expose_the_worst_listener_not_only_the_aggregate():
    """총합 재현율은 소수의 청취자를 방치하는 정책을 가려줍니다."""
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int64)
    groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    # A 는 완벽하게, B 는 최악으로 순위가 매겨진 점수.
    scores = np.array([0.9, 0.8, 0.2, 0.1, 0.1, 0.2, 0.8, 0.9])

    out = budget_metrics(y, groups, scores, (0.5,), seed=0)["0.5"]
    assert out["recall"] == pytest.approx(0.5)
    assert out["recall_worst_listener"] == pytest.approx(0.0)
    assert out["recall_worst_listener"] < out["recall"]


# ------------------------------------------------------------------ 교정 기록 (요청 vs 실제)


def test_calibration_records_requested_and_effective_method():
    described = CalibratedRiskModel(base=LogisticRiskModel(), method="platt").describe()
    assert described["requested_method"] == "platt"
    assert described["effective_method"] == "platt"
    assert described["fell_back"] is False


def test_fallback_preserves_the_requested_method():
    """회귀 테스트.

    이전 구현은 폴백할 때 ``self.method`` 를 ``"none"`` 으로 덮어썼습니다. 그러면 요청이
    무엇이었는지에 대한 유일한 기록이 사라져, **의도적으로 교정을 쓰지 않은 실행**과
    **교정이 조용히 실패한 실행**을 산출물만 보고 구분할 수 없었습니다. 게다가 ``name`` 은
    여전히 ``"...+platt"`` 라서 기록이 스스로 모순됐습니다.
    """
    rng = np.random.default_rng(0)
    groups = np.array([f"L{i // 10}" for i in range(40)])
    # 교정 슬라이스로 뽑히는 청취자의 라벨이 한 종류만 되도록 배치합니다.
    y = np.array([1] * 20 + [0] * 20, dtype=np.int64)
    matrix = FeatureMatrix(
        X=rng.normal(size=(40, 3)),
        feature_names=("a", "b", "c"),
        groups=groups,
        y=y,
        meta={},
    )
    model = CalibratedRiskModel(base=LogisticRiskModel(), method="platt", seed=0).fit(matrix)
    described = model.describe()

    assert described["requested_method"] == "platt", "요청한 방법이 보존되어야 합니다"
    assert described["effective_method"] == "none"
    assert described["fell_back"] is True
    assert described["fallback_reason"]
    # 이름과 기록이 서로 모순되지 않아야 합니다.
    assert described["name"] == "logistic+platt"


def test_refitting_clears_a_previous_fallback():
    """모델 인스턴스를 재사용해도 이전 폴드의 폴백 상태가 남으면 안 됩니다."""
    rng = np.random.default_rng(1)
    groups = np.array([f"L{i // 10}" for i in range(40)])
    bad = FeatureMatrix(
        X=rng.normal(size=(40, 3)),
        feature_names=("a", "b", "c"),
        groups=groups,
        y=np.array([1] * 20 + [0] * 20, dtype=np.int64),
        meta={},
    )
    good = FeatureMatrix(
        X=rng.normal(size=(40, 3)),
        feature_names=("a", "b", "c"),
        groups=groups,
        y=np.array([1, 0] * 20, dtype=np.int64),
        meta={},
    )
    model = CalibratedRiskModel(base=LogisticRiskModel(), method="platt", seed=0)
    assert model.fit(bad).describe()["fell_back"] is True
    assert model.fit(good).describe()["fell_back"] is False
    assert model.fit(good).describe()["fallback_reason"] is None


def test_calibration_is_tracked_per_fold_not_once_at_the_end():
    """폴드마다 폴백 여부가 다를 수 있으므로 마지막 폴드 스냅숏으로는 부족합니다."""
    cohort = build_cohort(TINY, 3)
    result = evaluate_arm(
        cohort,
        "clinical_plus_confusion",
        "logistic",
        seed=0,
        n_splits=4,
        calibration="platt",
        n_bootstrap=0,
    )
    per_fold = result.model_description["calibration_per_fold"]
    assert len(per_fold) == 4
    assert [e["fold"] for e in per_fold] == [0, 1, 2, 3]
    for entry in per_fold:
        assert entry["requested_method"] == "platt"
        assert entry["n_calibration_listeners"] >= 1
    assert result.model_description["n_folds_fell_back"] == sum(
        1 for e in per_fold if e["fell_back"]
    )


def test_uncalibrated_runs_carry_no_calibration_log():
    cohort = build_cohort(TINY, 3)
    result = evaluate_arm(
        cohort, "clinical_plus_confusion", "logistic", seed=0, n_splits=4, n_bootstrap=0
    )
    assert "calibration_per_fold" not in result.model_description


# --------------------------------------------------------------------- 집계 규칙 (선택 편향)


def test_every_model_and_calibration_combination_reaches_the_table(result):
    """지는 후보를 표에서 빼는 것은 선택 편향입니다. 구조적으로 불가능해야 합니다."""
    table = result["summary"]["table"]
    assert len(table) == 2 * 2  # 모델 2 x 교정 2
    assert {r["model"] for r in table} == {"logistic", "residual"}
    assert {r["calibration_requested"] for r in table} == {"none", "platt"}


def test_every_seed_is_aggregated(result):
    for row in result["summary"]["table"]:
        assert row["n_seeds"] == len(TINY.seeds)
        assert row["pr_auc"]["n"] == len(TINY.seeds)


def test_a_candidate_qualifies_only_by_winning_on_every_seed(result):
    """한 시드에서만 유리한 결과는 결과가 아닙니다."""
    head = result["summary"]["headline"]
    table = result["summary"]["table"]
    qualified = [r for r in table if r["n_seeds_beating_word_length"] == r["n_seeds"]]
    assert head["n_candidates_beating_word_length_on_every_seed"] == len(qualified)
    if head["best_candidate"] is None:
        assert not qualified


def test_headline_states_a_negative_result_rather_than_omitting_it(result):
    """이긴 후보가 없으면 그 사실이 명시되어야 합니다."""
    head = result["summary"]["headline"]
    assert "n_candidates_beating_word_length_on_every_seed" in head
    assert "best_candidate" in head  # 없으면 None 으로 존재해야 합니다
    if head["best_candidate"] is None:
        assert head["best_beats_reference_logistic"] is None


def test_the_reference_logistic_row_is_always_reported(result):
    """참조 기저선이 표에서 빠지면 새 계열의 우위를 판정할 근거가 없어집니다."""
    assert result["summary"]["headline"]["reference_logistic"] is not None


def test_word_length_baseline_is_evaluated_on_the_same_rows(result):
    """개인화와 휴리스틱이 다른 행에서 평가되면 비교 자체가 무의미합니다."""
    for row in result["rows"]:
        assert set(row["budgets"]) == set(row["word_length_budgets"])
        for key in row["budgets"]:
            assert row["gain_over_word_length"][key] == pytest.approx(
                row["budgets"][key]["recall"] - row["word_length_budgets"][key]["recall"]
            )


@pytest.mark.research_models
def test_ranking_scores_are_flagged_as_non_probabilities():
    """Brier/ECE 를 교정되지 않은 순위 점수에 대해 그대로 읽으면 잘못된 결론이 납니다."""
    out = run_model_comparison(_cfg(models=["lambdamart"], calibrations=["none"]))
    assert out["summary"]["table"][0]["output_is_probability"] is False


def test_caveat_states_the_synthetic_scope(result):
    caveat = result["summary"]["headline"]["caveat"]
    assert "합성" in caveat and "임상" in caveat


def test_run_is_recorded_in_the_registry(result):
    from audire.experiments.registry import load_runs

    ids = {r["run_id"] for r in load_runs()}
    assert result["run_id"] in ids


def test_uncalibrated_rows_state_their_effective_method_explicitly(result):
    """빈칸은 "기록이 없다" 와 "교정이 없었다" 를 구분하지 못합니다."""
    for row in result["summary"]["table"]:
        assert row["effective_methods"], row["model"]
        if row["calibration_requested"] == "none":
            assert row["effective_methods"] == ["none"]


def test_reference_verdict_uses_paired_seeds_not_a_mean_comparison():
    """회귀 테스트.

    이전 구현은 평균끼리 비교해 판정했습니다. 시드 간 변동보다 훨씬 작은 차이도 "이겼다"
    로 보고되며, 실제로 E23 축소 A/B 에서 평균 차이 +0.0006 이 승리로 보고됐는데 같은
    데이터의 시드별 짝지은 차이는 10개 중 6개만 양수인 동전 던지기였습니다.
    """
    from audire.experiments.model_comparison import _paired_against_reference

    best = {"model": "residual", "calibration_requested": "none", "n_seeds": 4}
    ref = {"model": "logistic", "calibration_requested": "none", "n_seeds": 4}
    # 평균으로는 후보가 이기지만(+0.005), 짝지으면 4개 중 1개 시드에서만 우세합니다.
    recalls = {
        ("residual", 1): 0.30,
        ("logistic", 1): 0.28,
        ("residual", 2): 0.20,
        ("logistic", 2): 0.21,
        ("residual", 3): 0.20,
        ("logistic", 3): 0.21,
        ("residual", 4): 0.20,
        ("logistic", 4): 0.21,
    }
    rows = [
        {
            "model": m,
            "seed": s,
            "calibration_requested": "none",
            "budgets": {"0.2": {"recall": v}},
        }
        for (m, s), v in recalls.items()
    ]
    paired = _paired_against_reference(rows, best, ref, "0.2")
    assert paired["n_seeds_best_beats_reference"] == 1
    assert paired["beats_on_every_seed"] is False
    assert paired["paired_gain_sd"] > abs(paired["paired_gain_mean"]), (
        "시드 간 편차가 평균 차이보다 크면 승리로 보고해서는 안 됩니다"
    )


def test_best_candidate_being_the_reference_is_not_reported_as_a_win():
    from audire.experiments.model_comparison import _paired_against_reference

    ref = {"model": "logistic", "calibration_requested": "none", "n_seeds": 3}
    paired = _paired_against_reference([], ref, ref, "0.2")
    assert paired["best_is_the_reference"] is True
    assert paired["beats_on_every_seed"] is False
