"""E25 — 검증 격자 하니스의 불변식.

여기서 고정하는 것은 결과가 아니라 **격자가 정직한가**입니다. 선언한 조건이 조용히
사라지면 "모든 조건에서 확인했다" 는 진술이 거짓이 되고, 하위군 경계를 사후에 움직일 수
있으면 원하는 결과를 만들어낼 수 있습니다.
"""

from __future__ import annotations

import numpy as np
import pytest

from audire.experiments.validation_sweep import (
    ValidationSweepConfig,
    _band,
    listener_subgroups,
    run_validation_sweep,
    slice_metrics,
)
from audire.sim import SimulationConfig, build_cohort

TINY = SimulationConfig(
    name="e25-tiny",
    seeds=[1, 2],
    n_listeners=12,
    n_calibration_trials=20,
    n_word_trials=30,
)


def _cfg(**overrides) -> ValidationSweepConfig:
    base = {
        "name": "e25-test",
        "base_simulation": TINY,
        "calibration_lengths": [10, 40],
        "snr_conditions_db": [20.0, 0.0],
        "n_splits": 3,
        "n_bootstrap": 0,
        "budgets": [0.1, 0.2],
        "primary_budget": 0.2,
    }
    return ValidationSweepConfig(**{**base, **overrides})


@pytest.fixture(scope="module")
def result():
    return run_validation_sweep(_cfg())


# ------------------------------------------------------------------------------ 설정


def test_config_rejects_unknown_arm_and_model():
    with pytest.raises(ValueError, match="알 수 없는 arm"):
        _cfg(arm="not_an_arm")
    with pytest.raises(ValueError, match="알 수 없는 모델"):
        _cfg(model="neural_magic")


def test_primary_budget_must_be_reported():
    with pytest.raises(ValueError, match="주 판정 예산"):
        _cfg(budgets=[0.1, 0.3], primary_budget=0.2)


def test_declared_cell_count_is_calibration_times_seeds():
    cfg = _cfg()
    assert cfg.n_cells == len(cfg.calibration_lengths) * len(TINY.seeds)


# --------------------------------------------------------------------- 격자 완전성


def test_every_declared_cell_appears_in_the_artifacts(result):
    """빠진 칸은 치명적입니다. 조건이 사라지면 '전부 확인했다' 가 거짓이 됩니다."""
    cfg = _cfg()
    produced = {(r["calibration_length"], r["seed"]) for r in result["rows"]}
    declared = {(length, seed) for length in cfg.calibration_lengths for seed in TINY.seeds}
    assert produced == declared


def test_a_missing_cell_is_fatal(monkeypatch):
    """조용히 건너뛰는 대신 실패해야 합니다."""
    from audire.experiments import validation_sweep as mod

    real = mod.build_cohort
    calls = {"n": 0}

    def flaky(sim, seed):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("코호트 생성 실패")
        return real(sim, seed)

    monkeypatch.setattr(mod, "build_cohort", flaky)
    with pytest.raises(RuntimeError):
        run_validation_sweep(_cfg(name="e25-missing"))


def test_all_declared_slice_axes_are_present(result):
    axes = {r["slice_axis"] for r in result["rows"]}
    assert axes == {
        "overall",
        "snr_db",
        "speaker",
        "severity",
        "wrs_band",
        "evidence_band",
        "idiosyncrasy_band",
    }


def test_every_declared_snr_condition_appears(result):
    values = {r["slice_value"] for r in result["rows"] if r["slice_axis"] == "snr_db"}
    assert values == {"20", "0"}


def test_both_arms_are_evaluated(result):
    """비개인화 대조군이 빠지면 이득을 해석할 기준이 없습니다."""
    assert {r["arm"] for r in result["rows"]} == {
        "clinical_plus_confusion",
        "clinical",
    }


# --------------------------------------------------------------------- 하위군 정의


def test_subgroup_bands_are_fixed_before_seeing_results():
    """경계를 사후에 움직이면 원하는 하위군 결과를 만들 수 있습니다."""
    labels = ("low", "mid", "high")
    assert _band(10.0, (60.0, 80.0), labels) == "low"
    assert _band(70.0, (60.0, 80.0), labels) == "mid"
    assert _band(95.0, (60.0, 80.0), labels) == "high"
    # 경계값은 아래 구간에 속하지 않습니다 (엄격한 <).
    assert _band(60.0, (60.0, 80.0), labels) == "mid"


def test_missing_subgroup_value_is_labelled_not_dropped():
    """결측을 조용히 버리면 그 청취자들이 분석에서 사라집니다."""
    assert _band(None, (1.0,), ("a", "b")) == "unknown"


def test_every_listener_receives_a_label_on_every_axis():
    cohort = build_cohort(TINY, 1)
    subgroups = listener_subgroups(cohort)
    assert set(subgroups) == {r.listener_id for r in cohort.records}
    for axes in subgroups.values():
        assert set(axes) == {"severity", "wrs_band", "evidence_band", "idiosyncrasy_band"}
        assert all(isinstance(v, str) and v for v in axes.values())


# ----------------------------------------------------------------------- 지표 정의


def test_slice_metrics_reports_the_full_battery():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)
    groups = np.array(["A"] * 4 + ["B"] * 4)
    p = np.array([0.9, 0.1, 0.8, 0.2, 0.9, 0.1, 0.8, 0.2])
    words = ["가나다", "가", "가나", "가", "가나다", "가", "가나", "가"]

    out = slice_metrics(y, p, groups, words, (0.5,), ece_bins=5, seed=0)
    for key in ("pr_auc", "roc_auc", "brier", "ece", "mce", "prevalence", "n_listeners"):
        assert key in out, key
    for key in (
        "recall@0.5",
        "recall_median@0.5",
        "recall_worst@0.5",
        "recall_q25@0.5",
        "frac_listeners_near_zero@0.5",
        "word_length_recall@0.5",
        "gain@0.5",
    ):
        assert key in out, key


def test_gain_is_model_recall_minus_word_length_recall():
    y = np.array([1, 0, 1, 0], dtype=np.int64)
    groups = np.array(["A", "A", "B", "B"])
    p = np.array([0.9, 0.1, 0.9, 0.1])
    words = ["가", "나", "다", "라"]
    out = slice_metrics(y, p, groups, words, (0.5,), ece_bins=5, seed=0)
    assert out["gain@0.5"] == pytest.approx(out["recall@0.5"] - out["word_length_recall@0.5"])


def test_near_zero_listener_fraction_detects_starved_listeners():
    """접근성 지표: 총합 재현율이 가려버리는 '자막을 거의 못 받는 청취자'."""
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int64)
    groups = np.array(["A"] * 4 + ["B"] * 4)
    # A 는 완벽, B 는 최악.
    p = np.array([0.9, 0.8, 0.2, 0.1, 0.1, 0.2, 0.8, 0.9])
    out = slice_metrics(y, p, groups, ["가"] * 8, (0.5,), ece_bins=5, seed=0)
    assert out["frac_listeners_near_zero@0.5"] == pytest.approx(0.5)
    assert out["recall_worst@0.5"] == pytest.approx(0.0)


def test_ece_and_mce_are_reported_separately():
    """MCE 는 최악 구간의 괴리라 평균만으로는 보이지 않는 국소 미교정을 드러냅니다."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200).astype(np.int64)
    p = rng.random(200)
    out = slice_metrics(y, p, np.array(["A"] * 200), ["가"] * 200, (0.5,), ece_bins=10, seed=0)
    assert out["mce"] >= out["ece"]


# ------------------------------------------------------------------ 요약 규칙


def test_summary_reports_every_budget(result):
    cfg = _cfg()
    for entry in result["summary"]["tables"]["overall"]:
        for budget in cfg.budgets:
            assert f"recall@{budget:g}" in entry
            assert f"gain@{budget:g}" in entry


def test_budget_monotonicity_is_measured_not_assumed(result):
    """예산에 따른 이득이 단조라고 가정하지 않고 실제로 확인합니다."""
    mono = result["summary"]["headline"]["budget_monotonicity"]
    assert mono
    for row in mono:
        assert set(row["gains_by_budget"]) == {"0.1", "0.2"}
        assert isinstance(row["is_monotone_increasing"], bool)


def test_minimum_calibration_requires_winning_on_every_seed(result):
    """한 시드에서만 이긴 교정 길이는 자격이 없습니다."""
    head = result["summary"]["headline"]
    overall = result["summary"]["tables"]["overall"]
    qualifying = [
        e["calibration_length"] for e in overall if e["n_seeds_beating_word_length"] == e["n_seeds"]
    ]
    got = head["min_calibration_beating_word_length_on_every_seed"]
    assert got == (min(qualifying) if qualifying else None)


def test_caveat_states_the_synthetic_scope_and_pairing(result):
    caveat = result["summary"]["headline"]["caveat"]
    assert "합성" in caveat
    assert "짝지어져" in caveat


def test_run_is_recorded_in_the_registry(result):
    from audire.experiments.registry import load_runs

    assert result["run_id"] in {r["run_id"] for r in load_runs()}


def test_evidence_band_is_not_degenerate_across_listeners():
    """회귀 테스트.

    처음에는 ``coverage`` 로 증거량 하위군을 나눴습니다. 측정해 보니 coverage·n_trials·
    total_observations 는 청취자 간 분산이 **정확히 0** 이었습니다 — 교정 자극 목록이
    결정론적이고 균형 잡혀 있어 모든 청취자가 같은 자극을 받기 때문입니다. 그 값으로
    나누면 전원이 한 칸에 들어가 축이 아무것도 구분하지 못하는데, 표에는 한 줄이 찍혀
    분석을 한 것처럼 보입니다.
    """
    cohort = build_cohort(
        SimulationConfig(
            name="spread", seeds=[7], n_listeners=40, n_calibration_trials=100, n_word_trials=20
        ),
        7,
    )
    # 전제: coverage 는 실제로 분산이 없습니다.
    coverage = [
        float(np.mean(list(r.estimated_confusion.coverage.values()))) for r in cohort.records
    ]
    assert np.std(coverage) == pytest.approx(0.0, abs=1e-12), "이 테스트의 전제가 깨졌습니다"

    # 따라서 축은 실제로 청취자마다 다른 양을 써야 합니다.
    bands = {v["evidence_band"] for v in listener_subgroups(cohort).values()}
    assert len(bands) > 1, f"증거량 축이 한 칸으로 붕괴했습니다: {bands}"


def test_summary_separates_overall_gain_from_within_condition_gain(result):
    """조건이 섞이면 전체 이득에 '이 조건이 어렵다' 가 섞여 들어옵니다."""
    rows = result["summary"]["headline"]["overall_vs_within_condition_gain"]
    assert rows
    for row in rows:
        assert {"gain_overall", "gain_within_condition_mean", "gain_within_condition_min"} <= set(
            row
        )


def test_budget_monotonicity_is_also_reported_within_condition(result):
    """전체가 단조여도 조건 내부는 아닐 수 있으므로 둘 다 봅니다."""
    within = result["summary"]["headline"]["budget_monotonicity_within_snr"]
    assert within
    for row in within:
        assert "snr_db" in row
        assert isinstance(row["is_monotone_increasing"], bool)
