"""Phase C — 민감도 하니스가 설정한 대로 실제로 도는지.

수정 전 결함 네 가지:

8.1 다중 예산 붕괴
  `by_strategy = {p.strategy: p for p in points}` 가 strategy × budget 을 strategy 로
  뭉갰다. `caption_budgets` 에 값이 여러 개면 **마지막 예산만 살아남고** 나머지 조건이
  조용히 사라졌다. 저장소 기본 설정이 예산 하나뿐이라 드러나지 않았을 뿐이다.

8.2 무시되는 n_bootstrap
  `SensitivityConfig.n_bootstrap` 이 선언돼 있는데 두 호출부가 모두 `n_bootstrap=0` 을
  하드코딩했다. 설정에 200 을 적어도 아무 일도 일어나지 않았다.

8.3 models 리스트의 거짓말
  `primary = cfg.models[0]` 로 **첫 모델만** 평가하면서 필드 이름은 복수형 `models` 라
  모든 모델 계열이 평가되는 것처럼 읽혔다.

8.4 검증 우회
  `model_copy(update=...)` 가 pydantic 검증을 건너뛴다. 스윕이 만들어 내는 설정이
  잘못돼도(예: 음수 디리클레 집중도) 그대로 실행됐다.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from audire.experiments.sensitivity import SensitivityConfig, run_sensitivity


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AUDIRE_EXPERIMENTS_DIR", str(tmp_path / "experiments"))
    monkeypatch.setenv("AUDIRE_ARTIFACTS_DIR", str(tmp_path / "experiments" / "artifacts"))
    yield


def _cfg(**overrides: object) -> SensitivityConfig:
    payload: dict = {
        "name": "sweep_test",
        "base_simulation": {
            "name": "c",
            "seeds": [7],
            "n_listeners": 20,
            "n_calibration_trials": 30,
            "n_word_trials": 40,
        },
        "dirichlet_concentration": [5.0],
        "calibration_lengths": [30],
        "arms": ["word_context_only", "clinical_plus_confusion"],
        "n_splits": 4,
        "n_bootstrap": 0,
        "caption_budgets": [0.20],
    }
    payload.update(overrides)
    return SensitivityConfig.model_validate(payload)


# =========================================================== 8.1 다중 예산


def test_every_budget_condition_survives() -> None:
    """예산 3개를 주면 격자에 예산 3개가 모두 남아야 한다."""
    budgets = [0.10, 0.20, 0.30]
    summary = run_sensitivity(_cfg(caption_budgets=budgets))["summary"]

    seen = {round(float(g["budget"]), 4) for g in summary["grid"]}
    assert seen == {0.10, 0.20, 0.30}, f"예산 조건이 사라졌다: {sorted(seen)}"


def test_grid_rows_are_keyed_by_strategy_and_budget() -> None:
    budgets = [0.10, 0.20, 0.30]
    summary = run_sensitivity(_cfg(caption_budgets=budgets))["summary"]

    keys = [(g["arm"], round(float(g["budget"]), 4)) for g in summary["grid"]]
    assert len(keys) == len(set(keys)), f"(arm, budget) 조합이 중복됐다: {keys}"
    assert len(keys) == 2 * len(budgets)  # arm 2개 × 예산 3개


def test_recall_actually_differs_between_budgets() -> None:
    """예산이 커지면 재현율이 올라야 한다. 같다면 예산이 실제로 적용되지 않은 것이다."""
    summary = run_sensitivity(_cfg(caption_budgets=[0.10, 0.50]))["summary"]
    by_budget: dict[float, float] = {}
    for g in summary["grid"]:
        if g["arm"] == "clinical_plus_confusion":
            by_budget[round(float(g["budget"]), 4)] = g["misheard_recall_mean"]
    assert by_budget[0.50] > by_budget[0.10], by_budget


# =========================================================== 8.2 n_bootstrap


def test_n_bootstrap_is_actually_used() -> None:
    """설정에 적은 부트스트랩 수가 실제로 신뢰구간을 만들어야 한다."""
    summary = run_sensitivity(_cfg(n_bootstrap=25))["summary"]
    row = next(g for g in summary["grid"] if g["arm"] == "clinical_plus_confusion")
    assert row["recall_ci_lo"] is not None, "n_bootstrap 을 줬는데 신뢰구간이 없다"
    assert row["recall_ci_lo"] <= row["misheard_recall_mean"] <= row["recall_ci_hi"]


def test_zero_bootstrap_means_no_interval() -> None:
    summary = run_sensitivity(_cfg(n_bootstrap=0))["summary"]
    row = next(g for g in summary["grid"] if g["arm"] == "clinical_plus_confusion")
    assert row["recall_ci_lo"] is None


# =========================================================== 8.3 models


def test_every_configured_model_family_is_evaluated() -> None:
    """models 가 리스트라면 전부 평가하거나, 필드 이름이 단수여야 한다."""
    summary = run_sensitivity(
        _cfg(models=["logistic", "gradient_boosting"], caption_budgets=[0.20])
    )["summary"]
    assert {g["model"] for g in summary["grid"]} == {"logistic", "gradient_boosting"}


def test_single_model_still_works() -> None:
    summary = run_sensitivity(_cfg(models=["logistic"]))["summary"]
    assert {g["model"] for g in summary["grid"]} == {"logistic"}


# =========================================================== 8.4 설정 검증


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dirichlet_concentration", [0.0]),
        ("dirichlet_concentration", [-1.0]),
        ("calibration_lengths", [0]),
        ("calibration_lengths", [-5]),
        ("calibration_lengths", []),
        ("dirichlet_concentration", []),
        ("caption_budgets", []),
        ("caption_budgets", [1.5]),
        ("arms", []),
        ("models", []),
    ],
)
def test_invalid_sweep_config_is_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        _cfg(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("snr_conditions_db", []),
        ("speakers", []),
        ("n_listeners", 1),
        ("n_calibration_trials", 0),
        ("n_word_trials", 0),
        ("seeds", []),
    ],
)
def test_invalid_base_simulation_is_rejected(field: str, value: object) -> None:
    base = {
        "name": "c",
        "seeds": [7],
        "n_listeners": 20,
        "n_calibration_trials": 30,
        "n_word_trials": 40,
    }
    base[field] = value
    with pytest.raises(ValidationError):
        _cfg(base_simulation=base)


def test_generated_sweep_configs_are_revalidated() -> None:
    """스윕이 만들어 내는 설정도 검증을 통과해야 한다. model_copy 는 그것을 건너뛴다."""
    from audire.experiments.sensitivity import build_cell_simulation

    cfg = _cfg()
    ok = build_cell_simulation(cfg, concentration=5.0, n_calibration=30)
    assert ok.n_calibration_trials == 30
    assert ok.confusion.dirichlet_concentration == 5.0

    with pytest.raises(ValidationError):
        build_cell_simulation(cfg, concentration=-1.0, n_calibration=30)
    with pytest.raises(ValidationError):
        build_cell_simulation(cfg, concentration=5.0, n_calibration=0)


# =========================================================== 격자 완전성


def test_no_grid_cell_is_skipped() -> None:
    """체리피킹 방지: 선언된 격자의 모든 칸이 결과에 나타나야 한다."""
    cfg = _cfg(
        dirichlet_concentration=[5.0, 40.0],
        calibration_lengths=[30, 60],
        caption_budgets=[0.20],
    )
    summary = run_sensitivity(cfg)["summary"]
    combos = {
        (g["dirichlet_concentration"], g["n_calibration_trials"], g["arm"], g["model"])
        for g in summary["grid"]
    }
    assert len(combos) == 2 * 2 * len(cfg.arms) * len(cfg.models), sorted(combos)


def test_headline_reports_the_rq2_question_per_budget() -> None:
    summary = run_sensitivity(_cfg(caption_budgets=[0.10, 0.20]))["summary"]
    for g in summary["grid"]:
        assert "recall_over_word_length_mean" in g
        assert 0 <= g["n_seeds_beating_word_length"] <= g["n_seeds"]
