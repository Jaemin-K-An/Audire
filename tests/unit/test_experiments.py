"""실험 실행 기록, 설정 로딩, 민감도 스윕 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from audire.config.paths import repo_root
from audire.experiments.registry import (
    fail_run,
    finish_run,
    git_sha,
    load_runs,
    new_run,
    save_artifact,
)
from audire.experiments.runner import ExperimentConfig
from audire.experiments.sensitivity import SensitivityConfig, run_sensitivity

#: 실제 저장소의 설정 디렉터리. 아래 격리 픽스처가 환경 변수를 바꾸기 전에 확정한다.
REAL_CONFIGS = repo_root() / "experiments" / "configs"


@pytest.fixture(autouse=True)
def _isolated_experiments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """레지스트리와 아티팩트를 임시 디렉터리로 격리한다."""
    monkeypatch.setenv("AUDIRE_EXPERIMENTS_DIR", str(tmp_path / "experiments"))
    monkeypatch.setenv("AUDIRE_ARTIFACTS_DIR", str(tmp_path / "experiments" / "artifacts"))
    yield


# =========================================================== 실행 기록


def test_run_record_captures_full_provenance() -> None:
    rec = new_run("demo", {"a": 1}, [1, 2, 3], notes="테스트")
    assert rec.run_id.startswith("demo-")
    assert rec.seeds == [1, 2, 3]
    assert rec.config == {"a": 1}
    assert rec.lock_hash
    assert rec.python
    assert rec.platform
    assert rec.status == "running"


def test_run_id_embeds_the_short_commit_sha() -> None:
    """run_id의 SHA 접미사가 'unknown'이 아니어야 한다(과거 인자 구성 버그 회귀)."""
    short = git_sha(short=True)
    if short == "unknown":
        pytest.skip("git 저장소가 아님")
    assert new_run("demo", {}, [1]).run_id.endswith(short)
    assert 6 <= len(short) < len(git_sha())


def test_finished_and_failed_runs_are_both_recorded() -> None:
    finish_run(new_run("ok", {}, [1]), {"pr_auc": 0.7})
    fail_run(new_run("bad", {}, [2]), "폭발함")

    runs = {r["experiment"]: r for r in load_runs()}
    assert runs["ok"]["status"] == "completed"
    assert runs["ok"]["metrics"]["pr_auc"] == 0.7
    # 실패한 실행도 반드시 남아야 한다 — 레지스트리에 조용한 공백이 생기면 안 된다.
    assert runs["bad"]["status"] == "failed"
    assert runs["bad"]["error"] == "폭발함"
    assert runs["bad"]["finished_at_utc"]


def test_rerunning_the_same_run_id_replaces_rather_than_duplicates() -> None:
    rec = new_run("dup", {}, [1])
    finish_run(rec, {"v": 1})
    finish_run(rec, {"v": 2})
    matching = [r for r in load_runs() if r["run_id"] == rec.run_id]
    assert len(matching) == 1
    assert matching[0]["metrics"]["v"] == 2


def test_artifacts_are_written_and_registered() -> None:
    import numpy as np

    rec = new_run("art", {}, [1])
    path = save_artifact(rec, "m.json", {"x": np.float64(0.5), "y": np.arange(3)})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {"x": 0.5, "y": [0, 1, 2]}
    assert len(rec.artifacts) == 1
    # 같은 이름을 다시 저장해도 목록이 중복되지 않는다.
    save_artifact(rec, "m.json", {"x": 1.0})
    assert len(rec.artifacts) == 1


def test_artifact_serialiser_rejects_unknown_types() -> None:
    with pytest.raises(TypeError, match="not JSON-serialisable"):
        save_artifact(new_run("bad", {}, [1]), "m.json", {"x": object()})


def test_empty_registry_reads_as_an_empty_list() -> None:
    assert load_runs() == []


# =========================================================== 설정


def test_shipped_experiment_configs_parse() -> None:
    """저장소에 담긴 설정이 실제로 로드되어야 한다."""
    for name in ("rq1_main.yaml", "smoke.yaml"):
        cfg = ExperimentConfig.load(REAL_CONFIGS / name)
        assert cfg.simulation.seeds
        assert cfg.arms and cfg.models
        assert cfg.n_bootstrap >= 0


def test_main_config_declares_every_required_comparison() -> None:
    """연구 계획이 요구하는 모델 계열과 arm이 모두 사전등록되어야 한다."""
    cfg = ExperimentConfig.load(REAL_CONFIGS / "rq1_main.yaml")
    assert {"pta_only", "clinical", "confusion_only", "clinical_plus_confusion"} <= set(cfg.arms)
    assert "word_context_only" in cfg.arms  # 비개인화 바닥
    assert "phoneme_independence" in cfg.models  # 결정론적 비교
    assert "gradient_boosting" in cfg.models  # 비선형 비교
    assert "logistic" in cfg.models
    assert len(cfg.simulation.seeds) >= 5  # 단일 시드 보고 금지
    assert ["clinical_plus_confusion", "clinical"] in [list(c) for c in cfg.contrasts]


def test_sensitivity_config_parses_and_counts_its_grid() -> None:
    cfg = SensitivityConfig.load(REAL_CONFIGS / "sensitivity.yaml")
    expected = (
        len(cfg.dirichlet_concentration)
        * len(cfg.calibration_lengths)
        * len(cfg.base_simulation.seeds)
    )
    assert cfg.n_cells == expected > 1
    assert "clinical_plus_confusion" in cfg.arms
    assert "word_context_only" in cfg.arms  # 비교 기준이 반드시 포함되어야 한다


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        yaml.safe_dump({"name": "x", "simulation": {"name": "s"}, "오타": 1}), encoding="utf-8"
    )
    with pytest.raises(Exception, match=r"[Ee]xtra"):
        ExperimentConfig.load(bad)


# =========================================================== 민감도 스윕


@pytest.fixture
def sweep_config() -> SensitivityConfig:
    return SensitivityConfig.model_validate(
        {
            "name": "sweep_test",
            "base_simulation": {
                "name": "sweep_cohort",
                "seeds": [7],
                "n_listeners": 20,
                "n_calibration_trials": 40,
                "n_word_trials": 40,
            },
            "dirichlet_concentration": [5.0, 80.0],
            "calibration_lengths": [40, 160],
            "arms": ["word_context_only", "clinical", "clinical_plus_confusion"],
            "models": ["logistic"],
            "n_splits": 4,
            "n_bootstrap": 0,
            "caption_budgets": [0.20],
        }
    )


@pytest.mark.slow
def test_sweep_visits_every_cell(sweep_config: SensitivityConfig) -> None:
    """어떤 격자 칸도 건너뛰지 않아야 한다."""
    summary = run_sensitivity(sweep_config)["summary"]
    combos = {
        (g["dirichlet_concentration"], g["n_calibration_trials"], g["arm"]) for g in summary["grid"]
    }
    assert len(combos) == 2 * 2 * 3
    assert summary["n_cells"] == 2 * 2 * 1
    assert summary["is_synthetic"] is True
    assert "사람 청취자에 대한 근거가 아닙니다" in summary["headline"]["caveat"]


@pytest.mark.slow
def test_arms_without_the_confusion_block_are_invariant_to_calibration_length(
    sweep_config: SensitivityConfig,
) -> None:
    """내부 일관성 검사: 교정 길이는 C_u를 쓰는 arm에만 영향을 주어야 한다.

    청취자별 난수 스트림이 분리되어 있으므로, 교정 시행 수를 바꿔도 단어 시행은
    바뀌지 않는다. 따라서 혼동 특징을 쓰지 않는 arm의 지표는 정확히 같아야 한다.
    이것이 깨지면 스윕이 두 축을 혼동하고 있다는 뜻이다.
    """
    grid = run_sensitivity(sweep_config)["summary"]["grid"]
    by = {(g["arm"], g["dirichlet_concentration"], g["n_calibration_trials"]): g for g in grid}

    for arm in ("word_context_only", "clinical"):
        for conc in sweep_config.dirichlet_concentration:
            short = by[(arm, conc, 40)]["pr_auc_mean"]
            long_ = by[(arm, conc, 160)]["pr_auc_mean"]
            assert short == pytest.approx(long_), (arm, conc)

    # 반대로 혼동 arm은 실제로 반응해야 한다.
    changed = [
        by[("clinical_plus_confusion", c, 40)]["pr_auc_mean"]
        != by[("clinical_plus_confusion", c, 160)]["pr_auc_mean"]
        for c in sweep_config.dirichlet_concentration
    ]
    assert any(changed), "교정 길이가 혼동 arm에 아무 영향도 주지 않았다"


@pytest.mark.slow
def test_sweep_reports_whether_personalization_beats_the_heuristic(
    sweep_config: SensitivityConfig,
) -> None:
    """RQ2의 핵심 질문이 격자 칸마다 기록되어야 한다."""
    result = run_sensitivity(sweep_config)
    head = result["summary"]["headline"]
    assert (
        0
        <= head["n_cells_where_personalization_always_beats_word_length"]
        <= head["n_combined_cells"]
    )
    for g in result["summary"]["grid"]:
        assert 0 <= g["n_seeds_beating_word_length"] <= g["n_seeds"]
        assert "recall_over_word_length_mean" in g
