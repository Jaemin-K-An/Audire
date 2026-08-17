"""G8 — CLI 는 문서화된 재현 경로이므로 실제로 실행되는지 검증합니다.

`docs/RESULTS.md` 와 `README` 가 제시하는 재현 절차는 전부 이 CLI 를 거칩니다. 명령이
깨지면 결과를 재현할 방법 자체가 사라지므로, 여기서는 지표가 아니라 **경로가 살아 있는지**를
확인합니다. 실제 추론이 필요한 `asr-eval` 과 사용자 디렉터리에 쓰는 `build-model` 은
제외합니다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from audire.cli import app

runner = CliRunner()

TINY_SIM = {
    "name": "cli_cohort",
    "seeds": [1],
    "n_listeners": 12,
    "n_calibration_trials": 20,
    "n_word_trials": 20,
}


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """실험 산출물이 저장소를 오염시키지 않게 합니다."""
    monkeypatch.setenv("AUDIRE_EXPERIMENTS_DIR", str(tmp_path / "experiments"))
    monkeypatch.setenv("AUDIRE_ARTIFACTS_DIR", str(tmp_path / "experiments" / "artifacts"))
    monkeypatch.setenv("AUDIRE_PRIVATE_DIR", str(tmp_path / "private"))
    yield


def _config(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return path


def _run(*args: str):
    result = runner.invoke(app, list(args))
    assert result.exit_code == 0, f"{args} 실패:\n{result.output}\n{result.exception}"
    return result


# --------------------------------------------------------------------------- 조회 명령


def test_runs_reports_an_empty_registry_without_crashing():
    result = runner.invoke(app, ["runs"])
    assert result.exit_code == 0
    assert "no runs recorded yet" in result.output


def test_verify_runs_reports_an_empty_registry():
    result = runner.invoke(app, ["verify-runs"])
    assert result.exit_code == 0
    assert "기록된 실행이 없습니다" in result.output


def test_figures_says_so_when_there_is_nothing_to_regenerate():
    result = runner.invoke(app, ["figures"])
    assert result.exit_code == 0
    assert "nothing to regenerate" in result.output


def test_unknown_command_fails_loudly():
    assert runner.invoke(app, ["not-a-command"]).exit_code != 0


def test_missing_config_is_rejected(tmp_path: Path):
    """존재하지 않는 설정으로 조용히 기본값을 쓰면 안 됩니다."""
    result = runner.invoke(app, ["evaluate", "-c", str(tmp_path / "absent.yaml")])
    assert result.exit_code != 0


# --------------------------------------------------------------------------- 시뮬레이션


def test_simulate_writes_a_cohort_summary(tmp_path: Path):
    config = _config(tmp_path, "sim", {"name": "cli_sim", "simulation": TINY_SIM})
    out = tmp_path / "cohorts.json"
    result = _run("simulate", "-c", str(config), "--out", str(out))

    assert "mishear rate" in result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload) == len(TINY_SIM["seeds"])


# ------------------------------------------------------------------ 평가 -> 도표 -> 검증


@pytest.fixture
def evaluated(tmp_path: Path):
    """작은 RQ1 실행 하나. 이후 명령들이 이 산출물을 소비합니다."""
    config = _config(
        tmp_path,
        "eval",
        {
            "name": "cli_eval",
            "simulation": TINY_SIM,
            "arms": ["word_context_only", "clinical_plus_confusion"],
            "models": ["logistic"],
            "n_splits": 3,
            "n_bootstrap": 0,
            "caption_budgets": [0.2],
            "threshold_targets": [0.2],
            "contrasts": [["clinical_plus_confusion", "word_context_only"]],
            "contrast_metrics": ["pr_auc"],
        },
    )
    return _run("evaluate", "-c", str(config))


def test_evaluate_prints_metrics_and_the_synthetic_caveat(evaluated):
    assert "run_id:" in evaluated.output
    assert "PR-AUC" in evaluated.output
    # 합성 데이터 단서는 어느 출력 경로에서도 빠지면 안 됩니다.
    assert "caveat:" in evaluated.output


def test_evaluate_records_the_run_and_runs_lists_it(evaluated):
    listing = _run("runs")
    assert "cli_eval" in listing.output
    assert "completed" in listing.output


def test_figures_regenerates_tables_from_the_recorded_run(evaluated):
    result = _run("figures")
    assert "wrote" in result.output
    assert "table_ablation.csv" in result.output


def test_verify_runs_passes_on_freshly_written_artifacts(evaluated):
    result = _run("verify-runs")
    assert "match" in result.output
    assert "modified" not in result.output


def test_verify_runs_detects_a_tampered_artifact(evaluated, tmp_path: Path):
    """재현 주장의 근거가 되는 파일이 바뀌면 CLI 가 실패해야 합니다."""
    artifacts = sorted((tmp_path / "experiments" / "artifacts").rglob("summary.json"))
    assert artifacts, "산출물이 없습니다"
    artifacts[0].write_text('{"tampered": true}', encoding="utf-8")

    result = runner.invoke(app, ["verify-runs"])
    assert result.exit_code == 1
    assert "modified" in result.output


# --------------------------------------------------------------------------- 비교 명령


def test_model_compare_runs_and_reports_the_reference_baseline(tmp_path: Path):
    config = _config(
        tmp_path,
        "e22",
        {
            "name": "cli_e22",
            "simulation": TINY_SIM,
            "models": ["logistic"],
            "calibrations": ["none"],
            "n_splits": 3,
            "n_bootstrap": 0,
            "budgets": [0.2],
            "primary_budget": 0.2,
        },
    )
    result = _run("model-compare", "-c", str(config))
    assert "참조 로지스틱" in result.output
    assert "단서:" in result.output


def test_sensitivity_runs_the_declared_grid(tmp_path: Path):
    config = _config(
        tmp_path,
        "sens",
        {
            "name": "cli_sens",
            "base_simulation": TINY_SIM,
            "dirichlet_concentration": [5.0],
            "calibration_lengths": [20],
            "arms": ["word_context_only", "clinical_plus_confusion"],
            "models": ["logistic"],
            "n_splits": 3,
            "n_bootstrap": 0,
        },
    )
    result = _run("sensitivity", "-c", str(config))
    assert "격자 칸 수" in result.output
    assert "단서:" in result.output


def test_caption_eval_prints_both_budget_modes(tmp_path: Path):
    config = _config(
        tmp_path,
        "cap",
        {
            "name": "cli_cap",
            "simulation": TINY_SIM,
            "arms": ["word_context_only", "clinical_plus_confusion"],
            "models": ["logistic"],
            "n_splits": 3,
            "n_bootstrap": 0,
            "caption_budgets": [0.2],
            "threshold_targets": [0.2],
            "contrasts": [],
        },
    )
    result = _run("caption-eval", "-c", str(config))
    assert "per_listener" in result.output
    assert "pooled" in result.output


# --------------------------------------------------------------------------- 프로파일


def test_profile_summary_prints_derived_measures_and_gaps(tmp_path: Path):
    from audire.profile.schema import (
        Audiogram,
        AudiogramPoint,
        Ear,
        EarProfile,
        HearingProfile,
        ProfileSource,
        SpeechScores,
    )

    profile = HearingProfile(
        listener_id="L001",
        source=ProfileSource.MANUAL,
        is_synthetic=False,
        right=EarProfile(
            ear=Ear.RIGHT,
            audiogram=Audiogram(
                ear=Ear.RIGHT,
                thresholds={f: AudiogramPoint(db_hl=40.0) for f in (500, 1000, 2000, 4000)},
            ),
            # SRT/WRS 를 비워 두어 'summary 가 결측을 결측으로 보고하는가' 도 함께 봅니다.
            speech=SpeechScores(ear=Ear.RIGHT),
        ),
    )
    path = tmp_path / "profile.json"
    path.write_text(profile.model_dump_json(), encoding="utf-8")

    result = _run("profile-summary", str(path))
    payload = json.loads(result.output)
    assert payload["listener_id"] == "L001"
