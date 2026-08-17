"""P0.4 — 실험 실행 생명주기가 구조적으로 보장되는지.

문서는 "실패한 실행도 기록된다"고 말합니다. 그것이 참이려면 실행이 도중에 예외로
죽어도 레지스트리에 항목이 **남아 있어야** 합니다.

수정 전 동작: `new_run()`은 RunRecord 객체를 만들 뿐 레지스트리에 쓰지 않았고,
`append_run`은 `finish_run`/`fail_run`에서만 호출됐습니다. 주 실행기에는 try/except가
없었으므로 도중에 실패한 실행은 **흔적 없이 사라졌습니다.** 이는 "failed로 기록된다"보다
나쁩니다. 레지스트리를 보는 사람이 그 실행이 있었다는 사실조차 알 수 없기 때문입니다.

여기서 강제하는 불변식:

1. 실행이 시작되면 즉시 `running` 상태로 레지스트리에 나타난다.
2. 예외가 나면 항목이 남고 `failed`가 되며 종료 시각과 오류 맥락을 갖는다.
3. 예외는 삼켜지지 않고 호출자에게 전파된다.
4. 정상 종료하면 `completed`가 된다.
5. KeyboardInterrupt 같은 BaseException 에서도 기록이 남는다.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from audire.experiments.registry import (
    RegistryCollision,
    append_run,
    finish_run,
    load_runs,
    new_run,
    registry_path,
    save_artifact,
    tracked_run,
    verify_artifacts,
)


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AUDIRE_EXPERIMENTS_DIR", str(tmp_path / "experiments"))
    monkeypatch.setenv("AUDIRE_ARTIFACTS_DIR", str(tmp_path / "experiments" / "artifacts"))
    yield


def _only_run() -> dict:
    runs = load_runs()
    assert len(runs) == 1, f"정확히 한 건이어야 하는데 {len(runs)}건"
    return runs[0]


# =========================================================== running 상태


def test_run_appears_as_running_before_it_finishes() -> None:
    """실행 중에도 레지스트리에서 보여야 한다. 그래야 중단된 실행을 발견할 수 있다."""
    with tracked_run("demo", {"a": 1}, [1, 2]) as rec:
        during = _only_run()
        assert during["run_id"] == rec.run_id
        assert during["status"] == "running"
        assert during["finished_at_utc"] is None
        assert during["seeds"] == [1, 2]


# =========================================================== 실패 경로


def test_exception_leaves_a_failed_record_and_propagates() -> None:
    class Boom(RuntimeError):
        pass

    with pytest.raises(Boom, match="시뮬레이터 폭발"), tracked_run("bad", {}, [7]):
        raise Boom("시뮬레이터 폭발")

    rec = _only_run()
    assert rec["status"] == "failed"
    assert rec["finished_at_utc"], "실패 시각이 없다"
    assert "Boom" in rec["error"]
    assert "시뮬레이터 폭발" in rec["error"]


def test_failure_stores_actionable_context() -> None:
    """오류 문자열만으로는 어디서 터졌는지 알 수 없다. 트레이스백이 남아야 한다."""

    def inner() -> None:
        raise ValueError("음소 목록 불일치")

    with pytest.raises(ValueError, match="음소 목록"), tracked_run("bad", {}, [7]):
        inner()

    rec = _only_run()
    assert rec["error_traceback"], "트레이스백이 비어 있다"
    assert "inner" in rec["error_traceback"]
    assert "음소 목록 불일치" in rec["error_traceback"]


def test_failed_run_keeps_its_config_and_provenance() -> None:
    """실패한 실행의 설정을 잃으면 재현 시도조차 할 수 없다."""
    cfg = {"n_listeners": 80, "seeds": [1, 2, 3]}
    with pytest.raises(RuntimeError), tracked_run("bad", cfg, [1, 2, 3], notes="메모"):
        raise RuntimeError("중단")

    rec = _only_run()
    assert rec["config"] == cfg
    assert rec["notes"] == "메모"
    assert rec["lock_hash"]
    assert rec["python"]


def test_keyboard_interrupt_is_also_recorded() -> None:
    """사용자가 Ctrl-C 로 끊은 장시간 실행도 흔적을 남겨야 한다."""
    with pytest.raises(KeyboardInterrupt), tracked_run("interrupted", {}, [1]):
        raise KeyboardInterrupt

    rec = _only_run()
    assert rec["status"] == "failed"
    assert "KeyboardInterrupt" in rec["error"]


# =========================================================== 성공 경로


def test_successful_run_becomes_completed() -> None:
    with tracked_run("ok", {}, [1]) as rec:
        rec.metrics = {"pr_auc": 0.71}

    done = _only_run()
    assert done["status"] == "completed"
    assert done["finished_at_utc"]
    assert done["metrics"]["pr_auc"] == 0.71
    assert done["error"] is None


def test_explicit_finish_inside_the_block_is_respected() -> None:
    """실행기가 이미 finish_run 을 호출했다면 컨텍스트가 덮어쓰지 않는다."""
    from audire.experiments.registry import finish_run

    with tracked_run("ok", {}, [1]) as rec:
        finish_run(rec, {"custom": 1.0})

    done = _only_run()
    assert done["status"] == "completed"
    assert done["metrics"] == {"custom": 1.0}


def test_no_duplicate_records_for_one_run() -> None:
    """running → completed 는 항목을 늘리지 않고 같은 run_id 를 갱신해야 한다."""
    with tracked_run("ok", {}, [1]) as rec:
        pass
    runs = load_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == rec.run_id


# =========================================================== 실행기 통합


def test_experiment_runner_records_a_failed_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """주 실행기가 도중에 죽어도 레지스트리에 failed 로 남아야 한다."""
    from audire.experiments import runner
    from audire.experiments.runner import ExperimentConfig, run_experiment

    cfg = ExperimentConfig.model_validate(
        {
            "name": "boom",
            "simulation": {
                "name": "c",
                "seeds": [1],
                "n_listeners": 20,
                "n_calibration_trials": 20,
                "n_word_trials": 20,
            },
            "arms": ["clinical"],
            "models": ["logistic"],
            "n_splits": 4,
            "n_bootstrap": 0,
        }
    )

    def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("평가 중 폭발")

    monkeypatch.setattr(runner, "evaluate_arm", explode)

    with pytest.raises(RuntimeError, match="평가 중 폭발"):
        run_experiment(cfg)

    rec = _only_run()
    assert rec["experiment"] == "boom"
    assert rec["status"] == "failed"
    assert "평가 중 폭발" in rec["error"]


def test_sensitivity_runner_records_a_failed_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from audire.experiments import sensitivity as sens_mod
    from audire.experiments.sensitivity import SensitivityConfig, run_sensitivity

    cfg = SensitivityConfig.model_validate(
        {
            "name": "boom_sweep",
            "base_simulation": {
                "name": "c",
                "seeds": [1],
                "n_listeners": 20,
                "n_calibration_trials": 20,
                "n_word_trials": 20,
            },
            "dirichlet_concentration": [5.0],
            "calibration_lengths": [20],
            "arms": ["word_context_only", "clinical_plus_confusion"],
            "models": ["logistic"],
            "n_splits": 4,
            "n_bootstrap": 0,
        }
    )

    def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("스윕 중 폭발")

    monkeypatch.setattr(sens_mod, "evaluate_arm", explode)

    with pytest.raises(RuntimeError, match="스윕 중 폭발"):
        run_sensitivity(cfg)

    rec = _only_run()
    assert rec["experiment"] == "boom_sweep"
    assert rec["status"] == "failed"


def test_successful_runner_still_completes() -> None:
    """실패 처리를 넣느라 정상 경로를 망가뜨리지 않았는지 확인한다."""
    from audire.experiments.runner import ExperimentConfig, run_experiment

    cfg = ExperimentConfig.model_validate(
        {
            "name": "ok_small",
            "simulation": {
                "name": "c",
                "seeds": [1],
                "n_listeners": 20,
                "n_calibration_trials": 20,
                "n_word_trials": 20,
            },
            "arms": ["word_context_only", "clinical"],
            "models": ["logistic"],
            "n_splits": 4,
            "n_bootstrap": 0,
            "caption_budgets": [0.2],
            "threshold_targets": [0.2],
            "contrasts": [["clinical", "word_context_only"]],
            "contrast_metrics": ["pr_auc"],
        }
    )
    run_experiment(cfg)

    rec = _only_run()
    assert rec["status"] == "completed"
    assert rec["artifacts"], "성공한 실행은 아티팩트를 남겨야 한다"


# ================================================ 레지스트리 견고성 (충돌·원자성·다이제스트)


def test_two_runs_started_in_the_same_second_get_distinct_ids():
    """회귀 테스트.

    run_id 는 `실험명-초단위시각-짧은sha` 였습니다. 같은 실험을 같은 초에 두 번 시작하면
    — 빠른 설정, 루프, 병렬 호출에서 쉽게 일어납니다 — id 가 같아지고 `append_run` 이
    run_id 로 교체하므로 **먼저 끝난 실행의 기록이 흔적 없이 사라졌습니다.**
    """
    a = new_run("exp", {"n": 1}, [1])
    b = new_run("exp", {"n": 2}, [2])
    assert a.run_id != b.run_id

    finish_run(a, {"r": "A"})
    finish_run(b, {"r": "B"})
    recorded = {r["run_id"]: r["metrics"] for r in load_runs()}
    assert len(recorded) == 2
    assert list(recorded.values()) != [{"r": "B"}], "먼저 끝난 실행이 사라졌습니다"


def test_a_colliding_run_id_is_refused_rather_than_overwriting():
    """id 가 충돌하면 조용히 덮어쓰지 않고 거부해야 합니다."""
    a = new_run("exp", {"n": 1}, [1])
    finish_run(a, {"r": "A"})

    intruder = new_run("exp", {"n": 2}, [2])
    intruder.run_id = a.run_id
    with pytest.raises(RegistryCollision, match="이미 다른 실행"):
        finish_run(intruder, {"r": "B"})

    survived = [r for r in load_runs() if r["run_id"] == a.run_id]
    assert len(survived) == 1
    assert survived[0]["metrics"] == {"r": "A"}


def test_a_runs_own_lifecycle_still_updates_in_place():
    """정상 경로는 막히면 안 됩니다: running -> completed 는 같은 기록을 갱신합니다."""
    record = new_run("exp", {"n": 1}, [1])
    append_run(record)
    assert [r["status"] for r in load_runs()] == ["running"]

    finish_run(record, {"r": "done"})
    rows = load_runs()
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"


def test_registry_survives_a_failed_write(monkeypatch):
    """레지스트리를 통째로 잃는 것은 모든 실행 기록을 잃는 것입니다."""
    finish_run(new_run("exp", {"n": 1}, [1]), {"r": "A"})
    before = registry_path().read_text(encoding="utf-8")

    def explode(src, dst):
        raise OSError("교체 도중 디스크 오류")

    # 이 패치만 되돌립니다. monkeypatch.undo() 는 autouse 격리 픽스처까지 되돌려
    # 이후 검사가 실제 저장소의 레지스트리를 보게 만듭니다.
    with monkeypatch.context() as patched:
        patched.setattr(os, "replace", explode)
        with pytest.raises(OSError, match="교체 도중"):
            finish_run(new_run("exp", {"n": 2}, [2]), {"r": "B"})

    assert registry_path().read_text(encoding="utf-8") == before
    assert len(load_runs()) == 1  # 여전히 읽을 수 있는 유효한 YAML


def test_no_temporary_files_are_left_in_the_registry_directory():
    finish_run(new_run("exp", {"n": 1}, [1]), {"r": "A"})
    leftovers = [p.name for p in registry_path().parent.iterdir() if ".tmp" in p.name]
    assert leftovers == []


def test_artifact_digest_is_recorded_and_detects_later_edits():
    """경로만 기록하면 '파일이 생겼다' 는 알지만 '무엇이 들었는지' 는 모릅니다."""
    record = new_run("exp", {"n": 1}, [1])
    path = save_artifact(record, "out.json", {"value": 1})
    finish_run(record, {})

    key = record.artifacts[0]
    assert verify_artifacts(record.run_id) == {key: "match"}

    path.write_text('{"value": 999}', encoding="utf-8")
    assert verify_artifacts(record.run_id) == {key: "modified"}

    path.unlink()
    assert verify_artifacts(record.run_id) == {key: "missing"}


def test_artifacts_without_a_recorded_digest_are_reported_as_unverifiable():
    """검증 불가와 검증 통과는 다릅니다. 다이제스트 도입 이전 기록이 조용히 통과하면 안 됩니다."""
    record = new_run("exp", {"n": 1}, [1])
    save_artifact(record, "out.json", {"value": 1})
    record.artifact_digests.clear()  # 예전 스키마로 기록된 실행을 흉내냅니다
    finish_run(record, {})
    assert verify_artifacts(record.run_id) == {record.artifacts[0]: "not_recorded"}


def test_verify_artifacts_rejects_an_unknown_run_id():
    with pytest.raises(KeyError, match="기록되지 않은"):
        verify_artifacts("no-such-run")


def test_verify_runs_cli_exits_nonzero_when_an_artifact_was_modified(monkeypatch):
    """무결성 확인은 종료 코드로 실패를 알려야 자동화에 쓸 수 있습니다."""
    from typer.testing import CliRunner

    from audire.cli import app

    record = new_run("exp", {"n": 1}, [1])
    path = save_artifact(record, "out.json", {"value": 1})
    finish_run(record, {})

    runner = CliRunner()
    ok = runner.invoke(app, ["verify-runs", "--run-id", record.run_id])
    assert ok.exit_code == 0, ok.output

    path.write_text('{"value": 999}', encoding="utf-8")
    tampered = runner.invoke(app, ["verify-runs", "--run-id", record.run_id])
    assert tampered.exit_code == 1
    assert "modified" in tampered.output
