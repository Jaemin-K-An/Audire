"""P0.2 / P0.3 — 데이터와 의존성 출처가 실제 바이트·실제 환경에 묶이는지.

AUDIRE 는 "모든 보고 수치가 그것을 만든 바이트와 코드까지 추적된다"고 주장합니다.
그 주장이 참이려면 두 가지가 필요합니다.

P0.2 — 매니페스트는 **취득 시점이 아니라 소비 시점**에 검증되어야 한다.
  수정 전: `zeroth.py` 와 `stimuli.py` 어디에도 `Manifest.verify` 호출이 없었다.
  매니페스트를 만든 뒤 파일을 바꿔도 실험은 그대로 돌았다. 즉
  `매니페스트 다이제스트 = 바이트 A` 인데 `모델 입력 = 바이트 B` 가 가능했다.
  또 레지스트리는 "디스크에 있는 모든 매니페스트"를 기록해, 그 실험이 쓰지도 않은
  데이터셋의 다이제스트가 출처인 것처럼 실렸다.

P0.3 — 의존성 지문은 **실제 설치된 환경**을 가리켜야 한다.
  수정 전: `lock_hash()` 는 `requirements.lock` 파일을 해시할 뿐이었다. 그 파일이
  실제 venv 와 아무 관련이 없어도 무방했다. 미션이 지목한 안티패턴 그대로다:
  `requirements.lock = 환경 X`, `실제 venv = 환경 Y`, `기록 = hash(X)`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from audire.data.manifest import (
    DataIntegrityError,
    Manifest,
    accessed_sources,
    require_verified,
    reset_accessed,
)
from audire.experiments.registry import (
    data_manifest_ids,
    environment_fingerprint,
    environment_matches_lock,
    load_runs,
    tracked_run,
)


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AUDIRE_MANIFESTS_DIR", str(tmp_path / "manifests"))
    monkeypatch.setenv("AUDIRE_EXPERIMENTS_DIR", str(tmp_path / "experiments"))
    monkeypatch.setenv("AUDIRE_ARTIFACTS_DIR", str(tmp_path / "experiments" / "artifacts"))
    reset_accessed()
    yield
    reset_accessed()


def _corpus(tmp_path: Path, name: str = "demo", body: bytes = b"alpha") -> Path:
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.bin").write_bytes(body)
    return root


def _manifest(tmp_path: Path, source_id: str = "demo", body: bytes = b"alpha") -> Manifest:
    root = _corpus(tmp_path, source_id, body)
    m = Manifest.build(source_id=source_id, license="CC-BY-4.0", local_path=root)
    m.save()
    return m


# =========================================================== P0.2 소비 시점 검증


def test_verified_source_loads(tmp_path: Path) -> None:
    _manifest(tmp_path)
    assert require_verified("demo").source_id == "demo"


def test_mutated_bytes_make_consumption_fail(tmp_path: Path) -> None:
    """가장 중요한 불변식: 매니페스트 생성 후 바이트가 바뀌면 사용이 실패해야 한다."""
    m = _manifest(tmp_path)
    # 길이는 같고 내용만 다르게 — 크기 검사만으로는 잡히지 않는다.
    (Path(m.local_path) / "data.bin").write_bytes(b"ALPHA")

    with pytest.raises(DataIntegrityError, match=r"checksum|무결성|integrity"):
        require_verified("demo")


def test_deleted_file_makes_consumption_fail(tmp_path: Path) -> None:
    m = _manifest(tmp_path)
    (Path(m.local_path) / "data.bin").unlink()
    with pytest.raises(DataIntegrityError):
        require_verified("demo")


def test_extra_file_makes_consumption_fail(tmp_path: Path) -> None:
    """기록되지 않은 파일이 생겼다면 그 데이터셋은 매니페스트가 말하는 것이 아니다."""
    m = _manifest(tmp_path)
    (Path(m.local_path) / "sneaked.bin").write_bytes(b"x")
    with pytest.raises(DataIntegrityError):
        require_verified("demo")


def test_missing_manifest_fails_with_an_actionable_message(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"make data|fetch_data"):
        require_verified("never_fetched")


def test_shallow_mode_is_available_but_deep_is_the_default(tmp_path: Path) -> None:
    """연구 실행은 깊은 검증이 기본이어야 한다. 얕은 검증은 명시적 선택이다."""
    m = _manifest(tmp_path)
    (Path(m.local_path) / "data.bin").write_bytes(b"ALPHA")  # 같은 길이

    require_verified("demo", deep=False)  # 크기만 보므로 통과
    with pytest.raises(DataIntegrityError):
        require_verified("demo")  # 기본값은 깊은 검증


# =========================================================== P0.2 실제 사용 추적


def test_only_sources_actually_used_are_recorded(tmp_path: Path) -> None:
    """쓰지도 않은 데이터셋의 다이제스트가 출처로 실려서는 안 된다."""
    _manifest(tmp_path, "used")
    _manifest(tmp_path, "unused")

    require_verified("used")

    assert accessed_sources() == frozenset({"used"})
    recorded = data_manifest_ids(only=accessed_sources())
    assert set(recorded) == {"used"}
    # 필터 없이 부르면 디스크의 모든 매니페스트가 나온다 — 옛 동작.
    assert set(data_manifest_ids()) == {"used", "unused"}


def test_run_records_only_the_sources_it_consumed(tmp_path: Path) -> None:
    _manifest(tmp_path, "used")
    _manifest(tmp_path, "unused")

    with tracked_run("demo", {}, [1]):
        require_verified("used")

    recorded = load_runs()[0]["data_manifests"]
    assert set(recorded) == {"used"}, recorded


def test_a_run_that_consumed_nothing_records_no_manifests(tmp_path: Path) -> None:
    """합성 전용 실행은 외부 데이터를 쓰지 않으므로 출처가 비어야 한다."""
    _manifest(tmp_path, "unused")
    with tracked_run("synthetic_only", {}, [1]):
        pass
    assert load_runs()[0]["data_manifests"] == {}


def test_failed_run_still_records_what_it_had_consumed(tmp_path: Path) -> None:
    _manifest(tmp_path, "used")
    with pytest.raises(RuntimeError), tracked_run("boom", {}, [1]):
        require_verified("used")
        raise RuntimeError("이후 폭발")

    rec = load_runs()[0]
    assert rec["status"] == "failed"
    assert set(rec["data_manifests"]) == {"used"}


def test_accessed_set_is_reset_per_run(tmp_path: Path) -> None:
    """앞선 실행이 만진 데이터셋이 다음 실행의 출처로 새어 들어가면 안 된다."""
    _manifest(tmp_path, "first")
    _manifest(tmp_path, "second")

    with tracked_run("run_a", {}, [1]):
        require_verified("first")
    with tracked_run("run_b", {}, [2]):
        require_verified("second")

    runs = {r["experiment"]: r for r in load_runs()}
    assert set(runs["run_a"]["data_manifests"]) == {"first"}
    assert set(runs["run_b"]["data_manifests"]) == {"second"}


# =========================================================== P0.3 의존성 지문


def test_environment_fingerprint_describes_the_installed_environment() -> None:
    fp = environment_fingerprint()
    assert fp["n_distributions"] > 10, "설치된 배포판이 거의 없다고 보고됐다"
    assert len(fp["digest"]) == 64
    assert fp["python"]
    # 실제로 설치된 것을 봐야 한다.
    assert "numpy" in fp["distributions"]
    assert "scikit-learn" in fp["distributions"]


def test_environment_fingerprint_is_stable_within_a_process() -> None:
    assert environment_fingerprint()["digest"] == environment_fingerprint()["digest"]


def test_fingerprint_changes_when_the_environment_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """지문이 실제로 환경에 반응해야 한다. 상수라면 아무것도 증명하지 않는다."""
    import audire.experiments.registry as reg

    baseline = environment_fingerprint()["digest"]
    monkeypatch.setattr(
        reg, "_installed_distributions", lambda: {"numpy": "0.0.0-fake", "audire": "0.1.0"}
    )
    assert environment_fingerprint()["digest"] != baseline


def test_lock_versus_environment_comparison_is_reported(tmp_path: Path) -> None:
    """선언된 락과 실제 환경이 어긋나는지를 판정해 보고해야 한다."""
    report = environment_matches_lock()
    assert report["status"] in {"match", "mismatch", "no_lockfile", "unknown"}
    if report["status"] == "mismatch":
        # 어긋난다면 무엇이 어긋났는지 말해야 한다.
        assert report["differences"], report


def test_run_records_both_declared_lock_and_actual_environment() -> None:
    with tracked_run("demo", {}, [1]):
        pass
    rec = load_runs()[0]
    assert rec["lock_hash"], "선언된 락 해시가 없다"
    assert rec["env_fingerprint"], "실제 환경 지문이 없다"
    assert rec["env_matches_lock"] in {"match", "mismatch", "no_lockfile", "unknown"}


def test_registry_does_not_claim_a_clean_tree_when_git_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """git 상태를 알 수 없으면 'unknown' 이어야지 false 로 거짓말하면 안 된다."""
    import audire.experiments.registry as reg

    monkeypatch.setattr(reg, "_git", lambda *_a, **_k: None)
    with tracked_run("demo", {}, [1]):
        pass
    rec = load_runs()[0]
    assert rec["git_dirty"] == "unknown"
    assert rec["git_sha"] == "unknown"


# =========================================================== 매니페스트 JSON


def test_manifest_roundtrip_preserves_digests(tmp_path: Path) -> None:
    m = _manifest(tmp_path)
    payload = json.loads((tmp_path / "manifests" / "demo.json").read_text(encoding="utf-8"))
    assert payload["content_digest"] == m.content_digest
    assert all(len(f["sha256"]) == 64 for f in payload["files"])
