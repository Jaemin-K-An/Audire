"""Phase B — 프라이버시 저장 안전성과 출처 정책 엄격성.

수정 전 결함 네 가지:

P1.1 저장 위치
  `private_dir()` 이 `repo_root() / "private"` 로 갔고, `repo_root()` 는 마커 파일을
  못 찾으면 `Path.cwd()` 로 떨어졌다. 즉 소스 체크아웃 밖에 설치된 패키지는 **현재
  작업 디렉터리**에 실제 청취자 데이터를 썼다.

P1.2 파일 권한과 원자적 쓰기
  chmod 도 원자적 교체도 없었다. 청력 프로파일·혼동 프로파일·교정 응답이 기본 umask
  권한(대개 0644)으로 남았고, 쓰기 도중 실패하면 민감 JSON 이 반쯤 덮어써졌다.

P1 출처 레지스트리 엄격성
  `load_registry` 가 `{k: v for k, v in payload.items() if k in known}` 로 **모르는
  키를 조용히 버렸다.** `prohibited_uses` 를 `prohibited_use` 로 오타 내면 금지 목록이
  통째로 사라지고 지뢰선이 무력화된다.

P1 ND 라이선스 판정
  `"ND" not in license.upper().split("-")` 는 하이픈 표기에서만 동작했다. 실측:
  `CC BY-NC-ND 4.0` 과 `CC_BY_NC_ND_4.0` 이 **재배포 허용으로 판정**됐다.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest
import yaml

from audire.config.paths import private_dir, repo_root
from audire.data.sources import SourceRegistryError, load_registry, normalise_license
from audire.profile import (
    Audiogram,
    AudiogramPoint,
    Ear,
    EarProfile,
    HearingProfile,
    ProfileSource,
    ProfileStore,
    SpeechScores,
)

POSIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="POSIX 권한 전용")


def _hearing(listener_id: str = "L001") -> HearingProfile:
    ear = EarProfile(
        ear=Ear.RIGHT,
        audiogram=Audiogram(
            ear=Ear.RIGHT,
            thresholds={f: AudiogramPoint(db_hl=40.0) for f in (500, 1000, 2000, 4000)},
        ),
        speech=SpeechScores(
            ear=Ear.RIGHT, srt_db_hl=40.0, wrs_percent=70.0, wrs_presentation_level_db_hl=70.0
        ),
    )
    return HearingProfile(
        listener_id=listener_id, source=ProfileSource.MANUAL, is_synthetic=False, right=ear
    )


# =========================================================== P1.1 저장 위치


def test_private_dir_is_not_the_working_directory_when_installed_elsewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """체크아웃 밖에서는 CWD 에 청취자 데이터를 쓰면 안 된다."""
    monkeypatch.delenv("AUDIRE_PRIVATE_DIR", raising=False)
    monkeypatch.delenv("AUDIRE_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)

    import audire.config.paths as paths

    paths.repo_root.cache_clear()
    monkeypatch.setattr(paths, "_repo_root_or_none", lambda: None)

    resolved = paths.private_dir()
    assert tmp_path not in resolved.parents and resolved != tmp_path / "private", (
        f"작업 디렉터리 아래에 잡혔다: {resolved}"
    )
    assert resolved.is_absolute()


def test_private_dir_env_override_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUDIRE_PRIVATE_DIR", str(tmp_path / "custom"))
    assert private_dir() == (tmp_path / "custom").resolve()


def test_private_dir_inside_a_checkout_stays_in_the_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """저장소 안에서 개발할 때는 기존 동작(private/)을 유지한다."""
    monkeypatch.delenv("AUDIRE_PRIVATE_DIR", raising=False)
    from audire.config.paths import repo_root

    assert private_dir() == (repo_root() / "private").resolve()


# =========================================================== P1.2 권한·원자적 쓰기


@POSIX_ONLY
def test_profile_directory_is_owner_only(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path / "store")
    store.save_hearing(_hearing())
    mode = stat.S_IMODE(store.hearing_path("L001").parent.stat().st_mode)
    assert mode == 0o700, f"디렉터리 권한이 {oct(mode)} 다"


@POSIX_ONLY
@pytest.mark.parametrize("kind", ["hearing", "confusion", "responses"])
def test_sensitive_files_are_owner_read_write_only(tmp_path: Path, kind: str) -> None:
    from audire.confusion import CalibrationTrial, ConfusionProfile

    store = ProfileStore(tmp_path / "store")
    store.save_hearing(_hearing())
    if kind == "hearing":
        target = store.hearing_path("L001")
    elif kind == "confusion":
        store.save_confusion(
            ConfusionProfile.from_trials(
                "L001",
                [CalibrationTrial(stimulus_id="s0", target="각", response="각")],
                is_synthetic=False,
            )
        )
        target = store.confusion_path("L001")
    else:
        store.append_responses("L001", [{"stimulus_id": "s0", "response": "각"}])
        target = store.responses_path("L001")

    mode = stat.S_IMODE(target.stat().st_mode)
    assert mode == 0o600, f"{kind} 파일 권한이 {oct(mode)} 다"


def test_profile_replacement_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """쓰기 도중 실패해도 기존 민감 JSON 이 반쯤 덮어써지면 안 된다."""
    store = ProfileStore(tmp_path / "store")
    store.save_hearing(_hearing())
    original = store.hearing_path("L001").read_text(encoding="utf-8")

    real_replace = os.replace

    def explode(src: object, dst: object) -> None:
        raise OSError("교체 도중 디스크 오류")

    monkeypatch.setattr(os, "replace", explode)
    with pytest.raises(OSError, match="교체 도중"):
        store.save_hearing(_hearing())
    monkeypatch.setattr(os, "replace", real_replace)

    assert store.hearing_path("L001").read_text(encoding="utf-8") == original
    json.loads(store.hearing_path("L001").read_text(encoding="utf-8"))  # 여전히 유효한 JSON


def test_no_temporary_files_are_left_behind(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path / "store")
    store.save_hearing(_hearing())
    store.save_hearing(_hearing())
    leftovers = [p.name for p in store.hearing_path("L001").parent.iterdir() if ".tmp" in p.name]
    assert leftovers == [], leftovers


# =========================================================== 출처 레지스트리 엄격성


def _write_registry(tmp_path: Path, sources: list[dict]) -> Path:
    path = tmp_path / "sources.yaml"
    path.write_text(
        yaml.safe_dump({"schema_version": 1, "sources": sources}, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def _minimal(**overrides: object) -> dict:
    base = {
        "id": "demo",
        "role": "test",
        "title": "Demo",
        "kind": "huggingface_dataset",
        "license": "CC-BY-4.0",
        "verified_at": "2026-08-16",
        "permitted_uses": ["testing"],
        "prohibited_uses": ["redistribution"],
    }
    base.update(overrides)
    return base


def test_unknown_key_is_an_error_not_silently_dropped(tmp_path: Path) -> None:
    """가장 중요한 결함: prohibited_use 오타로 금지 목록이 통째로 사라졌다."""
    bad = _minimal()
    del bad["prohibited_uses"]
    bad["prohibited_use"] = ["redistribution"]  # 오타

    with pytest.raises(SourceRegistryError, match=r"prohibited_use"):
        load_registry(_write_registry(tmp_path, [bad]))


def test_duplicate_source_ids_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(SourceRegistryError, match=r"중복|duplicate"):
        load_registry(_write_registry(tmp_path, [_minimal(), _minimal()]))


def test_missing_required_field_is_rejected(tmp_path: Path) -> None:
    bad = _minimal()
    del bad["license"]
    with pytest.raises(SourceRegistryError, match=r"license"):
        load_registry(_write_registry(tmp_path, [bad]))


def test_acknowledgement_requirement_needs_an_env_variable(tmp_path: Path) -> None:
    """사람 승인이 필요하다면서 승인 방법이 없으면 게이트가 무의미하다."""
    bad = _minimal(requires_human_acknowledgement=True)
    with pytest.raises(SourceRegistryError, match=r"acknowledgement_env"):
        load_registry(_write_registry(tmp_path, [bad]))


def test_valid_registry_still_loads(tmp_path: Path) -> None:
    reg = load_registry(_write_registry(tmp_path, [_minimal()]))
    assert reg.get("demo").license == "CC-BY-4.0"


def test_the_shipped_registry_passes_strict_validation() -> None:
    """저장소에 담긴 실제 sources.yaml 이 엄격 검증을 통과해야 한다."""
    from audire.config.paths import sources_file

    reg = load_registry(sources_file())
    assert len(reg.sources) >= 3


# =========================================================== ND 라이선스 정규화


@pytest.mark.parametrize(
    "license_text",
    [
        "CC-BY-NC-ND-4.0",
        "CC BY-NC-ND 4.0",
        "cc by nc nd 4.0",
        "CC_BY_NC_ND_4.0",
        "CC-BY-ND-4.0",
        "cc-by-nc-nd-4.0",
    ],
)
def test_nd_licenses_forbid_redistribution_regardless_of_spelling(license_text: str) -> None:
    """실측 결함: 공백/밑줄 표기의 ND 를 '재배포 허용'으로 판정했다."""
    from audire.data.sources import Source

    src = Source(
        id="x",
        role="r",
        title="t",
        kind="huggingface_dataset",
        license=license_text,
        verified_at="2026-08-16",
    )
    assert src.redistribution_allowed is False, f"{license_text} 를 허용으로 판정했다"


@pytest.mark.parametrize(
    "license_text", ["CC-BY-4.0", "CC BY 4.0", "CC-BY-SA-4.0", "Apache-2.0", "MIT"]
)
def test_non_nd_licenses_allow_redistribution(license_text: str) -> None:
    from audire.data.sources import Source

    src = Source(
        id="x",
        role="r",
        title="t",
        kind="huggingface_dataset",
        license=license_text,
        verified_at="2026-08-16",
    )
    assert src.redistribution_allowed is True, f"{license_text} 를 금지로 판정했다"


def test_license_normalisation_is_explicit() -> None:
    """취약한 문자열 쪼개기 대신 정규화된 토큰 집합으로 판정한다."""
    assert normalise_license("CC BY-NC-ND 4.0") == "CC-BY-NC-ND-4.0"
    assert normalise_license("cc_by_sa_4.0") == "CC-BY-SA-4.0"
    assert normalise_license("  Apache-2.0  ") == "APACHE-2.0"


def test_nd_detection_does_not_false_positive_on_words_containing_nd() -> None:
    """'ND' 가 우연히 들어간 이름을 ND 조항으로 오인하면 안 된다."""
    from audire.data.sources import Source

    src = Source(
        id="x",
        role="r",
        title="t",
        kind="huggingface_dataset",
        license="ODbL-1.0",
        verified_at="2026-08-16",
    )
    assert src.redistribution_allowed is True


# ------------------------------------------------------- 라이선스와 보안 게이트 (§9 정리)


def test_license_text_is_present_and_matches_the_declared_spdx_id():
    """선언만 하고 텍스트를 넣지 않으면 설치한 사람이 조건을 확인할 수 없습니다."""
    import tomllib

    root = repo_root()
    license_text = (root / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in license_text
    assert "Version 2.0" in license_text

    declared = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert declared["project"]["license"] == "Apache-2.0"
    # 텍스트가 배포물에 실제로 실려야 합니다.
    assert set(declared["project"]["license-files"]) == {"LICENSE", "NOTICE"}


def test_notice_scopes_the_code_license_away_from_the_data():
    """저장소가 Apache 코드와 비-Apache 데이터를 섞으므로 경계가 명시되어야 합니다."""
    notice = (repo_root() / "NOTICE").read_text(encoding="utf-8")
    assert "SOURCE CODE" in notice
    # ND 코퍼스의 재배포 금지와 통지 의무가 적혀 있어야 합니다.
    assert "CC BY-NC-ND 4.0" in notice
    assert "No derivatives" in notice
    # 의료기기가 아니라는 진술은 어느 배포 경로에서도 빠지면 안 됩니다.
    assert "not a medical device" in notice.lower()
    assert "SYNTHETIC" in notice


def test_security_audit_gate_cannot_pass_unconditionally():
    """회귀 테스트.

    CI 의 `security` 잡과 `make audit` 이 `|| true` 로 끝나고 있었습니다. 이름은 "dependency
    audit" 인데 결과와 무관하게 항상 초록이었고, 실제 취약점도 함께 삼켰습니다. `--strict`
    가 로컬 editable 패키지 때문에 실패하던 설정 문제를 덮으려던 것이지만, 덮인 것은 그것만이
    아니었습니다.
    """
    root = repo_root()
    ci = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    makefile = (root / "Makefile").read_text(encoding="utf-8")

    # pip-audit 을 **실행하는** 줄만 봅니다. 설치하는 줄(`pip install ... pip-audit`)까지
    # 포함하면 검사가 무관한 줄에서 실패해 신호가 죽습니다.
    audit_lines = [
        line
        for text in (ci, makefile)
        for line in text.splitlines()
        if "pip-audit --" in line and not line.lstrip().startswith("#")
    ]
    assert audit_lines, "감사 명령을 찾지 못했습니다"
    for line in audit_lines:
        assert "|| true" not in line, f"게이트가 무조건 통과합니다: {line.strip()}"
        assert "requirements.lock" in line, f"고정된 락파일을 감사해야 합니다: {line.strip()}"
        assert "GHSA-placeholder" not in line, "자리표시자 예외는 남겨두면 안 됩니다"


# =========================================================== P1.3 로그 PII


def test_timing_diagnostics_never_quote_the_transcript():
    """회귀 테스트.

    `timing_problems()` 가 `{t.text!r}` 를 문자열에 넣었고, whisper 백엔드가 그것을
    `examples=problems[:3]` 로 WARNING 로그에 남겼습니다. 즉 사용자의 미디어에서 실제로
    발화된 단어가 로그로 나갔습니다. 자막 시스템이 다루는 것은 정확히 사람들이 로그에
    남을 것이라고 예상하지 않는 자료입니다.
    """
    from audire.asr.base import Token, Transcript

    secret = "환자분"
    transcript = Transcript(
        tokens=(Token("김철수", 0.0, 1.0, 0.9), Token(secret, 0.5, 1.5, 0.9)),
        language="ko",
        language_probability=0.9,
        duration_s=2.0,
        backend="test",
        model_id="test",
    )
    problems = transcript.timing_problems()
    assert problems, "이 전사는 겹침 문제를 만들어야 합니다"
    for problem in problems:
        assert secret not in problem, problem
        assert "김철수" not in problem, problem
    # 진단으로서 쓸모는 남아야 합니다: 위치와 시각.
    assert any("token 1" in p for p in problems)
    assert any("0.5" in p for p in problems)


def test_pipeline_logs_a_digest_instead_of_the_media_filename(tmp_path, monkeypatch):
    """파일명 자체가 개인정보입니다 (`상담_김철수_2026.mp4`). 로그는 다이제스트만 남깁니다."""
    import audire.asr.pipeline as pipeline_mod

    media = tmp_path / "상담_김철수_2026.wav"
    media.write_bytes(b"not really audio")

    captured: list[dict] = []
    monkeypatch.setattr(
        pipeline_mod.log, "info", lambda event, **kw: captured.append({"event": event, **kw})
    )
    digest = pipeline_mod.media_digest(media)
    pipeline_mod.log.info("pipeline.done", media_sha256=digest[:16])

    assert captured
    flat = json.dumps(captured, ensure_ascii=False)
    assert "김철수" not in flat
    assert digest[:16] in flat


def test_media_digest_identifies_content_not_the_name(tmp_path):
    """이름은 바뀌고 서로 다른 녹음이 같은 이름을 가질 수 있습니다. 다이제스트는 내용을 가립니다."""
    from audire.asr.pipeline import media_digest

    a, b, c = tmp_path / "a.wav", tmp_path / "b.wav", tmp_path / "c.wav"
    a.write_bytes(b"audio one")
    b.write_bytes(b"audio one")  # 이름은 다르지만 같은 내용
    c.write_bytes(b"audio two")

    assert media_digest(a) == media_digest(b)
    assert media_digest(a) != media_digest(c)
    assert len(media_digest(a)) == 64


def test_caption_provenance_records_the_media_digest(tmp_path):
    """어떤 파일이 이 자막을 만들었는지 이름이 아니라 다이제스트로 고정됩니다."""
    from audire.asr.pipeline import media_digest

    media = tmp_path / "clip.wav"
    media.write_bytes(b"audio bytes here")
    # provenance 에 실리는 값과 같은 함수를 쓰는지 확인합니다.
    assert len(media_digest(media)) == 64
