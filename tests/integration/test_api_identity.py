"""P0.5(API 계층) — API 경계에서도 같은 신원 규칙이 적용되는지.

미션 §P0.5 는 "schema / profile store / scoring / **API** / logging" 전부가 같은
검증을 쓰라고 요구합니다. 보존 커밋 시점의 API 는 그렇지 않았습니다:

* 경로·폼 파라미터의 ``listener_id`` 가 ``min_length=1, max_length=64`` 길이 검증만
  받았습니다. ``../escape`` 나 사람 이름이 FastAPI 검증을 통과해 저장소까지 도달했고,
  거기서 나는 오류가 클라이언트에게 어떤 상태 코드로 보일지는 우연에 맡겨져 있었습니다.
* ``/api/process`` 가 ``check_ready(...)`` 를 ``listener_id=`` 없이 호출했습니다.
  디스크의 프로파일 내부 ``listener_id`` 가 경로와 어긋나도 API 자신은 잡지 못하고,
  한 계층 아래 ``score_transcript`` 가 ``IncompleteProfile`` 을 던져 500 으로
  드러났습니다. 같은 방어를 캘리브레이션 엔드포인트는 이미 하고 있었으므로
  (`stored listener id does not match path`) 일관성도 깨져 있었습니다.

여기서 강제하는 불변식:

1. 규칙에 맞지 않는 식별자는 **저장소에 닿기 전에** 4xx 로 거부된다.
2. 디스크 프로파일의 내부 식별자가 경로와 다르면 처리 엔드포인트가 409 로 거부한다.
3. 어떤 경우에도 5xx 로 새지 않는다 — 잘못된 입력은 서버 오류가 아니다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from audire.api import create_app
from audire.asr.replay import ReplayBackend
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

RECORDED = {
    "backend": "faster-whisper",
    "model_id": "small",
    "language": "ko",
    "language_probability": 0.99,
    "duration_s": 1.5,
    "provenance": {"media": "api-identity.wav"},
    "tokens": [
        {"text": "가족", "start_s": 0.0, "end_s": 0.6, "confidence": 0.95},
        {"text": "학교", "start_s": 0.7, "end_s": 1.3, "confidence": 0.91},
    ],
}

#: 규칙을 어기는 식별자. 경로 탈출·사람 이름·로그 위조 문자를 포함한다.
BAD_IDS = ["..", "%2e%2e", "김철수", "a b", ".hidden", "x" * 65]


def _hearing(listener_id: str) -> HearingProfile:
    ear = EarProfile(
        ear=Ear.RIGHT,
        audiogram=Audiogram(
            ear=Ear.RIGHT,
            thresholds={f: AudiogramPoint(db_hl=45.0) for f in (500, 1000, 2000, 4000)},
        ),
        speech=SpeechScores(
            ear=Ear.RIGHT, srt_db_hl=45.0, wrs_percent=68.0, wrs_presentation_level_db_hl=75.0
        ),
    )
    return HearingProfile(
        listener_id=listener_id,
        source=ProfileSource.SYNTHETIC,
        is_synthetic=True,
        right=ear,
    )


@pytest.fixture
def client(tmp_path: Path, fitted_scorer) -> TestClient:
    transcript = tmp_path / "recorded.json"
    transcript.write_text(json.dumps(RECORDED, ensure_ascii=False), encoding="utf-8")
    app = create_app(
        store=ProfileStore(tmp_path / "store"),
        backend=ReplayBackend(transcript, allow_media_mismatch=True),
        scorer=fitted_scorer,
        upload_dir=tmp_path / "uploads",
        auto_load_scorer=False,
    )
    # 서버 예외를 그대로 올리지 않아야 실제 클라이언트가 보는 상태 코드를 검사할 수 있다.
    return TestClient(app, raise_server_exceptions=False)


# =========================================================== 식별자 검증


@pytest.mark.parametrize("bad", BAD_IDS)
def test_invalid_listener_id_is_rejected_with_4xx(client: TestClient, bad: str) -> None:
    """잘못된 입력은 서버 오류가 아니다. 저장소에 닿기 전에 거부되어야 한다."""
    response = client.get(f"/api/profiles/{bad}")
    assert 400 <= response.status_code < 500, (
        f"{bad!r} 가 {response.status_code} 를 냈다 — 4xx 여야 한다"
    )


@pytest.mark.parametrize("bad", BAD_IDS)
def test_invalid_listener_id_never_produces_a_server_error(client: TestClient, bad: str) -> None:
    for path in (f"/api/profiles/{bad}/export", f"/api/profiles/{bad}"):
        assert client.get(path).status_code < 500, path


def test_invalid_listener_id_on_delete_is_rejected(client: TestClient) -> None:
    response = client.delete("/api/profiles/김철수")
    assert 400 <= response.status_code < 500


def test_invalid_listener_id_on_calibration_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/profiles/김철수/calibration",
        json={"responses": [{"stimulus_id": "s0", "target": "각", "response": "각"}]},
    )
    assert 400 <= response.status_code < 500


def test_profile_creation_rejects_an_unsafe_listener_id(client: TestClient) -> None:
    payload = _hearing("L001").model_dump(mode="json")
    payload["listener_id"] = "../escape"
    response = client.post("/api/profiles", json=payload)
    assert 400 <= response.status_code < 500


def test_valid_listener_id_still_works(client: TestClient) -> None:
    """검증을 넣느라 정상 경로를 막지 않았는지 확인한다."""
    created = client.post("/api/profiles", json=_hearing("L-001.a_2").model_dump(mode="json"))
    assert created.status_code == 201, created.text
    assert client.get("/api/profiles/L-001.a_2").status_code == 200


# =========================================================== 처리 경계의 신원 검사


def test_process_rejects_a_profile_whose_stored_id_disagrees(
    client: TestClient, tmp_path: Path
) -> None:
    """디스크 프로파일의 내부 식별자가 경로와 다르면 409 여야 한다(500 이 아니라).

    캘리브레이션 엔드포인트는 이미 이 방어를 하고 있었으므로 처리 엔드포인트만
    비어 있는 것은 일관성 결함이기도 하다.
    """
    store = ProfileStore(tmp_path / "store")
    # 저장은 정상 경로로 하고, 그 뒤 디스크에서 내부 식별자만 어긋나게 만든다.
    store.save_hearing(_hearing("L001"))
    path = store.hearing_path("L001")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["listener_id"] = "L999"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    response = client.post(
        "/api/process",
        data={"listener_id": "L001", "policy": "full"},
        files={"media": ("api-identity.wav", b"RIFF", "audio/wav")},
    )
    assert response.status_code < 500, f"5xx 로 샜다: {response.text[:200]}"
    assert response.status_code == 409, response.text


def test_process_rejects_an_unsafe_listener_id(client: TestClient) -> None:
    response = client.post(
        "/api/process",
        data={"listener_id": "../escape", "policy": "full"},
        files={"media": ("api-identity.wav", b"RIFF", "audio/wav")},
    )
    assert 400 <= response.status_code < 500, response.text
