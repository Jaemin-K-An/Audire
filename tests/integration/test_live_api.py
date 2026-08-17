"""Phase 5 — 로컬 라이브 API 계약.

미션 §27 이 요구하는 항목을 그대로 고정합니다. 특히 **자막 내용이 로그에 남지 않는 것**과
**실패 상태가 구분되는 것** 이 핵심입니다 — 추론이 성공한 것처럼 빈 자막을 내보내면
사용자는 자기가 안전하다고 잘못 믿게 됩니다.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from audire.live.service import MAX_CUE_CHARS, LiveScorer
from audire.profile.schema import (
    Audiogram,
    AudiogramPoint,
    Ear,
    EarProfile,
    HearingProfile,
    ProfileSource,
    SpeechScores,
)

LISTENER = "L001"
SECRET_CUE = "비밀자막내용입니다"


@pytest.fixture
def client(store, live_scorer):
    from audire.api import create_app

    return TestClient(create_app(store=store, auto_load_scorer=False, live_scorer=live_scorer))


@pytest.fixture
def token(client) -> str:
    return client.post("/api/live/pair", json={"label": "test"}).json()["token"]


def _score(client, token, **overrides):
    payload = {"profile_id": LISTENER, "cue_id": "c1", "text": "오늘 날씨가 좋네요"}
    payload.update(overrides)
    return client.post("/api/live/score-cue", headers={"X-Audire-Token": token}, json=payload)


# --------------------------------------------------------------------------- 페어링


def test_unpaired_requests_are_rejected(client):
    assert client.get("/api/live/profiles").status_code == 401
    assert client.post("/api/live/score-cue", json={}).status_code in (401, 422)


def test_a_valid_token_is_accepted(client, token):
    assert client.get("/api/live/profiles", headers={"X-Audire-Token": token}).status_code == 200


def test_an_invalid_token_is_rejected(client, token):
    response = client.get("/api/live/profiles", headers={"X-Audire-Token": "wrong"})
    assert response.status_code == 401
    assert response.json()["detail"]["reason"] == "not_paired"


def test_revoking_the_pairing_blocks_further_requests(client, token):
    assert client.get("/api/live/profiles", headers={"X-Audire-Token": token}).status_code == 200
    client.delete("/api/live/pair", headers={"X-Audire-Token": token})
    assert client.get("/api/live/profiles", headers={"X-Audire-Token": token}).status_code == 401


def test_re_pairing_invalidates_the_previous_token(client, token):
    """토큰을 잃어버렸을 때의 탈출구. 새로 만들면 옛것이 무효가 됩니다."""
    fresh = client.post("/api/live/pair", json={}).json()["token"]
    assert fresh != token
    assert client.get("/api/live/profiles", headers={"X-Audire-Token": token}).status_code == 401
    assert client.get("/api/live/profiles", headers={"X-Audire-Token": fresh}).status_code == 200


def test_a_web_page_origin_cannot_reach_the_live_api(client):
    """웹 페이지 출처는 라이브 API 에 닿을 수 없습니다.

    이것이 페어링을 지탱하는 전제입니다. `POST /pair` 는 토큰이 없는 상태에서 부르는
    엔드포인트이므로, 같은 기기의 아무 페이지나 이것을 부를 수 있으면 그 페이지가 조용히
    새 토큰을 받아 청취자 목록을 읽고 임의의 텍스트를 채점시킬 수 있습니다. 그 목록은
    실존 인물의 건강 관련 메타데이터입니다.

    `Origin` 은 브라우저가 붙이고 페이지가 바꿀 수 없는 헤더입니다. 그래서 브라우저 안의
    공격자에게는 실제 경계가 됩니다. 같은 사용자로 도는 네이티브 프로세스는 무엇이든
    위조할 수 있고, 그것은 이 설계가 막는다고 주장하지 않는 범위입니다.
    """
    page = {"Origin": "http://localhost:5173"}
    assert client.post("/api/live/pair", json={}, headers=page).status_code == 403
    assert client.get("/api/live/status", headers=page).status_code == 403
    assert client.get("/api/live/profiles", headers=page).status_code == 403
    assert client.delete("/api/live/pair", headers=page).status_code == 403


def test_the_extension_origin_is_accepted(client):
    extension = {"Origin": "chrome-extension://abcdefghijklmnopabcdefghijklmnop"}
    assert client.get("/api/live/status", headers=extension).status_code == 200
    assert client.post("/api/live/pair", json={}, headers=extension).status_code == 201


def test_a_refused_origin_is_told_why(client):
    body = client.get("/api/live/status", headers={"Origin": "http://localhost:5173"}).json()
    assert body["detail"]["reason"] == "origin_not_allowed"


def test_revoking_requires_the_token(client, token):
    """토큰 없이 페어링을 지울 수 있으면 아무나 확장을 끊을 수 있습니다."""
    assert client.delete("/api/live/pair").status_code == 401
    assert client.get("/api/live/profiles", headers={"X-Audire-Token": token}).status_code == 200

    revoked = client.delete("/api/live/pair", headers={"X-Audire-Token": token})
    assert revoked.status_code == 200
    assert revoked.json()["revoked"] is True
    assert client.get("/api/live/profiles", headers={"X-Audire-Token": token}).status_code == 401


def test_the_token_appears_only_in_the_pairing_response(client, token):
    """토큰이 로그에 실리면 로그를 읽을 수 있는 프로그램이 그대로 붙을 수 있습니다.

    구조화 로그는 설정 시점의 스트림에 묶이므로 caplog/capfd 로는 잡히지 않습니다.
    structlog 전용 캡처를 써야 검사가 자명하게 통과하지 않습니다.
    """
    from structlog.testing import capture_logs

    with capture_logs() as logs:
        client.get("/api/live/status")
        _score(client, token)
    assert token not in json.dumps(logs, ensure_ascii=False)


def test_status_reports_pairing_and_model_readiness(client):
    body = client.get("/api/live/status").json()
    for key in ("server", "paired", "model_ready", "input_contract", "disclaimer"):
        assert key in body
    assert body["input_contract"] == "live-caption-v1"


# ----------------------------------------------------------------- 프로파일 노출 범위


def test_profiles_expose_only_ui_safe_fields(client, token):
    profiles = client.get("/api/live/profiles", headers={"X-Audire-Token": token}).json()[
        "profiles"
    ]
    assert profiles
    allowed = {"id", "alias", "has_hearing_profile", "calibration_trials", "ready"}
    for entry in profiles:
        assert set(entry) == allowed
    # 임상 원값이 확장으로 나가면 안 됩니다.
    flat = json.dumps(profiles, ensure_ascii=False)
    for leaked in ("db_hl", "wrs_percent", "srt_db_hl", "audiogram", "thresholds"):
        assert leaked not in flat


def test_an_unknown_profile_is_rejected(client, token):
    response = _score(client, token, profile_id="nobody")
    assert response.status_code == 404
    assert response.json()["detail"]["reason"] == "unknown_profile"


def test_a_profile_without_calibration_is_rejected(client, token, store):
    ear = EarProfile(
        ear=Ear.LEFT,
        audiogram=Audiogram(
            ear=Ear.LEFT, thresholds={f: AudiogramPoint(db_hl=30.0) for f in (500, 1000)}
        ),
        speech=SpeechScores(ear=Ear.LEFT),
    )
    store.save_hearing(
        HearingProfile(
            listener_id="L002", source=ProfileSource.MANUAL, is_synthetic=False, left=ear
        )
    )
    response = _score(client, token, profile_id="L002")
    assert response.status_code == 409
    assert response.json()["detail"]["reason"] == "profile_not_ready"


# --------------------------------------------------------------------------- 큐 채점


def test_a_cue_is_scored_and_returns_provenance(client, token):
    body = _score(client, token).json()
    assert body["words"]
    model = body["model"]
    assert model["input_contract"] == "live-caption-v1"
    assert model["arm"] == "live_word_context_clinical_confusion"
    assert model["training_source"] == "synthetic simulation"
    assert model["human_efficacy_evidence"] is False
    assert model["threshold_policy"] == "per_listener"


def test_dom_captions_never_carry_an_asr_confidence(client, token):
    """DOM 자막에는 인식기가 없습니다. 값을 지어내면 안 됩니다."""
    assert _score(client, token).json()["asr_confidence"] is None


def test_the_same_cue_and_profile_are_deterministic(client, token):
    a = _score(client, token).json()
    b = _score(client, token).json()
    assert [w["risk"] for w in a["words"]] == [w["risk"] for w in b["words"]]
    assert a["threshold"] == b["threshold"]


def test_an_over_long_cue_is_rejected(client, token):
    response = _score(client, token, text="가" * (MAX_CUE_CHARS + 10))
    assert response.status_code == 422


def test_html_in_the_cue_stays_text(client, token):
    """페이지 내용은 신뢰할 수 없는 입력입니다. 해석하지 않고 텍스트로 둡니다."""
    body = _score(client, token, text="<script>alert(1)</script> 안녕").json()
    texts = [w["text"] for w in body["words"]]
    assert "<script>alert(1)</script>" in texts
    # 태그가 제거되거나 실행 가능한 형태로 재구성되지 않아야 합니다.
    assert "".join(texts).count("<script>") == 1


def test_non_hangul_tokens_are_kept_but_never_selected(client, token):
    """한국어 음소 혼동 프로파일로 채점할 수 없는 토큰을 지어내 채점하지 않습니다."""
    body = _score(client, token, text="ABC 123 안녕").json()
    by_text = {w["text"]: w for w in body["words"]}
    assert by_text["ABC"]["risk"] == 0.0
    assert by_text["ABC"]["selected"] is False
    assert by_text["123"]["selected"] is False


def test_the_threshold_is_per_listener_not_global(live_scorer, store):
    """ADR-0021 의 강제 조건.

    전역 임계값은 E30 에서 normal 청취자에게 자막률 0.0004 를 주고 중앙값 청취자의
    재현율을 0 으로 만들었습니다. 임계값은 청취자마다 달라야 합니다.
    """
    stored = store.load(LISTENER)

    mild_ear = EarProfile(
        ear=Ear.RIGHT,
        audiogram=Audiogram(
            ear=Ear.RIGHT,
            thresholds={f: AudiogramPoint(db_hl=15.0) for f in (500, 1000, 2000, 4000)},
        ),
        speech=SpeechScores(
            ear=Ear.RIGHT, srt_db_hl=15.0, wrs_percent=96.0, wrs_presentation_level_db_hl=55.0
        ),
    )
    mild = HearingProfile(
        listener_id=LISTENER, source=ProfileSource.MANUAL, is_synthetic=False, right=mild_ear
    )

    severe_threshold = live_scorer.threshold_for(LISTENER, stored.hearing, stored.confusion)
    mild_threshold = live_scorer.threshold_for(LISTENER, mild, stored.confusion)
    assert severe_threshold != mild_threshold, (
        "청력이 다른 청취자가 같은 임계값을 받으면 전역 임계값과 다를 바가 없습니다"
    )


def test_target_caption_rate_moves_the_threshold(client, token, live_scorer, store):
    stored = store.load(LISTENER)
    low = live_scorer.threshold_for(LISTENER, stored.hearing, stored.confusion, 0.10)
    high = live_scorer.threshold_for(LISTENER, stored.hearing, stored.confusion, 0.40)
    assert low > high, "자막을 더 보여주려면 임계값이 낮아져야 합니다"


# ------------------------------------------------------------------ 자막 내용 미기록


def test_cue_text_never_reaches_the_logs(client, token):
    """이 저장소는 전사 텍스트를 이미 민감 정보로 다룹니다. 라이브도 같아야 합니다."""
    from structlog.testing import capture_logs

    with capture_logs() as logs:
        response = _score(client, token, text=SECRET_CUE, source="youtube")
    assert response.status_code == 200
    assert SECRET_CUE not in json.dumps(logs, ensure_ascii=False)


def test_logs_keep_only_shape_and_latency(client, token):
    """진단에 필요한 것은 남되 내용은 남지 않아야 합니다."""
    from structlog.testing import capture_logs

    with capture_logs() as logs:
        _score(client, token, text=SECRET_CUE, source="youtube")

    scored = [entry for entry in logs if entry.get("event") == "live.cue_scored"]
    assert scored, logs
    entry = scored[0]
    for key in ("source", "cue_chars", "n_words", "n_selected", "latency_ms"):
        assert key in entry, key
    # 남은 값 어디에도 자막 내용이 없어야 합니다.
    assert SECRET_CUE not in json.dumps(entry, ensure_ascii=False)


def test_an_error_response_does_not_echo_the_cue(client, token):
    """오류 메시지로 자막이 새어 나가는 경로도 막아야 합니다."""
    response = _score(client, token, text=SECRET_CUE * 200)
    assert response.status_code == 422
    assert SECRET_CUE not in json.dumps(response.json(), ensure_ascii=False)


# --------------------------------------------------------------------- 계약 분리


def test_a_media_artifact_cannot_serve_the_live_route(store):
    from audire.live.contract import ContractViolation

    media_metadata = {"input_contract": "media-pipeline-v1"}
    with pytest.raises(ContractViolation, match="live-caption-v1"):
        LiveScorer(scorer=None, artifact_metadata=media_metadata)  # type: ignore[arg-type]


def test_missing_live_artifact_is_an_explicit_failure(store):
    from audire.api import create_app

    client = TestClient(create_app(store=store, auto_load_scorer=False))
    token = client.post("/api/live/pair", json={"label": "t"}).json()["token"]
    response = client.post(
        "/api/live/score-cue",
        headers={"X-Audire-Token": token},
        json={"profile_id": LISTENER, "cue_id": "c1", "text": "안녕"},
    )
    assert response.status_code == 503
    assert response.json()["detail"]["reason"] == "model_unavailable"
