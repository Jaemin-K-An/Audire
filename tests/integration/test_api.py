"""G6 API contract: private profiles, calibration, media processing and exports."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from audire.api import create_app
from audire.asr import ASRBackend, ASRUnavailable, ReplayBackend, Transcript
from audire.eval.ablation import cohort_matrix
from audire.profile import ProfileStore
from audire.risk import FeatureSpec, LogisticRiskModel, WordScorer
from audire.sim import SimulationConfig, build_cohort

RECORDED = {
    "backend": "faster-whisper",
    "model_id": "small",
    "language": "ko",
    "language_probability": 0.99,
    "duration_s": 2.0,
    "provenance": {"media": "api-fixture.wav", "recorded_for": "AUDIRE API test"},
    "tokens": [
        {"text": "오늘", "start_s": 0.1, "end_s": 0.5, "confidence": 0.97},
        {"text": "정말", "start_s": 0.7, "end_s": 1.1, "confidence": 0.42},
        {"text": "좋아요", "start_s": 1.3, "end_s": 1.8, "confidence": 0.94},
    ],
}


@pytest.fixture(scope="module")
def fitted_record():
    cfg = SimulationConfig(
        name="api-contract",
        n_listeners=16,
        n_calibration_trials=60,
        n_word_trials=60,
        seeds=[43],
    )
    cohort = build_cohort(cfg, 43)
    spec = FeatureSpec.arm("clinical_plus_confusion", speakers=("male", "female", "unknown"))
    scorer = WordScorer(LogisticRiskModel().fit(cohort_matrix(cohort, spec)), spec)
    return scorer, cohort.records[0]


@pytest.fixture
def api(tmp_path: Path, fitted_record):
    scorer, record = fitted_record
    transcript_path = tmp_path / "recorded.json"
    transcript_path.write_text(json.dumps(RECORDED, ensure_ascii=False), encoding="utf-8")
    store = ProfileStore(tmp_path / "profiles")
    upload_dir = tmp_path / "uploads"
    app = create_app(
        store=store,
        backend=ReplayBackend(transcript_path, allow_media_mismatch=True),
        scorer=scorer,
        upload_dir=upload_dir,
    )
    return TestClient(app), store, upload_dir, record


def _create_profile(client: TestClient, record) -> None:
    response = client.post("/api/profiles", json=record.hearing.model_dump(mode="json"))
    assert response.status_code == 201, response.text


def _calibrate(client: TestClient, record, n: int = 12) -> None:
    response = client.post(
        f"/api/profiles/{record.listener_id}/calibration",
        json={"trials": [asdict(t) for t in record.calibration[:n]]},
    )
    assert response.status_code == 200, response.text


def test_health_and_web_shell_report_readiness(api) -> None:
    client, _, _, _ = api
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_ready"] is True
    assert health.json()["asr"]["backend"] == "replay"

    page = client.get("/")
    assert page.status_code == 200
    assert "AUDIRE" in page.text
    assert "의료기기가 아닙니다" in page.text
    assert client.get("/static/app.js").status_code == 200


def test_profile_calibration_processing_and_erasure_vertical_path(api) -> None:
    client, store, upload_dir, record = api
    _create_profile(client, record)

    listed = client.get("/api/profiles").json()["profiles"]
    assert [row["listener_id"] for row in listed] == [record.listener_id]
    assert listed[0]["has_calibration"] is False

    stimuli = client.get("/api/calibration/stimuli", params={"limit": 12})
    assert stimuli.status_code == 200
    assert len(stimuli.json()["stimuli"]) == 12
    assert stimuli.json()["provenance"]["audio"].endswith("not clinical")

    _calibrate(client, record)
    profile = client.get(f"/api/profiles/{record.listener_id}").json()
    assert profile["has_calibration"] is True
    assert profile["calibration"]["n_trials"] == 12
    assert len(store.load_responses(record.listener_id)) == 12
    exported_profile = client.get(f"/api/profiles/{record.listener_id}/export").json()
    assert len(exported_profile["calibration_responses"]) == 12

    processed = client.post(
        "/api/process",
        data={
            "listener_id": record.listener_id,
            "policy": "budget",
            "budget": "0.34",
            "snr_db": "20",
            "speaker": "unknown",
        },
        files={"media": ("speech.wav", b"RIFF-test", "audio/wav")},
    )
    assert processed.status_code == 200, processed.text
    payload = processed.json()
    assert payload["summary"]["n_words"] == 3
    assert payload["summary"]["policy"]["policy"] == "budget"
    assert "-->" in payload["exports"]["srt"]
    assert "[Script Info]" in payload["exports"]["ass"]
    research_export = json.loads(payload["exports"]["json"])
    assert research_export["schema"] == "audire.caption.v1"
    assert research_export["provenance"]["asr"]["backend"] == "replay"
    assert all("listener_risk" in word for word in research_export["words"])
    assert all("asr_confidence" in word for word in research_export["words"])
    assert list(upload_dir.glob("*")) == []

    erased = client.delete(f"/api/profiles/{record.listener_id}")
    assert erased.status_code == 200
    assert erased.json()["removed"]
    assert store.list_ids() == []


def test_calibration_rejects_a_tampered_target_before_writing(api) -> None:
    client, store, _, record = api
    _create_profile(client, record)
    row = asdict(record.calibration[0])
    row["target"] = "힣"

    response = client.post(
        f"/api/profiles/{record.listener_id}/calibration", json={"trials": [row]}
    )

    assert response.status_code == 422
    assert "does not match" in response.json()["detail"]
    assert store.load_responses(record.listener_id) == []

    unknown = {**asdict(record.calibration[0]), "stimulus_id": "builtin-unknown"}
    response = client.post(
        f"/api/profiles/{record.listener_id}/calibration", json={"trials": [unknown]}
    )
    assert response.status_code == 422
    assert "unknown built-in stimulus" in response.json()["detail"]


def test_processing_failure_paths_are_explicit(api, tmp_path: Path, fitted_record) -> None:
    client, _, _, record = api
    _create_profile(client, record)

    missing_calibration = client.post(
        "/api/process",
        data={"listener_id": record.listener_id},
        files={"media": ("speech.wav", b"RIFF", "audio/wav")},
    )
    assert missing_calibration.status_code == 409
    assert "run a calibration first" in missing_calibration.json()["detail"]

    unsupported = client.post(
        "/api/process",
        data={"listener_id": record.listener_id},
        files={"media": ("speech.exe", b"not-media", "application/octet-stream")},
    )
    assert unsupported.status_code == 415

    scorer, _ = fitted_record
    no_model_app = create_app(
        store=ProfileStore(tmp_path / "no-model-profiles"),
        backend=client.app.state.services.backend,
        scorer=None,
        upload_dir=tmp_path / "no-model-uploads",
        auto_load_scorer=False,
    )
    no_model = TestClient(no_model_app)
    assert no_model.get("/health").json()["model_ready"] is False
    response = no_model.post(
        "/api/process",
        data={"listener_id": "absent"},
        files={"media": ("speech.wav", b"RIFF", "audio/wav")},
    )
    assert response.status_code == 503
    assert "fitted risk model" in response.json()["detail"]
    assert scorer.model.is_fitted


def test_profile_conflict_and_unsafe_identifier_are_not_silent(api) -> None:
    client, _, _, record = api
    _create_profile(client, record)
    assert (
        client.post("/api/profiles", json=record.hearing.model_dump(mode="json")).status_code == 409
    )
    assert client.get("/api/profiles/..%2Fescape").status_code in {404, 422}
    assert client.get("/api/profiles/.hidden").status_code == 422
    assert client.get("/api/profiles/nobody").status_code == 404


@pytest.mark.parametrize(
    ("policy", "extra", "expected_shown"),
    [("full", {}, 3), ("threshold", {"tau": "1.0"}, 0)],
)
def test_process_supports_full_and_threshold_policies(
    api, policy: str, extra: dict[str, str], expected_shown: int
) -> None:
    client, _, _, record = api
    _create_profile(client, record)
    _calibrate(client, record)

    response = client.post(
        "/api/process",
        data={"listener_id": record.listener_id, "policy": policy, **extra},
        files={"media": ("speech.wav", b"RIFF", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json()["summary"]["n_shown"] == expected_shown


def test_upload_size_limit_is_enforced_before_any_asr_work(
    api, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _, upload_dir, record = api
    _create_profile(client, record)
    _calibrate(client, record)
    monkeypatch.setattr("audire.api.MAX_UPLOAD_BYTES", 4)

    response = client.post(
        "/api/process",
        data={"listener_id": record.listener_id},
        files={"media": ("speech.wav", b"12345", "audio/wav")},
    )

    assert response.status_code == 413
    assert list(upload_dir.glob("*")) == []


class _RaisingBackend(ASRBackend):
    name = "raising"

    def __init__(self, error: Exception) -> None:
        self.error = error

    def transcribe(self, media: Path, *, language: str = "ko") -> Transcript:
        del media, language
        raise self.error

    def describe(self) -> dict[str, str]:
        return {"backend": self.name}


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [(ASRUnavailable("weights unavailable"), 503), (ValueError("invalid media bytes"), 422)],
)
def test_backend_errors_are_actionable_and_upload_is_removed(
    api, fitted_record, tmp_path: Path, error: Exception, expected_status: int
) -> None:
    _, store, _, record = api
    scorer, _ = fitted_record
    store.save_hearing(record.hearing)
    store.save_confusion(record.estimated_confusion)
    uploads = tmp_path / "raising-uploads"
    client = TestClient(
        create_app(
            store=store,
            backend=_RaisingBackend(error),
            scorer=scorer,
            upload_dir=uploads,
        )
    )

    response = client.post(
        "/api/process",
        data={"listener_id": record.listener_id},
        files={"media": ("speech.wav", b"RIFF", "audio/wav")},
    )

    assert response.status_code == expected_status
    assert error.args[0] in response.json()["detail"]
    assert list(uploads.glob("*")) == []
