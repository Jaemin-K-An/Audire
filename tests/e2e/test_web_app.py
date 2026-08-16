"""Browser E2E for the G6 local application, including its visible failure path."""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import uvicorn

playwright = pytest.importorskip("playwright.sync_api")

from audire.api import create_app  # noqa: E402
from audire.asr import ReplayBackend  # noqa: E402
from audire.eval.ablation import cohort_matrix  # noqa: E402
from audire.profile import ProfileStore  # noqa: E402
from audire.risk import FeatureSpec, LogisticRiskModel, WordScorer  # noqa: E402
from audire.sim import SimulationConfig, build_cohort  # noqa: E402

pytestmark = pytest.mark.e2e

RECORDED = {
    "backend": "faster-whisper",
    "model_id": "small",
    "language": "ko",
    "language_probability": 0.99,
    "duration_s": 2.0,
    "provenance": {"media": "browser-e2e.wav", "recorded_for": "AUDIRE browser E2E"},
    "tokens": [
        {"text": "오늘", "start_s": 0.1, "end_s": 0.5, "confidence": 0.97},
        {"text": "정말", "start_s": 0.7, "end_s": 1.1, "confidence": 0.42},
        {"text": "좋아요", "start_s": 1.3, "end_s": 1.8, "confidence": 0.94},
    ],
}


@pytest.fixture(scope="module")
def scorer() -> WordScorer:
    cfg = SimulationConfig(
        name="browser-e2e",
        n_listeners=16,
        n_calibration_trials=30,
        n_word_trials=40,
        seeds=[47],
    )
    cohort = build_cohort(cfg, 47)
    spec = FeatureSpec.arm("clinical_plus_confusion", speakers=("male", "female", "unknown"))
    return WordScorer(LogisticRiskModel().fit(cohort_matrix(cohort, spec)), spec)


@pytest.fixture
def server_url(tmp_path: Path, scorer: WordScorer) -> Iterator[str]:
    transcript = tmp_path / "recorded.json"
    transcript.write_text(json.dumps(RECORDED, ensure_ascii=False), encoding="utf-8")
    app = create_app(
        store=ProfileStore(tmp_path / "profiles"),
        backend=ReplayBackend(transcript, allow_media_mismatch=True),
        scorer=scorer,
        upload_dir=tmp_path / "uploads",
    )

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", access_log=False)
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10.0
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=2)
        pytest.fail("local E2E server did not start")
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def page() -> Iterator[object]:
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        current = browser.new_page(accept_downloads=True)
        yield current
        browser.close()


def _create_example_profile(page: object, server_url: str) -> None:
    page.goto(server_url)
    page.get_by_role("button", name="입력 예시").click()
    page.get_by_role("button", name="프로필 저장").click()
    playwright.expect(page.locator("#listener")).to_have_value("L001")


def test_browser_happy_path_calibrates_processes_and_downloads(
    page: object, server_url: str
) -> None:
    _create_example_profile(page, server_url)
    page.locator("#calibration-count").fill("10")
    page.get_by_role("button", name="교정 시작").click()
    playwright.expect(page.locator("[data-response]")).to_have_count(10)
    for response in page.locator("[data-response]").all():
        response.fill("가")
    page.get_by_role("button", name="응답 저장").click()
    playwright.expect(page.locator("#listener option")).to_contain_text("교정 10회")

    page.locator('input[name="media"]').set_input_files(
        {"name": "speech.wav", "mimeType": "audio/wav", "buffer": b"RIFF-e2e"}
    )
    page.get_by_role("button", name="선택 자막 생성").click()
    playwright.expect(page.locator("#result")).to_be_visible()
    playwright.expect(page.locator("#words .word")).to_have_count(3)
    assert "전체 3단어" in page.locator("#metrics").inner_text()

    with page.expect_download() as download_info:
        page.get_by_role("button", name="SRT 받기").click()
    assert download_info.value.suggested_filename == "audire-captions.srt"


def test_browser_failure_path_surfaces_missing_calibration(page: object, server_url: str) -> None:
    _create_example_profile(page, server_url)
    page.locator('input[name="media"]').set_input_files(
        {"name": "speech.wav", "mimeType": "audio/wav", "buffer": b"RIFF-e2e"}
    )
    page.get_by_role("button", name="선택 자막 생성").click()

    playwright.expect(page.locator("#message")).to_contain_text("run a calibration first")
    playwright.expect(page.locator("#result")).to_be_hidden()
