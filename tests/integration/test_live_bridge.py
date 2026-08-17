"""Phase 9 — 확장의 실제 JS 클라이언트를 실제 서버에 붙입니다.

왜 이것이 따로 필요한가
-----------------------
지금까지 양쪽은 서로를 **가정**해 왔습니다. 확장 시험은 `fetch` 를 흉내 내고, 서버 시험은
`TestClient` 를 씁니다. 둘 다 통과해도 필드 이름 하나가 어긋나 있으면 아무도 모릅니다.

실제로 어긋나기 쉬운 지점들이 있습니다.

* `ScoreCueRequest` 는 ``extra="forbid"`` 입니다. 클라이언트가 필드를 하나 더 보내면
  즉시 422 입니다.
* 클라이언트는 실패를 ``detail.reason`` 문자열로 갈라냅니다. 서버가 그 문자열을 바꾸면
  모든 실패가 "알 수 없는 오류" 로 뭉개집니다.
* 자바스크립트는 스네이크 케이스를, 파이썬은 카멜 케이스를 모릅니다.

그래서 여기서는 **소켓 위에서** 진짜 `AudireClient` 를 태웁니다. 진짜 HTTP, 진짜 헤더,
진짜 CORS, 진짜 페어링 파일.
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest
import uvicorn

pytestmark = pytest.mark.e2e

EXTENSION = Path(__file__).resolve().parents[2] / "extensions" / "audire-live"
LISTENER = "L001"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def live_server(store, live_scorer):
    """진짜 소켓 위의 AUDIRE 서버. `TestClient` 가 아닙니다."""
    from audire.api import create_app

    app = create_app(store=store, auto_load_scorer=False, live_scorer=live_scorer)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 30
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("the live server did not start in time")
        time.sleep(0.05)

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def _run_client(script: str, base_url: str) -> dict:
    """확장의 실제 클라이언트 모듈을 Node 에서 실행합니다."""
    source = f"""
import {{ AudireClient, LiveState }} from '{EXTENSION}/src/bridge/audireClient.js';
const BASE = {json.dumps(base_url)};
const out = {{}};
{script}
process.stdout.write(JSON.stringify(out));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", source],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise AssertionError(f"the extension client failed:\n{result.stderr}")
    return json.loads(result.stdout)


needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js가 없습니다")


@needs_node
def test_the_extension_client_completes_the_whole_bridge(live_server):
    """페어링 → 프로파일 → 채점 → 연결 해제. 실제 클라이언트, 실제 서버."""
    out = _run_client(
        """
const fresh = new AudireClient({ baseUrl: BASE });
out.beforePairing = (await fresh.profiles()).state;

const paired = await fresh.pair('e2e');
out.pairState = paired.state;
const client = new AudireClient({ baseUrl: BASE, token: paired.body.token });

const status = await client.status();
out.statusState = status.state;
out.inputContract = status.body.input_contract;
out.modelReady = status.body.model_ready;

const profiles = await client.profiles();
out.profileIds = profiles.body.profiles.map((p) => p.id);
out.profileKeys = Object.keys(profiles.body.profiles[0]).sort();

const scored = await client.scoreCue({
  profileId: 'L001',
  cueId: 'c1',
  text: '오늘 병원에 갑니다',
  source: 'local-fixture',
  streamId: 'tab-e2e',
  targetCaptionRate: 0.3,
});
out.scoreState = scored.state;
out.stale = scored.stale;
out.words = scored.body.words.map((w) => w.text);
out.asrConfidence = scored.body.asr_confidence;
out.thresholdPolicy = scored.body.model.threshold_policy;

out.unpairState = (await client.unpair()).state;
out.afterUnpair = (await client.profiles()).state;
""",
        live_server,
    )

    # 페어링 전에는 못 씁니다. 사유가 "알 수 없음" 으로 뭉개지지 않아야 합니다.
    assert out["beforePairing"] == "not_paired"
    assert out["pairState"] == "ok"

    assert out["statusState"] == "ok"
    assert out["inputContract"] == "live-caption-v1"
    assert out["modelReady"] is True

    # 확장 UI 가 기대하는 필드 이름이 그대로여야 합니다.
    assert out["profileIds"] == [LISTENER]
    assert out["profileKeys"] == [
        "alias",
        "calibration_trials",
        "has_hearing_profile",
        "id",
        "ready",
    ]

    assert out["scoreState"] == "ok"
    assert out["stale"] is False
    assert out["words"] == ["오늘", "병원에", "갑니다"]
    assert out["asrConfidence"] is None
    assert out["thresholdPolicy"] == "per_listener"

    # 연결 해제가 서버에서 실제로 지워야 합니다.
    assert out["unpairState"] == "ok"
    assert out["afterUnpair"] == "not_paired"


@needs_node
def test_every_failure_reason_survives_the_wire(live_server):
    """클라이언트의 상태 어휘가 서버의 사유와 실제로 맞물리는지.

    이 목록이 어긋나면 모든 실패가 "알 수 없는 오류" 가 되고, 사용자는 서버를 켜야 할지
    교정을 해야 할지 알 수 없게 됩니다.
    """
    out = _run_client(
        """
const paired = await new AudireClient({ baseUrl: BASE }).pair('e2e');
const client = new AudireClient({ baseUrl: BASE, token: paired.body.token });

out.knownStates = Object.values(LiveState);
out.unknownProfile = (await client.scoreCue({
  profileId: 'nobody', cueId: 'c1', text: '오늘', source: 'x',
})).state;
out.invalidCue = (await client.scoreCue({
  profileId: 'L001', cueId: 'c2', text: '가'.repeat(9000), source: 'x',
})).state;

const wrong = new AudireClient({ baseUrl: BASE, token: 'not-the-token' });
out.badToken = (await wrong.profiles()).state;

const offline = new AudireClient({ baseUrl: 'http://127.0.0.1:1', token: 'x' });
out.offline = (await offline.status()).state;
""",
        live_server,
    )

    assert out["unknownProfile"] == "unknown_profile"
    assert out["invalidCue"] == "invalid_cue"
    assert out["badToken"] == "not_paired"
    assert out["offline"] == "server_offline"
    for reason in ("unknown_profile", "invalid_cue", "not_paired", "server_offline"):
        assert reason in out["knownStates"]


@needs_node
def test_the_client_never_sends_a_field_the_server_forbids(live_server):
    """`ScoreCueRequest` 는 extra="forbid" 입니다.

    클라이언트가 필드를 하나만 더 보내도 전부 422 가 됩니다. 확장 내부 값(`streamId`)이
    본문에 새지 않는지 실물로 확인합니다.
    """
    out = _run_client(
        """
const paired = await new AudireClient({ baseUrl: BASE }).pair('e2e');
const client = new AudireClient({ baseUrl: BASE, token: paired.body.token });
const scored = await client.scoreCue({
  profileId: 'L001', cueId: 'c1', text: '오늘', source: 'x', streamId: 'tab-secret',
});
out.state = scored.state;
""",
        live_server,
    )
    assert out["state"] == "ok", "a forbidden extra field would have produced 422"


@needs_node
def test_a_web_page_origin_is_refused_over_the_real_socket(live_server):
    """브라우저 페이지 흉내. CORS 와 별개로 서버가 스스로 거절해야 합니다."""
    out = _run_client(
        """
const res = await fetch(BASE + '/api/live/status', {
  headers: { origin: 'http://localhost:5173' },
});
out.status = res.status;
out.reason = (await res.json()).detail.reason;
""",
        live_server,
    )
    assert out["status"] == 403
    assert out["reason"] == "origin_not_allowed"
