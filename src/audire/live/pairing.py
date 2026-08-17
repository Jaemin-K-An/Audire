"""로컬 페어링 토큰. 브라우저 확장이 로컬 AUDIRE 에 붙을 때만 쓰입니다.

왜 필요한가
-----------
라이브 엔드포인트는 localhost 에 열립니다. 같은 기기에서 도는 **아무 웹페이지나** 로컬
서버로 요청을 보낼 수 있으므로, CORS 를 좁히는 것만으로는 부족합니다. 페이지가 사용자의
청취자 프로파일 목록을 읽거나 임의의 텍스트를 채점시킬 수 있으면 안 됩니다.

토큰은 한 번 페어링할 때 생성되어 로컬에만 저장되고, 이후 요청의 ``X-Audire-Token`` 으로
확인됩니다. 저장소에 하드코딩된 비밀은 없습니다.

무엇을 하지 않는가
------------------
이것은 **로컬 기기 안에서의 출처 구분**이지 인증 체계가 아닙니다. 같은 사용자 계정으로
기기에 접근할 수 있는 프로그램은 토큰 파일도 읽을 수 있습니다. 그 이상을 주장하지
않습니다.
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from audire.config.paths import private_dir

_FILE_MODE = 0o600
_DIR_MODE = 0o700
#: 토큰 길이(바이트). 로컬 출처 구분에 충분하고 헤더에 넣기 편한 크기입니다.
_TOKEN_BYTES = 32


class PairingError(Exception):
    """페어링이 없거나 토큰이 맞지 않습니다."""


@dataclass(frozen=True, slots=True)
class Pairing:
    """한 번의 페어링 기록. **토큰 값은 로그나 응답에 실리지 않습니다.**"""

    token: str
    created_at_utc: str
    label: str

    def public(self) -> dict[str, Any]:
        """UI 에 보여도 되는 부분만. 토큰은 포함하지 않습니다."""
        return {"paired": True, "created_at_utc": self.created_at_utc, "label": self.label}


def pairing_path() -> Path:
    return private_dir() / "live" / "pairing.json"


def _write_secure(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, _DIR_MODE)  # noqa: PTH101 - 소유자 전용 권한을 명시적으로 강제
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.chmod(tmp, _FILE_MODE)  # noqa: PTH101
        os.replace(tmp, path)  # noqa: PTH105 - 원자적 교체
    finally:
        tmp.unlink(missing_ok=True)


def create_pairing(label: str = "browser-extension") -> Pairing:
    """새 토큰을 만들고 로컬에 저장합니다. 기존 페어링은 이 호출로 무효가 됩니다."""
    pairing = Pairing(
        token=secrets.token_urlsafe(_TOKEN_BYTES),
        created_at_utc=datetime.now(UTC).isoformat(),
        label=label,
    )
    _write_secure(
        pairing_path(),
        {
            "token": pairing.token,
            "created_at_utc": pairing.created_at_utc,
            "label": pairing.label,
        },
    )
    return pairing


def load_pairing() -> Pairing | None:
    path = pairing_path()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return Pairing(
            token=str(payload["token"]),
            created_at_utc=str(payload["created_at_utc"]),
            label=str(payload.get("label", "")),
        )
    except (OSError, json.JSONDecodeError, KeyError):
        # 손상된 페어링 파일은 "페어링 없음" 과 같게 다룹니다. 사용자가 다시 페어링하면
        # 됩니다 — 잘못된 파일로 요청을 통과시키는 것보다 낫습니다.
        return None


def revoke_pairing() -> bool:
    """페어링을 삭제합니다. 삭제할 것이 있었으면 ``True``."""
    path = pairing_path()
    if not path.exists():
        return False
    path.unlink()
    return True


def verify_token(presented: str | None) -> None:
    """제시된 토큰을 확인합니다. 맞지 않으면 :class:`PairingError`.

    비교는 :func:`hmac.compare_digest` 로 합니다. 일반 문자열 비교는 일치하는 앞부분의
    길이에 따라 시간이 달라져 토큰을 한 글자씩 알아낼 여지를 줍니다.
    """
    pairing = load_pairing()
    if pairing is None:
        raise PairingError("이 기기에 페어링이 없습니다. 확장에서 먼저 페어링하십시오.")
    if not presented or not hmac.compare_digest(presented, pairing.token):
        raise PairingError("페어링 토큰이 올바르지 않습니다.")
