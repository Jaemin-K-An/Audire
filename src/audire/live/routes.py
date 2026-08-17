"""브라우저 확장이 붙는 로컬 라이브 API.

여기서 지켜지는 것
------------------
* **자막 내용은 절대 로그에 남지 않습니다.** 남는 것은 출처·글자 수·단어 수·지연·상태뿐
  입니다. 자막 텍스트는 이 저장소가 이미 민감 정보로 다루는 자료입니다.
* **임상 세부는 확장에 나가지 않습니다.** ``/profiles`` 는 id·별칭·준비 상태만 냅니다.
* **실패 상태가 구분됩니다.** 서버 미가동, 미페어링, 프로파일 없음, 준비 안 됨, 모델 없음,
  계약 불일치가 각각 다른 응답을 냅니다. 추론이 성공한 것처럼 빈 자막을 내보내지 않습니다.
* **전역 임계값을 쓰지 않습니다.** 임계값은 청취자별로 정해집니다(ADR-0021).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from audire.config.logging import get_logger
from audire.live.contract import LIVE_CAPTION_V1, ContractViolation
from audire.live.pairing import (
    PairingError,
    create_pairing,
    load_pairing,
    revoke_pairing,
    verify_token,
)
from audire.live.service import (
    DEFAULT_TARGET_CAPTION_RATE,
    LiveScorer,
    LiveServiceError,
)

log = get_logger(__name__)

#: 확장이 붙을 수 있는 출처. ``*`` 를 쓰지 않습니다 — 같은 기기의 아무 페이지나 로컬
#: 서버를 호출할 수 있으면 프로파일 목록이 새어 나갑니다.
ALLOWED_ORIGIN_SCHEMES = ("chrome-extension://", "moz-extension://")
ALLOWED_ORIGINS = ("http://127.0.0.1", "http://localhost")


def origin_is_allowed(origin: str | None) -> bool:
    if not origin:
        return True  # 확장 배경 스크립트는 Origin 을 붙이지 않을 수 있습니다.
    return origin.startswith(ALLOWED_ORIGIN_SCHEMES) or origin.startswith(ALLOWED_ORIGINS)


class ScoreCueRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str
    cue_id: str
    # 길이 상한을 여기 두지 않습니다. Pydantic 의 max_length 위반 응답은 **입력값을 그대로
    # 되돌려주므로**, 긴 자막이 422 본문으로 새어 나갑니다. 길이 검사는 내용을 인용하지 않는
    # validate_cue 가 합니다.
    text: str = Field(min_length=1)
    source: str = "unknown"
    target_caption_rate: float = Field(default=DEFAULT_TARGET_CAPTION_RATE, gt=0.0, lt=1.0)


class PairRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = "browser-extension"


def build_live_router() -> APIRouter:
    router = APIRouter(prefix="/api/live", tags=["live"])

    def _services(request: Request) -> Any:
        return request.app.state.services

    def _require_pairing(token: str | None) -> None:
        try:
            verify_token(token)
        except PairingError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"reason": "not_paired", "message": str(exc)},
            ) from exc

    def _require_live_scorer(services: Any) -> LiveScorer:
        scorer: LiveScorer | None = getattr(services, "live_scorer", None)
        if scorer is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "reason": "model_unavailable",
                    "message": (
                        "라이브 아티팩트가 없습니다. `make live-model` 로 만들어야 합니다."
                    ),
                },
            )
        return scorer

    @router.get("/status")
    async def live_status(request: Request) -> dict[str, Any]:
        """확장이 처음 묻는 것. 실패 사유가 구분되어야 합니다."""
        services = _services(request)
        scorer = getattr(services, "live_scorer", None)
        pairing = load_pairing()
        return {
            "server": "ok",
            "paired": pairing is not None,
            "pairing": None if pairing is None else pairing.public(),
            "model_ready": scorer is not None,
            "model": None if scorer is None else scorer.describe(),
            "input_contract": LIVE_CAPTION_V1.version,
            "n_profiles": len(services.store.list_ids()),
            "disclaimer": (
                "연구/접근성 소프트웨어이며 의료기기가 아닙니다. 합성 학습 아티팩트이므로 "
                "사람 청취 이득의 근거가 아닙니다."
            ),
        }

    @router.post("/pair", status_code=status.HTTP_201_CREATED)
    async def pair(payload: PairRequest) -> dict[str, Any]:
        """새 페어링 토큰을 만듭니다. 기존 페어링은 무효가 됩니다.

        토큰은 **이 응답에서만** 나갑니다. 이후 어떤 로그·응답에도 실리지 않습니다.
        """
        pairing = create_pairing(payload.label)
        log.info("live.paired", label=payload.label)
        return {"token": pairing.token, **pairing.public()}

    @router.delete("/pair")
    async def unpair() -> dict[str, Any]:
        removed = revoke_pairing()
        log.info("live.unpaired", had_pairing=removed)
        return {"revoked": removed}

    @router.get("/profiles")
    async def live_profiles(
        request: Request, x_audire_token: str | None = Header(default=None)
    ) -> dict[str, Any]:
        """UI 에 필요한 최소 필드만. 청력도·PTA·WRS 원값은 나가지 않습니다."""
        _require_pairing(x_audire_token)
        services = _services(request)
        out = []
        for listener_id in services.store.list_ids():
            stored = services.store.load(listener_id)
            confusion = stored.confusion
            out.append(
                {
                    "id": listener_id,
                    "alias": listener_id,
                    "has_hearing_profile": stored.hearing is not None,
                    "calibration_trials": 0 if confusion is None else confusion.n_trials,
                    "ready": stored.hearing is not None
                    and confusion is not None
                    and confusion.n_trials > 0,
                }
            )
        return {"profiles": out}

    @router.post("/score-cue")
    async def score_cue(
        payload: ScoreCueRequest,
        request: Request,
        x_audire_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _require_pairing(x_audire_token)
        services = _services(request)
        live = _require_live_scorer(services)

        try:
            stored = services.store.load(payload.profile_id)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"reason": "unknown_profile", "message": str(exc)},
            ) from exc

        if stored.hearing is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"reason": "profile_not_ready", "message": "청력 프로파일이 없습니다."},
            )
        if stored.confusion is None or stored.confusion.n_trials == 0:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "reason": "profile_not_ready",
                    "message": "교정이 아직 없습니다. 먼저 교정을 진행하십시오.",
                },
            )

        try:
            result = live.score_cue(
                cue_id=payload.cue_id,
                text=payload.text,
                listener_id=payload.profile_id,
                hearing=stored.hearing,
                confusion=stored.confusion,
                target_rate=payload.target_caption_rate,
            )
        except ContractViolation as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"reason": "contract_mismatch", "message": str(exc)},
            ) from exc
        except LiveServiceError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"reason": "invalid_cue", "message": str(exc)},
            ) from exc

        # 자막 내용은 남기지 않습니다. 길이·개수·지연만 남깁니다.
        log.info(
            "live.cue_scored",
            source=payload.source,
            cue_chars=len(payload.text),
            n_words=len(result.words),
            n_selected=sum(w.selected for w in result.words),
            latency_ms=round(result.latency_ms, 2),
            status="ok",
        )
        return result.to_dict()

    return router
