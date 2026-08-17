"""Local FastAPI application for profiles, calibration and selective captions.

The application is intentionally a thin adapter around the same production functions
used by the experiment suite. Listener data stays in :class:`ProfileStore`; uploaded
media is deleted after each request; and a missing fitted risk model is a visible 503,
never a fallback to mock scores.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from audire.asr import (
    ASRBackend,
    ASRUnavailable,
    FasterWhisperBackend,
    IncompleteProfile,
    caption_media,
    check_ready,
)
from audire.caption import (
    BudgetPolicy,
    CaptionPolicy,
    FullCaptionPolicy,
    ThresholdPolicy,
    to_ass,
    to_json,
    to_srt,
)
from audire.config.logging import get_logger
from audire.config.paths import private_dir, repo_root
from audire.confusion import CalibrationTrial, ConfusionProfile
from audire.data.stimuli import Stimulus, build_balanced_catalog
from audire.profile import HearingProfile, ProfileStore, ProfileStoreError, StoredProfile
from audire.risk import WordScorer
from audire.risk.artifact import DeploymentArtifact, ModelArtifactError

log = get_logger(__name__)

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
SUPPORTED_MEDIA_SUFFIXES = frozenset({".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg"})


def default_model_artifact_path() -> Path:
    configured = os.environ.get("AUDIRE_MODEL_ARTIFACT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (private_dir() / "models" / "audire-logistic.joblib").resolve()


def default_live_artifact_path() -> Path:
    configured = os.environ.get("AUDIRE_LIVE_ARTIFACT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (private_dir() / "models" / "audire-live-caption-v1.joblib").resolve()


def _load_live_scorer() -> Any:
    """라이브 아티팩트를 계약 확인과 함께 읽습니다.

    없거나 계약이 맞지 않으면 ``None`` 을 돌려주고, 라우트가 그 사실을 구분된 실패 상태로
    보고합니다. 미디어 아티팩트를 대신 집어 드는 일은 로더가 막습니다.
    """
    from audire.live.contract import LIVE_CAPTION_V1, ContractViolation
    from audire.live.service import LiveScorer

    path = default_live_artifact_path()
    if not path.exists():
        return None
    try:
        artifact = DeploymentArtifact.load(path, expect_contract=LIVE_CAPTION_V1.version)
        sidecar = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
        metadata = {**artifact.metadata, "artifact_sha256": sidecar.get("artifact_sha256")}
        return LiveScorer(scorer=artifact.scorer, artifact_metadata=metadata)
    except (ModelArtifactError, ContractViolation, OSError, json.JSONDecodeError) as exc:
        log.warning("live.artifact_unavailable", reason=type(exc).__name__)
        return None


def _load_default_scorer() -> WordScorer | None:
    path = default_model_artifact_path()
    if not path.exists():
        return None
    return DeploymentArtifact.load(path).scorer


class CalibrationTrialInput(BaseModel):
    """One response submitted by the local calibration UI."""

    model_config = ConfigDict(extra="forbid")

    stimulus_id: str = Field(min_length=1, max_length=64)
    target: str = Field(min_length=1, max_length=8)
    response: str = Field(max_length=32)
    condition: str = Field(default="default", min_length=1, max_length=64)


class CalibrationSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trials: list[CalibrationTrialInput] = Field(min_length=1, max_length=500)


@dataclass(frozen=True, slots=True)
class AppServices:
    store: ProfileStore
    backend: ASRBackend
    scorer: WordScorer | None
    upload_dir: Path
    #: 라이브 자막 경로용. 미디어 채점기와 **별도**입니다 — 두 경로는 볼 수 있는 정보가
    #: 다르므로 아티팩트를 교차 사용할 수 없습니다(ADR-0021).
    live_scorer: Any = None


@lru_cache(maxsize=1)
def _stimuli() -> tuple[Stimulus, ...]:
    return build_balanced_catalog().stimuli


def _profile_payload(stored: StoredProfile) -> dict[str, Any]:
    confusion = stored.confusion
    return {
        **stored.hearing.summary(),
        "has_calibration": stored.has_calibration,
        "calibration": (
            None
            if confusion is None
            else {
                "n_trials": confusion.n_trials,
                "n_unusable_responses": confusion.n_unusable_responses,
                "overall_accuracy": confusion.overall_accuracy(),
                "coverage": confusion.coverage,
                "is_synthetic": confusion.is_synthetic,
                "provenance": confusion.provenance,
            }
        ),
    }


def _caption_policy(
    name: Literal["full", "budget", "threshold"],
    *,
    budget: float,
    tau: float,
    asr_confidence_floor: float | None,
) -> CaptionPolicy:
    if name == "full":
        return FullCaptionPolicy()
    if name == "budget":
        return BudgetPolicy(budget=budget, asr_confidence_floor=asr_confidence_floor)
    return ThresholdPolicy(tau=tau, asr_confidence_floor=asr_confidence_floor)


def create_app(
    *,
    store: ProfileStore | None = None,
    backend: ASRBackend | None = None,
    scorer: WordScorer | None = None,
    upload_dir: Path | None = None,
    auto_load_scorer: bool = True,
    live_scorer: Any = None,
) -> FastAPI:
    """Build the application with overridable production services.

    When ``scorer`` is omitted, a validated artifact is loaded from
    ``AUDIRE_MODEL_ARTIFACT`` or the private default path. It is never fitted implicitly
    at startup: ``make model`` is the explicit, provenance-recorded build step.
    """

    services = AppServices(
        store=store or ProfileStore(),
        backend=backend or FasterWhisperBackend(),
        scorer=scorer if scorer is not None or not auto_load_scorer else _load_default_scorer(),
        upload_dir=(upload_dir or (private_dir() / "uploads")).resolve(),
        live_scorer=live_scorer
        if live_scorer is not None or not auto_load_scorer
        else _load_live_scorer(),
    )
    application = FastAPI(
        title="AUDIRE",
        version="0.1.0",
        description=(
            "개인화 선택 자막 연구 API. 의료기기가 아니며 진단 또는 임상 효능을 제공하지 않습니다."
        ),
    )
    application.state.services = services

    from audire.live.routes import ALLOWED_ORIGIN_SCHEMES, ALLOWED_ORIGINS, build_live_router

    # CORS 를 좁힙니다. `*` 를 쓰면 같은 기기의 아무 페이지나 로컬 서버를 호출해 프로파일
    # 목록을 읽을 수 있습니다. 확장 출처와 localhost 만 허용합니다.
    application.add_middleware(
        CORSMiddleware,
        allow_origin_regex=(
            r"^(chrome-extension|moz-extension)://[a-z0-9]+$"
            r"|^http://(127\.0\.0\.1|localhost)(:\d+)?$"
        ),
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["content-type", "x-audire-token"],
    )
    application.include_router(build_live_router())
    _ = (ALLOWED_ORIGIN_SCHEMES, ALLOWED_ORIGINS)

    web_root = repo_root() / "apps" / "web"
    if web_root.exists():
        application.mount("/static", StaticFiles(directory=web_root), name="static")

    @application.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        """검증 오류에서 **입력값을 제거**합니다.

        Pydantic 의 기본 응답은 위반한 값을 그대로 실어 보냅니다. 라이브 경로에서 그 값은
        사용자가 보고 있던 자막이며, 이 저장소는 전사 텍스트를 민감 정보로 다룹니다.
        오류를 진단하는 데 필요한 것은 어느 필드가 왜 틀렸는가이지 그 내용이 아닙니다.
        """
        stripped = [
            {k: v for k, v in error.items() if k not in {"input", "ctx", "url"}}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": stripped})

    @application.exception_handler(ProfileStoreError)
    async def profile_store_error(_request: Request, exc: ProfileStoreError) -> JSONResponse:
        code = status.HTTP_404_NOT_FOUND if str(exc).startswith(("no ", "nothing ")) else 422
        return JSONResponse(status_code=code, content={"detail": str(exc)})

    @application.get("/", include_in_schema=False)
    async def web_index() -> FileResponse:
        index = web_root / "index.html"
        if not index.exists():
            raise HTTPException(status_code=503, detail="web application assets are missing")
        return FileResponse(index)

    @application.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_ready": services.scorer is not None and services.scorer.model.is_fitted,
            "model": None if services.scorer is None else services.scorer.describe(),
            "asr_available": services.backend.is_available(),
            "asr": services.backend.describe(),
            "disclaimer": "AUDIRE is research/accessibility software, not a medical device.",
        }

    @application.get("/api/profiles")
    async def list_profiles() -> dict[str, Any]:
        return {
            "profiles": [
                _profile_payload(services.store.load(listener_id))
                for listener_id in services.store.list_ids()
            ]
        }

    @application.post("/api/profiles", status_code=status.HTTP_201_CREATED)
    async def create_profile(profile: HearingProfile) -> dict[str, Any]:
        if services.store.exists(profile.listener_id):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"profile {profile.listener_id!r} already exists; delete it explicitly "
                    "before replacing private listener data"
                ),
            )
        services.store.save_hearing(profile)
        return _profile_payload(services.store.load(profile.listener_id))

    @application.get("/api/profiles/{listener_id}")
    async def get_profile(listener_id: str) -> dict[str, Any]:
        return _profile_payload(services.store.load(listener_id))

    @application.get("/api/profiles/{listener_id}/export")
    async def export_profile(listener_id: str) -> dict[str, Any]:
        return services.store.export(listener_id)

    @application.delete("/api/profiles/{listener_id}")
    async def delete_profile(listener_id: str) -> dict[str, Any]:
        return {"listener_id": listener_id, "removed": services.store.delete(listener_id)}

    @application.get("/api/calibration/stimuli")
    async def calibration_stimuli(
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
    ) -> dict[str, Any]:
        catalog = build_balanced_catalog(limit)
        return {
            "stimuli": [stimulus.to_dict() for stimulus in catalog],
            "provenance": catalog.provenance,
        }

    @application.post("/api/profiles/{listener_id}/calibration")
    async def save_calibration(
        listener_id: str, submission: CalibrationSubmission
    ) -> dict[str, Any]:
        stored = services.store.load(listener_id)
        if stored.hearing.listener_id != listener_id:  # defensive disk-corruption check
            raise HTTPException(status_code=409, detail="stored listener id does not match path")

        stimulus_by_id = {stimulus.stimulus_id: stimulus for stimulus in _stimuli()}
        incoming: list[CalibrationTrial] = []
        incoming_rows: list[dict[str, Any]] = []
        for row in submission.trials:
            stimulus = stimulus_by_id.get(row.stimulus_id)
            if stimulus is None:
                raise HTTPException(
                    status_code=422, detail=f"unknown built-in stimulus {row.stimulus_id!r}"
                )
            if row.target != stimulus.syllable:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"target {row.target!r} does not match catalog target "
                        f"{stimulus.syllable!r} for {row.stimulus_id!r}"
                    ),
                )
            trial = CalibrationTrial(**row.model_dump())
            trial.parse()  # validate before changing the append-only response log
            incoming.append(trial)
            incoming_rows.append(row.model_dump(mode="json"))

        previous_rows = services.store.load_responses(listener_id)
        previous = [
            CalibrationTrial(
                stimulus_id=str(row["stimulus_id"]),
                target=str(row["target"]),
                response=str(row["response"]),
                condition=str(row.get("condition", "default")),
            )
            for row in previous_rows
        ]
        combined = previous + incoming
        profile = ConfusionProfile.from_trials(
            listener_id,
            combined,
            is_synthetic=stored.hearing.is_synthetic,
            provenance={
                "estimated_from": "locally submitted calibration responses",
                "catalog": "audire builtin balanced v1",
                "n_stimuli": len(combined),
                "audio": "browser speech synthesis; not level-calibrated",
            },
        )
        services.store.append_responses(listener_id, incoming_rows)
        services.store.save_confusion(profile)
        return _profile_payload(services.store.load(listener_id))

    @application.post("/api/process")
    async def process_media(
        media: Annotated[UploadFile, File()],
        listener_id: Annotated[str, Form(min_length=1, max_length=64)],
        policy: Annotated[Literal["full", "budget", "threshold"], Form()] = "budget",
        budget: Annotated[float, Form(ge=0.0, le=1.0)] = 0.20,
        tau: Annotated[float, Form(ge=0.0, le=1.0)] = 0.50,
        asr_confidence_floor: Annotated[float | None, Form(ge=0.0, le=1.0)] = None,
        snr_db: Annotated[float, Form(ge=-30.0, le=80.0)] = 20.0,
        speaker: Annotated[str, Form(min_length=1, max_length=64)] = "unknown",
    ) -> dict[str, Any]:
        if services.scorer is None or not services.scorer.model.is_fitted:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "no fitted risk model is configured; provide a validated model artifact "
                    "before processing media"
                ),
            )

        suffix = Path(media.filename or "").suffix.lower()
        if suffix not in SUPPORTED_MEDIA_SUFFIXES:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=(
                    f"unsupported media extension {suffix!r}; "
                    f"allowed: {sorted(SUPPORTED_MEDIA_SUFFIXES)}"
                ),
            )
        stored = services.store.load(listener_id)
        # `listener_id` 를 넘겨 신원 불변식을 이 계층에서 국소적으로 강제한다.
        # 넘기지 않아도 caption_media -> score_transcript 가 결국 잡아 409 가 되지만,
        # 그때는 이 사전 검사가 실제보다 강해 보이는 채로 남고 방어가 우연에 의존한다.
        problems = check_ready(
            services.scorer, stored.hearing, stored.confusion, listener_id=listener_id
        )
        if problems:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="; ".join(problems))

        data = await media.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"upload exceeds {MAX_UPLOAD_BYTES} bytes",
            )

        services.upload_dir.mkdir(parents=True, exist_ok=True)
        temporary = services.upload_dir / f"{uuid4().hex}{suffix}"
        temporary.write_bytes(data)
        try:
            active_policy = _caption_policy(
                policy,
                budget=budget,
                tau=tau,
                asr_confidence_floor=asr_confidence_floor,
            )
            result = caption_media(
                temporary,
                services.backend,
                services.scorer,
                listener_id=listener_id,
                hearing=stored.hearing,
                confusion=stored.confusion,
                policy=active_policy,
                snr_db=snr_db,
                speaker=speaker,
            )
            return {
                "summary": result.summary(),
                "words": [word.to_dict() for word in result.words],
                "exports": {
                    "srt": to_srt(result.words),
                    "ass": to_ass(result.words),
                    "json": to_json(
                        result.words,
                        listener_id=result.listener_id,
                        policy=result.policy,
                        provenance=result.provenance,
                    ),
                },
            }
        except IncompleteProfile as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ASRUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            temporary.unlink(missing_ok=True)
            await media.close()

    return application
