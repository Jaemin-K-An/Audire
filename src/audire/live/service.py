"""라이브 자막 채점 서비스.

브라우저가 이미 그리고 있는 자막 큐 하나를 받아, 그 청취자에게 위험한 단어만 골라
돌려줍니다.

지켜지는 제약
-------------
* **음향 맥락 없음.** ``live-caption-v1`` 계약 아래에서만 동작하며, SNR·화자를 지어내지
  않습니다(ADR-0021).
* **전역 임계값 금지.** 임계값은 청취자마다 다르고, 그 청취자의 프로파일로 고정된 참조
  어휘를 채점해 미리 정합니다. E30 에서 전역 임계값이 normal 청취자에게 자막률 0.0004 를
  주고 중앙값 청취자의 재현율을 0 으로 만든다는 것이 측정되었습니다.
* **자막 내용은 기록하지 않습니다.** 로그와 지표에는 길이·개수·지연만 남습니다.
* **ASR 신뢰도 없음.** DOM 자막에는 인식기가 없으므로 ``None`` 입니다.
"""

from __future__ import annotations

import hashlib
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from audire.confusion.profile import ConfusionProfile
from audire.hangul.syllable import is_hangul_syllable
from audire.live.contract import LIVE_CAPTION_V1, ContractViolation, feature_schema_hash
from audire.live.reference import (
    REFERENCE_VOCABULARY_VERSION,
    build_reference_vocabulary,
    reference_provenance,
)
from audire.profile.schema import HearingProfile
from audire.risk.features import WordContext
from audire.risk.models import WordScorer

#: 한 큐가 가질 수 있는 최대 글자 수. 페이지 내용은 신뢰할 수 없는 입력이므로 상한을 둡니다.
MAX_CUE_CHARS = 500
#: 한 큐의 최대 토큰 수.
MAX_CUE_TOKENS = 60
#: 기본 목표 자막률. 사용자가 조절할 수 있습니다.
DEFAULT_TARGET_CAPTION_RATE = 0.20


class LiveServiceError(Exception):
    """라이브 채점을 진행할 수 없는 상태. 사유가 구분되어야 합니다."""


@dataclass(frozen=True, slots=True)
class ScoredWord:
    text: str
    risk: float
    selected: bool


@dataclass(frozen=True, slots=True)
class CueResult:
    cue_id: str
    words: tuple[ScoredWord, ...]
    display_text: str
    threshold: float
    latency_ms: float
    model: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cue_id": self.cue_id,
            "words": [{"text": w.text, "risk": w.risk, "selected": w.selected} for w in self.words],
            "display_text": self.display_text,
            "threshold": self.threshold,
            "latency_ms": self.latency_ms,
            "model": self.model,
            # DOM 자막에는 인식기가 없습니다. 값을 지어내지 않습니다.
            "asr_confidence": None,
        }


def normalise_cue(text: str) -> str:
    """자막 텍스트를 채점 가능한 형태로 정규화합니다.

    NFC 로 통일하고 공백을 접습니다. macOS 경로를 거친 한글은 NFD 로 오며, 그 상태로는
    같은 단어가 다른 문자열이 되어 캐시와 채점이 갈라집니다.
    """
    return " ".join(unicodedata.normalize("NFC", text).split())


def validate_cue(text: str) -> str:
    """신뢰할 수 없는 페이지 입력을 검사합니다.

    HTML 을 해석하지 않습니다 — 들어온 것은 끝까지 텍스트이며, 렌더러도 ``textContent``
    로만 씁니다.
    """
    if not isinstance(text, str):
        raise LiveServiceError("cue text must be a string")
    normalised = normalise_cue(text)
    if not normalised:
        raise LiveServiceError("cue is empty")
    if len(normalised) > MAX_CUE_CHARS:
        raise LiveServiceError(f"cue is too long: {len(normalised)} chars (max {MAX_CUE_CHARS})")
    if len(normalised.split()) > MAX_CUE_TOKENS:
        raise LiveServiceError(f"cue has too many tokens (max {MAX_CUE_TOKENS})")
    return normalised


def profile_digest(hearing: HearingProfile, confusion: ConfusionProfile | None) -> str:
    """프로파일이 바뀌면 값이 바뀌는 다이제스트. 캐시 무효화와 임계값 재계산에 씁니다."""
    h = hashlib.sha256()
    h.update(hearing.model_dump_json().encode("utf-8"))
    if confusion is not None:
        h.update(b"\0")
        for position in sorted(confusion.matrices, key=lambda p: p.value):
            h.update(confusion.matrix(position).counts.tobytes())
    return h.hexdigest()[:16]


@dataclass
class LiveScorer:
    """계약이 확인된 아티팩트로 큐를 채점합니다.

    임계값은 청취자마다 **한 번** 계산되어 재사용됩니다. 큐마다 다시 계산하면 같은 청취자가
    문장에 따라 다른 기준을 받게 되어, 자막량이 문장 길이에 따라 출렁입니다.
    """

    scorer: WordScorer
    artifact_metadata: dict[str, Any]
    #: (프로파일 다이제스트, 목표 자막률) -> 임계값.
    _thresholds: dict[tuple[str, float], float] = field(default_factory=dict, repr=False)
    #: (아티팩트 다이제스트, 프로파일 다이제스트, 단어) -> 위험도.
    _word_cache: dict[tuple[str, str, str], float] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        contract = str(self.artifact_metadata.get("input_contract", ""))
        if contract != LIVE_CAPTION_V1.version:
            raise ContractViolation(
                f"라이브 경로는 {LIVE_CAPTION_V1.version} 아티팩트를 요구하는데 "
                f"{contract!r} 가 주어졌습니다."
            )
        names = self.scorer.spec_feature_names()
        LIVE_CAPTION_V1.validate_columns(names)
        declared = self.artifact_metadata.get("feature_schema_hash")
        actual = feature_schema_hash(contract, names)
        if declared != actual:
            raise ContractViolation(
                f"특징 스키마 불일치: 아티팩트 {declared!r}, 실행 환경 {actual!r}"
            )

    @property
    def artifact_digest(self) -> str:
        return str(self.artifact_metadata.get("artifact_sha256", "unknown"))[:16]

    def _score_words(
        self,
        words: list[str],
        listener_id: str,
        hearing: HearingProfile,
        confusion: ConfusionProfile | None,
    ) -> np.ndarray:
        # WordContext 는 인터페이스상 필요하지만 라이브 arm 은 그 값을 쓰지 않습니다.
        # 계약 테스트가 그 사실을 고정합니다.
        contexts = [WordContext(snr_db=0.0, speaker="unknown")] * len(words)
        return np.asarray(
            self.scorer.score(listener_id, words, contexts, hearing, confusion),
            dtype=np.float64,
        )

    def threshold_for(
        self,
        listener_id: str,
        hearing: HearingProfile,
        confusion: ConfusionProfile | None,
        target_rate: float = DEFAULT_TARGET_CAPTION_RATE,
    ) -> float:
        """이 청취자의 임계값.

        전역 임계값을 쓰지 않기 위해, 이 청취자의 프로파일로 고정 참조 어휘를 채점하고
        그 분포의 분위수를 씁니다. 큐가 도착하기 전에 정해지므로 상태가 없고 결정적입니다.
        """
        if not 0.0 < target_rate < 1.0:
            raise LiveServiceError("target caption rate must be in (0, 1)")
        key = (profile_digest(hearing, confusion), round(target_rate, 4))
        if key not in self._thresholds:
            reference = list(build_reference_vocabulary())
            scores = self._score_words(reference, listener_id, hearing, confusion)
            self._thresholds[key] = float(np.quantile(scores, 1.0 - target_rate))
        return self._thresholds[key]

    def score_cue(
        self,
        *,
        cue_id: str,
        text: str,
        listener_id: str,
        hearing: HearingProfile,
        confusion: ConfusionProfile | None,
        target_rate: float = DEFAULT_TARGET_CAPTION_RATE,
    ) -> CueResult:
        started = time.perf_counter()
        normalised = validate_cue(text)
        tokens = normalised.split()

        threshold = self.threshold_for(listener_id, hearing, confusion, target_rate)
        digest = profile_digest(hearing, confusion)

        # 한글이 없는 토큰(숫자, 라틴 문자)은 한국어 음소 혼동 프로파일로 채점할 수
        # 없습니다. 지어내지 않고 위험도 0 으로 두되 선택하지 않습니다.
        scoreable = {t for t in tokens if any(is_hangul_syllable(c) for c in t)}
        uncached = [
            t
            for t in sorted(scoreable)
            if (self.artifact_digest, digest, t) not in self._word_cache
        ]
        if uncached:
            values = self._score_words(uncached, listener_id, hearing, confusion)
            for token, value in zip(uncached, values, strict=True):
                self._word_cache[(self.artifact_digest, digest, token)] = float(value)

        words: list[ScoredWord] = []
        for token in tokens:
            if token in scoreable:
                risk = self._word_cache[(self.artifact_digest, digest, token)]
                words.append(ScoredWord(text=token, risk=risk, selected=risk >= threshold))
            else:
                words.append(ScoredWord(text=token, risk=0.0, selected=False))

        latency_ms = (time.perf_counter() - started) * 1000.0
        return CueResult(
            cue_id=cue_id,
            words=tuple(words),
            display_text=" ".join(w.text for w in words if w.selected),
            threshold=threshold,
            latency_ms=latency_ms,
            model=self.describe(),
        )

    def describe(self) -> dict[str, Any]:
        meta = self.artifact_metadata
        return {
            "family": meta.get("model", "logistic"),
            "artifact_sha256": meta.get("artifact_sha256"),
            "input_contract": meta.get("input_contract"),
            "feature_schema_hash": meta.get("feature_schema_hash"),
            "arm": meta.get("arm"),
            "training_source": meta.get("training_source"),
            "simulator_version": meta.get("simulator_version"),
            "human_efficacy_evidence": meta.get("human_efficacy_evidence", False),
            "reference_vocabulary": REFERENCE_VOCABULARY_VERSION,
            "threshold_policy": "per_listener",
            "reference_provenance": reference_provenance(),
        }
