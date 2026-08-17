"""``live-caption-v1`` — 브라우저가 이미 그리고 있는 자막만 볼 때의 입력 계약.

왜 별도 계약이 필요한가
-----------------------
배포 모델(`clinical_plus_confusion`, 50열)은 음향 맥락 블록을 씁니다.

    c_snr_db, c_speaker_male, c_speaker_female, c_speaker_unknown

브라우저 DOM 자막은 **텍스트만** 줍니다. 신호대잡음비도, 화자 정체도, 측정된 ASR 신뢰도도
없습니다. 그 값들을 채워 넣으면 측정하지 않은 것을 측정한 것처럼 보고하게 됩니다.

특히 ``c_snr_db = 20`` 을 조용히 넣는 것은 금지입니다. E25 에서 전체 개인화 이득의
6.4–13.7배가 **조건 간 예산 배분**에서 왔고 조건 내부 이득은 그 1/10 이었습니다
(docs/RESULTS.md §13). 즉 SNR 은 이 모델에서 장식이 아니라 이득의 주된 출처이며, 그것을
상수로 채우면 모델이 학습한 분포 밖에서 돌게 됩니다.

따라서 원칙은 하나입니다.

    없는 정보는 **추측하지 말고 모델의 세계에서 제거한다.**

라이브 모델은 추론 시 갖게 될 것과 **같은 정보 제약 아래에서 학습**됩니다.

이 모듈이 정하는 것
-------------------
* 어떤 특징 계열이 라이브에서 정직하게 얻어지는가(:class:`LiveAvailability`).
* 계약 버전과 그 아래 허용/금지되는 것.
* 아티팩트와 실행 환경의 계약·스키마 일치 검사.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class LiveAvailability(StrEnum):
    """브라우저 DOM 자막 모드에서 이 특징을 정직하게 얻을 수 있는가."""

    #: 현재 자막 텍스트 또는 저장된 청취자 프로파일에서 바로 나옵니다.
    AVAILABLE = "live_available"
    #: 어떤 환경에서는 존재할 수 있으나 가정할 수 없습니다.
    OPTIONAL = "live_optional"
    #: 날조 없이는 얻을 수 없습니다.
    UNAVAILABLE = "live_unavailable"


@dataclass(frozen=True, slots=True)
class FeatureFamily:
    """한 특징 블록의 라이브 가용성과 그 판단 근거."""

    block: str
    prefix: str
    availability: LiveAvailability
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "block": self.block,
            "prefix": self.prefix,
            "availability": self.availability.value,
            "reason": self.reason,
        }


#: 배포 경로가 쓰는 모든 특징 계열의 분류. 근거를 함께 적어 두어야 나중에 "왜 뺐는가" 를
#: 다시 논증하지 않아도 됩니다.
FEATURE_FAMILIES: tuple[FeatureFamily, ...] = (
    FeatureFamily(
        block="word",
        prefix="w_",
        availability=LiveAvailability.AVAILABLE,
        reason=(
            "자막 텍스트 자체에서 나옵니다. 음절 수, 자모 구성, 종성 비율, 음절 구조는 "
            "표시된 문자열만으로 결정됩니다."
        ),
    ),
    FeatureFamily(
        block="context",
        prefix="c_",
        availability=LiveAvailability.UNAVAILABLE,
        reason=(
            "c_snr_db 는 측정된 신호대잡음비이고 c_speaker_* 는 화자 정체입니다. DOM 자막은 "
            "둘 다 주지 않습니다. 상수로 채우면 측정하지 않은 값을 측정한 것처럼 보고하게 "
            "되며, E25 에 따르면 이 블록이 개인화 이득의 주된 출처이므로 그 대체는 결과를 "
            "조용히 무의미하게 만듭니다."
        ),
    ),
    FeatureFamily(
        block="pta",
        prefix="h_",
        availability=LiveAvailability.AVAILABLE,
        reason="저장된 청력 프로파일에서 읽습니다. 추론 시점에 이미 존재합니다.",
    ),
    FeatureFamily(
        block="clinical",
        prefix="h_",
        availability=LiveAvailability.AVAILABLE,
        reason="PTA/SRT/WRS 등 저장된 임상 측정값입니다. 자막과 무관하게 이미 있습니다.",
    ),
    FeatureFamily(
        block="confusion",
        prefix="x_",
        availability=LiveAvailability.AVAILABLE,
        reason=(
            "교정으로 만들어 저장해 둔 혼동 프로파일과 현재 단어에서 계산됩니다. 새 음향 "
            "정보를 필요로 하지 않습니다."
        ),
    ),
    FeatureFamily(
        block="confusion_rich",
        prefix="ix_/w2_/x2_",
        availability=LiveAvailability.AVAILABLE,
        reason=(
            "혼동 프로파일과 단어만 씁니다. 다만 E28 에서 이 블록이 청취자 내 순위를 "
            "악화시켰으므로 라이브 초기 arm 에는 넣지 않습니다 — 가용성 문제가 아니라 "
            "성능 근거에 따른 제외입니다."
        ),
    ),
    FeatureFamily(
        block="exact_target",
        prefix="et_",
        availability=LiveAvailability.AVAILABLE,
        reason="confusion_rich 와 같습니다. 가용하지만 E28 근거로 제외합니다.",
    ),
    FeatureFamily(
        block="exact_target_offdiag",
        prefix="eo_",
        availability=LiveAvailability.AVAILABLE,
        reason="confusion_rich 와 같습니다. 가용하지만 E28 근거로 제외합니다.",
    ),
)

#: 라이브 모드에서 결코 존재해서는 안 되는 열 이름. 실수로 새어 들어오면 즉시 실패합니다.
FORBIDDEN_LIVE_COLUMNS: frozenset[str] = frozenset(
    {"c_snr_db", "c_speaker_male", "c_speaker_female", "c_speaker_unknown"}
)

#: 라이브 arm 이 쓸 수 있는 블록. ``context`` 가 없다는 것이 이 계약의 핵심입니다.
LIVE_ALLOWED_BLOCKS: frozenset[str] = frozenset(
    {
        "word",
        "pta",
        "clinical",
        "confusion",
        "confusion_rich",
        "exact_target",
        "exact_target_offdiag",
    }
)


class ContractViolation(Exception):
    """입력 계약을 어긴 특징·아티팩트·요청."""


@dataclass(frozen=True, slots=True)
class LiveCaptionInputContract:
    """라이브 자막 모드에서 모델이 볼 수 있는 것의 명세."""

    version: str
    source: str
    #: 음향 맥락(SNR 등)을 볼 수 있는가. 라이브에서는 항상 False 입니다.
    acoustic_context: bool
    #: 측정된 ASR 신뢰도가 붙는가. DOM 자막에는 인식기가 없으므로 False 입니다.
    asr_confidence: bool
    #: 화자 정체가 필요한가. 필요하다고 두면 날조 유인이 생깁니다.
    speaker_required: bool
    allowed_blocks: frozenset[str] = field(default_factory=lambda: LIVE_ALLOWED_BLOCKS)
    forbidden_columns: frozenset[str] = field(default_factory=lambda: FORBIDDEN_LIVE_COLUMNS)

    def validate_columns(self, feature_names: Sequence[str]) -> None:
        """열 목록이 계약을 지키는지 확인합니다. 어기면 :class:`ContractViolation`."""
        leaked = sorted(set(feature_names) & self.forbidden_columns)
        if leaked:
            raise ContractViolation(
                f"{self.version} 아래에서는 존재할 수 없는 열이 들어왔습니다: {leaked}. "
                f"라이브 모드에는 음향 맥락이 없으므로 이 값들은 채워 넣는 것이 아니라 "
                f"모델의 세계에서 제거되어야 합니다."
            )
        # 접두사로도 한 번 더 봅니다. 새 음향 특징이 추가되어도 계약이 먼저 막습니다.
        # 음향 맥락을 허용하는 계약(미디어 경로)에는 적용하지 않습니다 — 그쪽에서는
        # c_ 열이 정상입니다.
        if self.acoustic_context:
            return
        acoustic = sorted(n for n in feature_names if n.startswith("c_"))
        if acoustic:
            raise ContractViolation(
                f"{self.version} 아래에서 음향 맥락 접두사 'c_' 열이 발견되었습니다: {acoustic}"
            )

    def validate_blocks(self, blocks: Sequence[str]) -> None:
        unknown = sorted(set(blocks) - self.allowed_blocks)
        if unknown:
            raise ContractViolation(
                f"{self.version} 이 허용하지 않는 특징 블록: {unknown}. "
                f"허용: {sorted(self.allowed_blocks)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source": self.source,
            "acoustic_context": self.acoustic_context,
            "asr_confidence": self.asr_confidence,
            "speaker_required": self.speaker_required,
            "allowed_blocks": sorted(self.allowed_blocks),
            "forbidden_columns": sorted(self.forbidden_columns),
        }


#: 이 프로젝트의 라이브 계약 1판.
LIVE_CAPTION_V1 = LiveCaptionInputContract(
    version="live-caption-v1",
    source="browser_dom_caption",
    acoustic_context=False,
    asr_confidence=False,
    speaker_required=False,
)

#: 기존 미디어/ASR 경로가 쓰는 계약. 이름을 붙여 두어야 아티팩트를 서로 바꿔 끼울 수
#: 없습니다.
MEDIA_PIPELINE_V1 = LiveCaptionInputContract(
    version="media-pipeline-v1",
    source="asr_media",
    acoustic_context=True,
    asr_confidence=True,
    speaker_required=False,
    allowed_blocks=frozenset({*LIVE_ALLOWED_BLOCKS, "context"}),
    forbidden_columns=frozenset(),
)

_CONTRACTS = {c.version: c for c in (LIVE_CAPTION_V1, MEDIA_PIPELINE_V1)}


def get_contract(version: str) -> LiveCaptionInputContract:
    try:
        return _CONTRACTS[version]
    except KeyError as exc:
        raise ContractViolation(
            f"알 수 없는 입력 계약 {version!r}; 알려진 계약: {sorted(_CONTRACTS)}"
        ) from exc


def feature_schema_hash(contract_version: str, feature_names: Sequence[str]) -> str:
    """계약 버전과 **순서가 있는** 열 이름에 대한 안정적 다이제스트.

    아티팩트를 정확한 스키마에 묶습니다. 열이 하나라도 바뀌거나 순서가 달라지면 값이
    달라지므로, 로드 시점에 실행 환경과 대조해 조용한 불일치를 막습니다. 순서까지 포함하는
    이유는 선형 모델의 계수가 열 순서에 묶여 있기 때문입니다.
    """
    h = hashlib.sha256()
    h.update(contract_version.encode("utf-8"))
    h.update(b"\0")
    for name in feature_names:
        h.update(name.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def assert_contract_compatible(artifact_version: str, runtime_version: str) -> None:
    """아티팩트가 선언한 계약과 지금 쓰려는 계약이 같은가.

    파일이 존재한다는 이유만으로 모델을 고르면, 음향 맥락을 기대하는 모델이 그것이 없는
    경로에서 조용히 돌게 됩니다. 그 경우 출력은 그럴듯하지만 학습 분포 밖입니다.
    """
    if artifact_version != runtime_version:
        raise ContractViolation(
            f"입력 계약 불일치: 아티팩트는 {artifact_version!r} 로 학습되었는데 "
            f"{runtime_version!r} 경로에서 쓰려 합니다. 두 경로는 볼 수 있는 정보가 다르므로 "
            f"교차 사용할 수 없습니다."
        )


def availability_report() -> list[dict[str, str]]:
    """특징 계열별 라이브 가용성 표. 보고서와 테스트가 함께 씁니다."""
    return [family.to_dict() for family in FEATURE_FAMILIES]
