"""라이브 자막 모드: 브라우저가 이미 그리고 있는 자막만 보고 예측하는 경로.

핵심 원칙은 :mod:`audire.live.contract` 에 있습니다 — 없는 정보는 추측하지 않고 모델의
세계에서 제거합니다.
"""

from audire.live.contract import (
    FEATURE_FAMILIES,
    FORBIDDEN_LIVE_COLUMNS,
    LIVE_ALLOWED_BLOCKS,
    LIVE_CAPTION_V1,
    MEDIA_PIPELINE_V1,
    ContractViolation,
    FeatureFamily,
    LiveAvailability,
    LiveCaptionInputContract,
    assert_contract_compatible,
    availability_report,
    feature_schema_hash,
    get_contract,
)

__all__ = [
    "FEATURE_FAMILIES",
    "FORBIDDEN_LIVE_COLUMNS",
    "LIVE_ALLOWED_BLOCKS",
    "LIVE_CAPTION_V1",
    "MEDIA_PIPELINE_V1",
    "ContractViolation",
    "FeatureFamily",
    "LiveAvailability",
    "LiveCaptionInputContract",
    "assert_contract_compatible",
    "availability_report",
    "feature_schema_hash",
    "get_contract",
]
