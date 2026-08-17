"""Phase 4 — ``live-caption-v1`` 입력 계약의 불변식.

브라우저 DOM 자막은 텍스트만 줍니다. 음향 맥락도, 화자 정체도, 측정된 ASR 신뢰도도
없습니다. 여기서 고정하는 원칙은 하나입니다 — **없는 정보는 추측하지 않고 모델의 세계에서
제거한다.** 그것을 상수로 채우면 측정하지 않은 값을 측정한 것처럼 보고하게 됩니다.
"""

from __future__ import annotations

import math

import pytest

from audire.confusion import CalibrationTrial, ConfusionProfile
from audire.live import (
    FEATURE_FAMILIES,
    FORBIDDEN_LIVE_COLUMNS,
    LIVE_CAPTION_V1,
    MEDIA_PIPELINE_V1,
    ContractViolation,
    LiveAvailability,
    assert_contract_compatible,
    availability_report,
    feature_schema_hash,
    get_contract,
)
from audire.risk import ABLATION_ARMS, FeatureSpec, WordContext

LIVE_ARMS = (
    "live_word_context",
    "live_word_context_clinical",
    "live_word_context_clinical_confusion",
)
SPEAKERS = ("male", "female", "unknown")


@pytest.fixture(scope="module")
def hearing():
    from audire.profile.schema import (
        Audiogram,
        AudiogramPoint,
        Ear,
        EarProfile,
        HearingProfile,
        ProfileSource,
        SpeechScores,
    )

    ear = EarProfile(
        ear=Ear.RIGHT,
        audiogram=Audiogram(
            ear=Ear.RIGHT,
            thresholds={f: AudiogramPoint(db_hl=40.0) for f in (500, 1000, 2000, 4000)},
        ),
        speech=SpeechScores(
            ear=Ear.RIGHT, srt_db_hl=40.0, wrs_percent=70.0, wrs_presentation_level_db_hl=70.0
        ),
    )
    return HearingProfile(
        listener_id="L1", source=ProfileSource.SYNTHETIC, is_synthetic=True, right=ear
    )


@pytest.fixture(scope="module")
def confusion():
    return ConfusionProfile.from_trials(
        "L1",
        [CalibrationTrial(stimulus_id="s0", target="각", response="각")],
        is_synthetic=True,
    )


def _row(arm: str, hearing, confusion) -> dict[str, float]:
    spec = FeatureSpec.arm(arm, speakers=SPEAKERS)
    return spec.row("가족", WordContext(snr_db=20.0, speaker="male"), hearing, confusion)


# ------------------------------------------------------------------------ 계약 자체


def test_contract_version_is_stable():
    """버전이 조용히 바뀌면 기록된 아티팩트의 의미가 달라집니다."""
    assert LIVE_CAPTION_V1.version == "live-caption-v1"
    assert LIVE_CAPTION_V1.source == "browser_dom_caption"


def test_live_contract_declares_no_acoustic_context():
    assert LIVE_CAPTION_V1.acoustic_context is False


def test_live_contract_declares_no_asr_confidence():
    """DOM 자막에는 인식기가 없습니다. 신뢰도를 지어내면 안 됩니다."""
    assert LIVE_CAPTION_V1.asr_confidence is False


def test_live_contract_does_not_require_a_speaker():
    """화자를 필수로 두면 날조 유인이 생깁니다."""
    assert LIVE_CAPTION_V1.speaker_required is False


def test_snr_is_forbidden_by_name_and_by_prefix():
    """이름만 막으면 새 음향 특징이 추가될 때 계약이 뚫립니다."""
    assert "c_snr_db" in FORBIDDEN_LIVE_COLUMNS
    with pytest.raises(ContractViolation, match="c_snr_db"):
        LIVE_CAPTION_V1.validate_columns(["w_n_syllables", "c_snr_db"])
    # 아직 존재하지 않는 음향 열도 접두사로 막혀야 합니다.
    with pytest.raises(ContractViolation, match="c_"):
        LIVE_CAPTION_V1.validate_columns(["w_n_syllables", "c_reverberation_time"])


def test_speaker_one_hots_are_forbidden():
    with pytest.raises(ContractViolation):
        LIVE_CAPTION_V1.validate_columns(["c_speaker_male"])


def test_context_block_is_not_an_allowed_live_block():
    assert "context" not in LIVE_CAPTION_V1.allowed_blocks
    with pytest.raises(ContractViolation, match="context"):
        LIVE_CAPTION_V1.validate_blocks(["word", "context"])


def test_unknown_block_is_rejected():
    with pytest.raises(ContractViolation, match="not_a_block"):
        LIVE_CAPTION_V1.validate_blocks(["word", "not_a_block"])


def test_media_contract_still_allows_acoustic_context():
    """기존 미디어 경로의 의미론은 바뀌지 않아야 합니다."""
    assert MEDIA_PIPELINE_V1.acoustic_context is True
    assert "context" in MEDIA_PIPELINE_V1.allowed_blocks
    MEDIA_PIPELINE_V1.validate_columns(["c_snr_db"])  # 예외 없이 통과


def test_unknown_contract_version_is_rejected():
    with pytest.raises(ContractViolation, match="알 수 없는 입력 계약"):
        get_contract("live-caption-v99")


# --------------------------------------------------------- 가용성 분류 (감사 표)


def test_every_deployment_block_is_classified():
    """분류되지 않은 블록이 있으면 '검토했다' 는 진술이 거짓이 됩니다."""
    classified = {f.block for f in FEATURE_FAMILIES}
    for arm_blocks in ABLATION_ARMS.values():
        for block in arm_blocks:
            assert block in classified, f"{block} 이 가용성 분류에 없습니다"


def test_context_is_the_only_unavailable_family():
    unavailable = [
        f.block for f in FEATURE_FAMILIES if f.availability is LiveAvailability.UNAVAILABLE
    ]
    assert unavailable == ["context"]


def test_every_family_states_its_reason():
    """근거 없이 분류하면 나중에 왜 뺐는지 다시 논증해야 합니다."""
    for family in FEATURE_FAMILIES:
        assert family.reason.strip(), family.block
    assert len(availability_report()) == len(FEATURE_FAMILIES)


# --------------------------------------------------------------- 라이브 arm 의 스키마


@pytest.mark.parametrize("arm", LIVE_ARMS)
def test_live_arms_satisfy_the_contract(arm, hearing, confusion):
    row = _row(arm, hearing, confusion)
    LIVE_CAPTION_V1.validate_columns(list(row))
    LIVE_CAPTION_V1.validate_blocks(ABLATION_ARMS[arm])


@pytest.mark.parametrize("arm", LIVE_ARMS)
def test_live_arms_never_contain_an_acoustic_column(arm, hearing, confusion):
    assert not [n for n in _row(arm, hearing, confusion) if n.startswith("c_")]


def test_the_deployment_arm_is_rejected_by_the_live_contract(hearing, confusion):
    """배포 모델을 그대로 라이브에 끼워 넣는 경로가 막혀 있어야 합니다."""
    row = _row("clinical_plus_confusion", hearing, confusion)
    assert "c_snr_db" in row
    with pytest.raises(ContractViolation):
        LIVE_CAPTION_V1.validate_columns(list(row))


def test_live_arms_are_nested_so_the_ablation_is_interpretable():
    """각 arm 이 앞의 것을 포함해야 차이가 곧 그 블록의 몫이 됩니다."""
    live0, live1, live2 = (set(ABLATION_ARMS[a]) for a in LIVE_ARMS)
    assert live0 < live1 < live2
    assert live0 == {"word"}
    assert live1 == {"word", "clinical"}
    assert live2 == {"word", "clinical", "confusion"}


def test_clinical_block_appears_only_in_live1_and_live2():
    assert "clinical" not in ABLATION_ARMS["live_word_context"]
    assert "clinical" in ABLATION_ARMS["live_word_context_clinical"]
    assert "clinical" in ABLATION_ARMS["live_word_context_clinical_confusion"]


def test_confusion_block_appears_only_in_live2():
    assert "confusion" not in ABLATION_ARMS["live_word_context"]
    assert "confusion" not in ABLATION_ARMS["live_word_context_clinical"]
    assert "confusion" in ABLATION_ARMS["live_word_context_clinical_confusion"]


def test_rich_blocks_are_excluded_from_the_initial_live_arms():
    """E28 에서 이 블록들이 청취자 내 순위를 악화시켰습니다. 가용성이 아니라 근거의 문제입니다."""
    for arm in LIVE_ARMS:
        blocks = set(ABLATION_ARMS[arm])
        assert not blocks & {"confusion_rich", "exact_target", "exact_target_offdiag"}


@pytest.mark.parametrize("arm", LIVE_ARMS)
def test_live_feature_order_is_deterministic(arm, hearing, confusion):
    """선형 모델의 계수가 열 순서에 묶여 있으므로 순서가 흔들리면 아티팩트가 깨집니다."""
    assert list(_row(arm, hearing, confusion)) == list(_row(arm, hearing, confusion))


def test_live_arms_do_not_depend_on_the_speaker_argument(hearing, confusion):
    """화자 목록이 달라져도 라이브 스키마가 변하면 안 됩니다 — 화자를 쓰지 않으니까요."""
    a = FeatureSpec.arm("live_word_context_clinical_confusion", speakers=("male",))
    b = FeatureSpec.arm("live_word_context_clinical_confusion", speakers=("male", "female", "x"))
    context = WordContext(snr_db=20.0, speaker="male")
    assert list(a.row("가족", context, hearing, confusion)) == list(
        b.row("가족", context, hearing, confusion)
    )


def test_live_schema_ignores_the_supplied_snr_value(hearing, confusion):
    """SNR 을 넘겨도 라이브 arm 의 값이 달라지면, 어딘가로 새고 있다는 뜻입니다."""
    spec = FeatureSpec.arm("live_word_context_clinical_confusion", speakers=SPEAKERS)
    quiet = spec.row("가족", WordContext(snr_db=20.0, speaker="male"), hearing, confusion)
    noisy = spec.row("가족", WordContext(snr_db=-5.0, speaker="female"), hearing, confusion)
    # 결측 임상값은 NaN 이라 dict 비교가 실패합니다. NaN 은 NaN 과 같은 자리로 봅니다.
    assert list(quiet) == list(noisy)
    for key in quiet:
        a, b = quiet[key], noisy[key]
        assert (math.isnan(a) and math.isnan(b)) or a == b, key


# ------------------------------------------------------------------- 스키마 다이제스트


def test_schema_hash_is_stable_and_order_sensitive():
    names = ["w_a", "w_b", "h_c"]
    assert feature_schema_hash("live-caption-v1", names) == feature_schema_hash(
        "live-caption-v1", names
    )
    # 순서가 바뀌면 다른 모델입니다.
    assert feature_schema_hash("live-caption-v1", names) != feature_schema_hash(
        "live-caption-v1", ["w_b", "w_a", "h_c"]
    )


def test_schema_hash_separates_contracts():
    names = ["w_a"]
    assert feature_schema_hash("live-caption-v1", names) != feature_schema_hash(
        "media-pipeline-v1", names
    )


def test_schema_hash_changes_when_a_column_is_added():
    assert feature_schema_hash("live-caption-v1", ["w_a"]) != feature_schema_hash(
        "live-caption-v1", ["w_a", "w_b"]
    )


# --------------------------------------------------------------- 계약 교차 사용 금지


def test_compatible_contracts_pass():
    # 같은 계약이면 조용히 통과해야 합니다 (예외 없음).
    assert_contract_compatible("live-caption-v1", "live-caption-v1")


def test_a_media_artifact_cannot_be_used_on_the_live_route():
    with pytest.raises(ContractViolation, match="입력 계약 불일치"):
        assert_contract_compatible("media-pipeline-v1", "live-caption-v1")


def test_a_live_artifact_cannot_be_used_on_the_media_route():
    """반대 방향도 막아야 합니다.

    음향 맥락을 기대하는 경로에 그것 없이 학습한 모델을 넣으면 안 됩니다.
    """
    with pytest.raises(ContractViolation, match="입력 계약 불일치"):
        assert_contract_compatible("live-caption-v1", "media-pipeline-v1")


# ------------------------------------------------------------ 아티팩트 계약 결합


@pytest.fixture(scope="module")
def live_artifact(tmp_path_factory):
    from audire.risk.artifact import fit_live_artifact

    art = fit_live_artifact(_tiny_live_config(tmp_path_factory.mktemp("live")))
    return art


def _tiny_live_config(directory):
    import yaml as _yaml

    path = directory / "live.yaml"
    path.write_text(
        _yaml.safe_dump(
            {
                "name": "live_tiny",
                "simulation": {
                    "name": "c",
                    "seeds": [1],
                    "n_listeners": 12,
                    "n_calibration_trials": 20,
                    "n_word_trials": 20,
                },
                "arms": ["live_word_context_clinical_confusion"],
                "models": ["logistic"],
                "n_splits": 3,
                "n_bootstrap": 0,
                "contrasts": [],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return path


def test_live_artifact_declares_its_contract_and_schema(live_artifact):
    meta = live_artifact.metadata
    assert meta["input_contract"] == "live-caption-v1"
    assert meta["artifact_type"] == "live_caption"
    assert meta["feature_schema_hash"]
    assert meta["intended_use"] == "browser_dom_live_caption_engineering_demo"


def test_live_artifact_records_synthetic_provenance(live_artifact):
    """합성 학습을 사람 효능 근거로 읽지 않도록 아티팩트 자체가 못박습니다."""
    meta = live_artifact.metadata
    assert meta["training_source"] == "synthetic simulation"
    assert meta["human_efficacy_evidence"] is False
    assert meta["clinical_efficacy_claim"] is False
    assert "simulator_version" in meta


def test_live_artifact_has_no_acoustic_columns(live_artifact):
    names = live_artifact.scorer.spec_feature_names()
    assert names
    LIVE_CAPTION_V1.validate_columns(names)


def test_live_artifact_is_rejected_by_the_media_loader(live_artifact, tmp_path):
    from audire.risk.artifact import DeploymentArtifact

    path, _ = live_artifact.save(tmp_path / "live.joblib")
    DeploymentArtifact.load(path, expect_contract="live-caption-v1")
    with pytest.raises(ContractViolation, match="입력 계약 불일치"):
        DeploymentArtifact.load(path, expect_contract="media-pipeline-v1")


def test_a_schema_hash_mismatch_is_rejected(live_artifact):
    """열 이름이나 순서가 달라지면 계수를 그대로 쓸 수 없습니다."""
    import dataclasses

    from audire.risk.artifact import ModelArtifactError

    tampered = dataclasses.replace(
        live_artifact,
        metadata={**live_artifact.metadata, "feature_schema_hash": "0" * 64},
    )
    with pytest.raises(ModelArtifactError, match="feature schema mismatch"):
        tampered.validate()


def test_an_artifact_without_a_contract_is_rejected(live_artifact):
    import dataclasses

    from audire.risk.artifact import ModelArtifactError

    meta = {k: v for k, v in live_artifact.metadata.items() if k != "input_contract"}
    with pytest.raises(ModelArtifactError, match="input contract"):
        dataclasses.replace(live_artifact, metadata=meta).validate()


def test_fit_live_artifact_refuses_an_arm_with_acoustic_context(tmp_path_factory):
    from audire.risk.artifact import fit_live_artifact

    config = _tiny_live_config(tmp_path_factory.mktemp("bad"))
    with pytest.raises(ContractViolation, match="context"):
        fit_live_artifact(config, arm="clinical_plus_confusion")
