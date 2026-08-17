"""Versioned deployment artifacts for the production word-risk scorer.

The evaluated model family is selected by listener-level cross-validation, then fitted on
the complete configured *synthetic* cohort for local deployment. That last fit has no
held-out performance claim of its own: its provenance points back to the separately
recorded evaluation and labels the training source as synthetic.

Artifacts use joblib because the fitted scikit-learn pipeline cannot be represented by a
plain data file. Loading pickle-compatible formats can execute code, so only locally
created artifacts whose sidecar checksum matches are accepted; never load an artifact
received from an untrusted party.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import joblib
import numpy as np

from audire.eval.ablation import cohort_matrix
from audire.experiments.registry import (
    data_manifest_ids,
    git_is_dirty,
    git_sha,
    lock_hash,
)
from audire.experiments.runner import ExperimentConfig
from audire.live.contract import (
    LIVE_CAPTION_V1,
    assert_contract_compatible,
    feature_schema_hash,
    get_contract,
)
from audire.risk.features import ABLATION_ARMS, FeatureMatrix, FeatureSpec
from audire.risk.models import MODEL_VERSION, LogisticRiskModel, WordScorer
from audire.sim.cohort import build_cohort

ARTIFACT_SCHEMA_VERSION = 1
DEFAULT_DEPLOYMENT_ARM = "clinical_plus_confusion"
DEFAULT_DEPLOYMENT_MODEL = "logistic"


class ModelArtifactError(RuntimeError):
    """The artifact is missing, corrupt, incompatible or scientifically ambiguous."""


@dataclass(frozen=True, slots=True)
class DeploymentArtifact:
    schema_version: int
    scorer: WordScorer
    metadata: dict[str, Any]

    def validate(self) -> None:
        if self.schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ModelArtifactError(
                f"unsupported model artifact schema {self.schema_version}; "
                f"expected {ARTIFACT_SCHEMA_VERSION}"
            )
        if not self.scorer.model.is_fitted:
            raise ModelArtifactError("deployment artifact contains an unfitted risk model")
        version = self.scorer.model.describe().get("model_version")
        if version != MODEL_VERSION:
            raise ModelArtifactError(
                f"model version mismatch: artifact={version!r}, runtime={MODEL_VERSION!r}"
            )
        if "training_source" not in self.metadata:
            raise ModelArtifactError("artifact metadata does not declare its training source")

        # 입력 계약과 스키마 결합. 파일이 존재한다는 이유만으로 모델을 고르면, 음향 맥락을
        # 기대하는 모델이 그것이 없는 경로에서 조용히 돌 수 있습니다. 출력은 그럴듯하지만
        # 학습 분포 밖입니다(ADR-0021).
        contract_version = self.metadata.get("input_contract")
        if contract_version is None:
            raise ModelArtifactError(
                "artifact metadata does not declare its input contract; "
                "cannot decide which pipeline may use it"
            )
        contract = get_contract(str(contract_version))
        names = self.scorer.spec_feature_names()
        contract.validate_columns(names)

        declared = self.metadata.get("feature_schema_hash")
        actual = feature_schema_hash(str(contract_version), names)
        if declared != actual:
            raise ModelArtifactError(
                f"feature schema mismatch: artifact={declared!r}, runtime={actual!r}. "
                f"열 이름이나 순서가 달라졌으므로 계수를 그대로 쓸 수 없습니다."
            )

    def save(self, path: Path) -> tuple[Path, Path]:
        """Write the model and a human-readable, checksummed provenance sidecar."""
        self.validate()
        target = path.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        sidecar = target.with_suffix(target.suffix + ".json")
        sidecar.write_text(
            json.dumps(
                {
                    "schema_version": self.schema_version,
                    "artifact_sha256": digest,
                    "metadata": self.metadata,
                    "scorer": self.scorer.describe(),
                    "warning": "Load only this locally generated artifact; joblib is executable.",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return target, sidecar

    @classmethod
    def load(cls, path: Path, *, expect_contract: str | None = None) -> Self:
        """아티팩트를 읽습니다.

        ``expect_contract`` 를 주면 아티팩트가 선언한 계약과 대조합니다. 미디어 경로가
        라이브 아티팩트를, 라이브 경로가 미디어 아티팩트를 집어 드는 것을 막습니다 —
        두 경로는 볼 수 있는 정보가 다르므로 교차 사용할 수 없습니다.
        """
        target = path.resolve()
        sidecar = target.with_suffix(target.suffix + ".json")
        if not target.exists() or not sidecar.exists():
            raise ModelArtifactError(
                f"model artifact or checksum sidecar is missing: {target}, {sidecar}"
            )
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelArtifactError(f"invalid model artifact sidecar: {exc}") from exc
        expected = metadata.get("artifact_sha256")
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if expected != actual:
            raise ModelArtifactError(
                f"model artifact checksum mismatch: expected {expected!r}, got {actual!r}"
            )
        loaded = joblib.load(target)
        if not isinstance(loaded, cls):
            raise ModelArtifactError(
                f"artifact contains {type(loaded).__name__}, not DeploymentArtifact"
            )
        loaded.validate()
        if expect_contract is not None:
            assert_contract_compatible(str(loaded.metadata.get("input_contract")), expect_contract)
        return loaded


def _stack(matrices: list[FeatureMatrix]) -> FeatureMatrix:
    if not matrices:
        raise ValueError("cannot train a deployment model from zero cohorts")
    names = matrices[0].feature_names
    if any(matrix.feature_names != names for matrix in matrices[1:]):
        raise ValueError("feature columns differ across deployment training cohorts")
    if any(matrix.y is None for matrix in matrices):
        raise ValueError("deployment training matrices must carry labels")
    return FeatureMatrix(
        X=np.concatenate([matrix.X for matrix in matrices]),
        feature_names=names,
        groups=np.concatenate([matrix.groups for matrix in matrices]),
        y=np.concatenate([matrix.y for matrix in matrices if matrix.y is not None]),
        meta={"stacked_cohorts": len(matrices)},
    )


def fit_deployment_artifact(
    config_path: Path,
    *,
    arm: str = DEFAULT_DEPLOYMENT_ARM,
    model_name: str = DEFAULT_DEPLOYMENT_MODEL,
    contract_version: str = "media-pipeline-v1",
    artifact_type: str = "media_pipeline",
    intended_use: str = "asr_media_selective_caption",
) -> DeploymentArtifact:
    """Fit the preselected deployment family on every seed declared by ``config_path``.

    ``contract_version`` 이 이 아티팩트가 어느 경로에서 쓰일 수 있는지를 정합니다. 기본값은
    음향 맥락을 보는 미디어 경로이고, 라이브 자막 아티팩트는
    :func:`fit_live_artifact` 가 만듭니다 — 두 경로는 볼 수 있는 정보가 다르므로 아티팩트를
    교차 사용할 수 없습니다(ADR-0021).
    """
    cfg = ExperimentConfig.load(config_path)
    if arm not in cfg.arms:
        raise ValueError(f"deployment arm {arm!r} is not declared in {config_path}")
    if model_name not in cfg.models:
        raise ValueError(f"deployment model {model_name!r} is not declared in {config_path}")
    if model_name != "logistic":
        raise ValueError("only the preregistered interpretable logistic family is deployable")

    speakers = tuple(sorted({*cfg.simulation.speakers, "unknown"}))
    spec = FeatureSpec.arm(arm, speakers=speakers)
    matrices = [
        cohort_matrix(build_cohort(cfg.simulation, seed), spec) for seed in cfg.simulation.seeds
    ]
    training = _stack(matrices)
    model = LogisticRiskModel(random_state=0).fit(training)
    config_bytes = config_path.read_bytes()
    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "training_source": "synthetic simulation",
        "is_synthetic_training": True,
        "clinical_efficacy_claim": False,
        "caveat": (
            "Synthetic training artifact for engineering/accessibility validation; "
            "not evidence of clinical efficacy on human listeners."
        ),
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "experiment": cfg.name,
        "seeds": list(cfg.simulation.seeds),
        "n_seed_cohorts": len(matrices),
        # Synthetic listener ids intentionally repeat between independent seed cohorts;
        # count realizations per seed rather than pretending the repeated labels identify
        # the same person across simulations.
        "n_listener_realizations": sum(int(np.unique(m.groups).size) for m in matrices),
        "n_listener_ids_per_seed": [int(np.unique(m.groups).size) for m in matrices],
        "n_word_trials": len(training),
        "arm": arm,
        "model": model_name,
        "model_version": MODEL_VERSION,
        "input_contract": contract_version,
        "feature_schema_hash": feature_schema_hash(contract_version, model.feature_names),
        "artifact_type": artifact_type,
        "intended_use": intended_use,
        "simulator_version": cfg.simulation.simulator_version,
        # 사람 청취 이득의 근거가 아님을 아티팩트 자체에 못박습니다.
        "human_efficacy_evidence": False,
        "git_sha": git_sha(),
        "git_dirty": git_is_dirty(),
        "lock_hash": lock_hash(),
        "data_manifests": data_manifest_ids(),
    }
    scorer = WordScorer(model=model, spec=spec, provenance=metadata)
    return DeploymentArtifact(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        scorer=scorer,
        metadata=metadata,
    )


#: Phase 4(E30)에서 선택된 라이브 arm. 근거는 docs/RESULTS.md §19 와 ADR-0021.
DEFAULT_LIVE_ARM = "live_word_context_clinical_confusion"


def fit_live_artifact(
    config_path: Path,
    *,
    arm: str = DEFAULT_LIVE_ARM,
) -> DeploymentArtifact:
    """브라우저 DOM 자막 경로용 아티팩트를 적합합니다.

    미디어 아티팩트를 덮어쓰지 않는 **별도 계열**입니다. 학습과 추론이 같은 정보 제약
    아래에 있어야 하므로, 음향 맥락을 포함하지 않는 arm 으로만 적합합니다.

    이 아티팩트가 유효한 정책
    -------------------------
    E30 은 **전역 임계값**에서 이 arm 이 normal/mild 청취자에게 사실상 자막을 주지 않는다는
    것을 보였습니다(자막률 0.0004 / 0.0204). 따라서 소비 측은 **청취자별 임계값**을 써야
    하며, 그 조건에서는 모든 중증도 계층이 동일한 자막률(~0.20)과 재현율을 받습니다.
    이 제약은 ADR-0021 에 기록되어 있고 Phase 5 의 전제 조건입니다.
    """
    LIVE_CAPTION_V1.validate_blocks(ABLATION_ARMS[arm])
    return fit_deployment_artifact(
        config_path,
        arm=arm,
        model_name="logistic",
        contract_version=LIVE_CAPTION_V1.version,
        artifact_type="live_caption",
        intended_use="browser_dom_live_caption_engineering_demo",
    )
