"""Deployment model artifact provenance, compatibility and corruption guards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import pytest
import yaml

from audire.risk import FeatureSpec, LogisticRiskModel, WordScorer
from audire.risk.artifact import (
    DeploymentArtifact,
    ModelArtifactError,
    fit_deployment_artifact,
)


def _config(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "name": "deployment-artifact-test",
                "simulation": {
                    "name": "deployment-artifact-test-cohort",
                    "seeds": [7, 8],
                    "n_listeners": 12,
                    "n_calibration_trials": 20,
                    "n_word_trials": 30,
                },
                "arms": ["clinical_plus_confusion"],
                "models": ["logistic"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_fit_save_and_load_preserve_synthetic_training_provenance(tmp_path: Path) -> None:
    artifact = fit_deployment_artifact(_config(tmp_path / "deployment.yaml"))

    assert artifact.scorer.model.is_fitted
    assert artifact.metadata["training_source"] == "synthetic simulation"
    assert artifact.metadata["is_synthetic_training"] is True
    assert artifact.metadata["seeds"] == [7, 8]
    assert artifact.metadata["n_seed_cohorts"] == 2
    assert artifact.metadata["n_listener_realizations"] == 24
    assert artifact.metadata["n_listener_ids_per_seed"] == [12, 12]
    assert artifact.metadata["n_word_trials"] == 12 * 30 * 2
    assert artifact.scorer.describe()["provenance"]["clinical_efficacy_claim"] is False

    model, sidecar = artifact.save(tmp_path / "audire.joblib")
    restored = DeploymentArtifact.load(model)
    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))

    assert restored.scorer.model.is_fitted
    assert restored.metadata == artifact.metadata
    assert sidecar_payload["scorer"]["arm"] == "clinical_plus_confusion"
    assert len(sidecar_payload["artifact_sha256"]) == 64


def test_checksum_mismatch_is_rejected_before_deserialization(tmp_path: Path) -> None:
    artifact = fit_deployment_artifact(_config(tmp_path / "deployment.yaml"))
    model, _ = artifact.save(tmp_path / "audire.joblib")
    model.write_bytes(model.read_bytes() + b"tampered")

    with pytest.raises(ModelArtifactError, match="checksum mismatch"):
        DeploymentArtifact.load(model)


def test_unfitted_or_source_ambiguous_artifact_is_rejected() -> None:
    scorer = WordScorer(model=LogisticRiskModel(), spec=FeatureSpec.arm("clinical_plus_confusion"))
    artifact = DeploymentArtifact(schema_version=1, scorer=scorer, metadata={})

    with pytest.raises(ModelArtifactError, match="unfitted"):
        artifact.validate()


def test_artifact_validation_rejects_schema_version_and_missing_source(tmp_path: Path) -> None:
    fitted = fit_deployment_artifact(_config(tmp_path / "deployment.yaml"))
    wrong_schema = DeploymentArtifact(
        schema_version=999, scorer=fitted.scorer, metadata=fitted.metadata
    )
    with pytest.raises(ModelArtifactError, match="unsupported model artifact schema"):
        wrong_schema.validate()

    missing_source = DeploymentArtifact(schema_version=1, scorer=fitted.scorer, metadata={})
    with pytest.raises(ModelArtifactError, match="does not declare its training source"):
        missing_source.validate()


def test_artifact_load_rejects_missing_invalid_and_wrong_object(tmp_path: Path) -> None:
    missing = tmp_path / "missing.joblib"
    with pytest.raises(ModelArtifactError, match="is missing"):
        DeploymentArtifact.load(missing)

    model = tmp_path / "wrong.joblib"
    joblib.dump({"not": "an artifact"}, model)
    sidecar = model.with_suffix(".joblib.json")
    sidecar.write_text("not json", encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="invalid model artifact sidecar"):
        DeploymentArtifact.load(model)

    sidecar.write_text(
        json.dumps({"artifact_sha256": hashlib.sha256(model.read_bytes()).hexdigest()}),
        encoding="utf-8",
    )
    with pytest.raises(ModelArtifactError, match="not DeploymentArtifact"):
        DeploymentArtifact.load(model)


def test_deployment_build_refuses_undeclared_or_nonlinear_choices(tmp_path: Path) -> None:
    config = _config(tmp_path / "deployment.yaml")
    with pytest.raises(ValueError, match=r"arm .* is not declared"):
        fit_deployment_artifact(config, arm="clinical")
    with pytest.raises(ValueError, match=r"model .* is not declared"):
        fit_deployment_artifact(config, model_name="gradient_boosting")

    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload["models"].append("gradient_boosting")
    config.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"only .* logistic"):
        fit_deployment_artifact(config, model_name="gradient_boosting")
