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
from audire.risk.features import FeatureMatrix, FeatureSpec
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
    def load(cls, path: Path) -> Self:
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
) -> DeploymentArtifact:
    """Fit the preselected deployment family on every seed declared by ``config_path``."""
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
