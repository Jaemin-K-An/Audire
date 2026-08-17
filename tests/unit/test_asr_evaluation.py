"""E7 text normalization, edit accounting, speaker selection and metric aggregation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from audire.asr import ASRBackend, Token, Transcript
from audire.asr.evaluation import (
    ASREvalConfig,
    EditCounts,
    aggregate_asr_metrics,
    edit_counts,
    evaluate_pair,
    first_per_speaker_indices,
    normalize_korean_text,
    run_asr_evaluation,
)
from audire.data.zeroth import ZerothUtterance


def _utterance(uid: str, speaker: int, text: str) -> ZerothUtterance:
    return ZerothUtterance(uid, speaker, text, 16000)


def test_korean_normalization_is_version_stable() -> None:
    assert normalize_korean_text("  안녕, WORLD!_１２３  ") == "안녕 world 123"


def test_edit_counts_separate_substitution_deletion_and_insertion() -> None:
    counts = edit_counts(["가", "나", "다"], ["가", "라", "또"])
    assert counts == EditCounts(substitutions=2, deletions=0, insertions=0, reference_units=3)
    assert counts.rate == pytest.approx(2 / 3)

    deletion = edit_counts(list("가나다"), list("가다"))
    insertion = edit_counts(list("가다"), list("가나다"))
    assert deletion.deletions == 1 and deletion.errors == 1
    assert insertion.insertions == 1 and insertion.errors == 1


def test_first_per_speaker_selection_is_preinference_and_deterministic() -> None:
    rows = [
        _utterance("a", 1, "가"),
        _utterance("b", 1, "나"),
        _utterance("c", 2, "다"),
        _utterance("d", 3, "라"),
    ]
    assert first_per_speaker_indices(rows, 3) == [0, 2, 3]
    with pytest.raises(ValueError, match="contains only 3"):
        first_per_speaker_indices(rows, 4)


def test_pair_and_aggregate_keep_wer_cer_timing_and_confidence_separate() -> None:
    utterance = _utterance("u1", 7, "가 나 다")
    transcript = Transcript(
        tokens=(Token("가", 0.0, 0.3, 0.9), Token("라", 0.2, 0.6, None)),
        language="ko",
        language_probability=0.99,
        duration_s=1.0,
        backend="test",
        model_id="test",
    )
    row = evaluate_pair(utterance, transcript, runtime_s=0.5)
    metrics = aggregate_asr_metrics([row])

    assert metrics["wer"]["rate"] == pytest.approx(2 / 3)
    assert metrics["cer_no_spaces"]["rate"] == pytest.approx(2 / 3)
    assert metrics["real_time_factor"] == 0.5
    assert metrics["n_tokens_with_confidence"] == 1
    assert metrics["utterances_with_timing_problems"] == 1
    assert metrics["n_timing_problems"] == 1


class _DeterministicBackend(ASRBackend):
    name = "unit-asr"

    def transcribe(self, media: Path, *, language: str = "ko") -> Transcript:
        del language
        return Transcript(
            tokens=(Token("가", 0.0, 0.1, 0.8),),
            language="ko",
            language_probability=1.0,
            duration_s=0.1,
            backend=self.name,
            model_id="unit",
            provenance={"media": media.name},
        )

    def describe(self) -> dict[str, str]:
        return {"backend": self.name, "model_id": "unit"}


@pytest.mark.asr
def test_asr_runner_records_success_and_failure_without_listener_risk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import audire.asr.evaluation as evaluation

    metadata = [_utterance("u1", 1, "가"), _utterance("u2", 2, "나")]

    def fake_loader(
        limit: int | None = None,
        *,
        indices: list[int] | None = None,
        with_audio: bool = False,
    ) -> list[ZerothUtterance]:
        del limit
        selected = metadata if indices is None else [metadata[index] for index in indices]
        if not with_audio:
            return selected
        return [
            ZerothUtterance(
                row.utterance_id,
                row.speaker_id,
                row.text,
                16000,
                np.zeros(1600, dtype=np.float32),
            )
            for row in selected
        ]

    monkeypatch.setattr(evaluation, "load_zeroth_utterances", fake_loader)
    monkeypatch.setenv("AUDIRE_EXPERIMENTS_DIR", str(tmp_path / "experiments"))
    monkeypatch.setenv("AUDIRE_MANIFESTS_DIR", str(tmp_path / "manifests"))
    config_path = tmp_path / "asr.yaml"
    config_path.write_text(
        yaml.safe_dump({"name": "asr-unit", "n_utterances": 2}), encoding="utf-8"
    )
    cfg = ASREvalConfig.load(config_path)

    result = run_asr_evaluation(cfg, backend=_DeterministicBackend())

    assert result["summary"]["metrics"]["n_utterances"] == 2
    assert result["summary"]["metrics"]["n_speakers"] == 2
    assert result["summary"]["is_synthetic"] is False
    assert "listener" not in result["summary"]["metrics"]
    registry = yaml.safe_load((tmp_path / "experiments" / "registry.yaml").read_text())
    assert registry["runs"][-1]["status"] == "completed"

    def missing_audio_loader(
        limit: int | None = None,
        *,
        indices: list[int] | None = None,
        with_audio: bool = False,
    ) -> list[ZerothUtterance]:
        del limit, indices, with_audio
        return metadata

    monkeypatch.setattr(evaluation, "load_zeroth_utterances", missing_audio_loader)
    failed_cfg = cfg.model_copy(update={"name": "asr-unit-failed"})
    with pytest.raises(ValueError, match="audio was not decoded"):
        run_asr_evaluation(failed_cfg, backend=_DeterministicBackend())
    registry = yaml.safe_load((tmp_path / "experiments" / "registry.yaml").read_text())
    assert registry["runs"][-1]["status"] == "failed"
