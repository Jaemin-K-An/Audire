"""Pinned Korean ASR regression: WER/CER, timestamps and runtime provenance.

This evaluates recognition quality only. It never consumes a hearing/confusion profile
and therefore cannot be confused with listener-specific mishearing risk. The default
subset contains the first utterance from each Zeroth test speaker: small enough for a
CPU regression, balanced across the ten test speakers, and selected without looking at
recognition results.
"""

from __future__ import annotations

import re
import tempfile
import time
import unicodedata
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from audire.asr.base import ASRBackend, Transcript
from audire.asr.whisper_backend import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    FasterWhisperBackend,
)
from audire.data.zeroth import ZerothUtterance, load_zeroth_utterances
from audire.experiments.registry import (
    RunRecord,
    fail_run,
    finish_run,
    new_run,
    save_artifact,
)

NORMALIZATION_VERSION = "ko-basic-v1"


class ASREvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "asr_zeroth_regression"
    description: str = ""
    source_id: Literal["zeroth_korean_test"] = "zeroth_korean_test"
    selection: Literal["first_per_speaker"] = "first_per_speaker"
    n_utterances: int = Field(default=10, ge=1, le=10)
    language: Literal["ko"] = "ko"
    model_id: str = DEFAULT_MODEL_ID
    model_revision: str = DEFAULT_MODEL_REVISION
    device: Literal["cpu"] = "cpu"
    compute_type: str = "int8"
    normalization: Literal["ko-basic-v1"] = "ko-basic-v1"

    @classmethod
    def load(cls, path: Path) -> ASREvalConfig:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@dataclass(frozen=True, slots=True)
class EditCounts:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_units: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        return self.errors / self.reference_units if self.reference_units else 0.0

    def __add__(self, other: EditCounts) -> EditCounts:
        return EditCounts(
            substitutions=self.substitutions + other.substitutions,
            deletions=self.deletions + other.deletions,
            insertions=self.insertions + other.insertions,
            reference_units=self.reference_units + other.reference_units,
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "errors": self.errors, "rate": self.rate}


def normalize_korean_text(text: str) -> str:
    """NFKC, lowercase, replace punctuation/underscores by spaces, collapse whitespace."""
    normalized = unicodedata.normalize("NFKC", text).lower().replace("_", " ")
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> EditCounts:
    """Deterministic Levenshtein alignment with substitution/deletion/insertion counts."""
    # Each cell stores (cost, S, D, I). Candidate order is the deterministic tie break.
    rows: list[list[tuple[int, int, int, int]]] = [
        [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    ]
    for i in range(1, len(reference) + 1):
        rows.append([(i, 0, i, 0)] + [(0, 0, 0, 0)] * len(hypothesis))
        for j in range(1, len(hypothesis) + 1):
            diagonal = rows[i - 1][j - 1]
            if reference[i - 1] == hypothesis[j - 1]:
                rows[i][j] = diagonal
                continue
            substitution = (diagonal[0] + 1, diagonal[1] + 1, diagonal[2], diagonal[3])
            above = rows[i - 1][j]
            deletion = (above[0] + 1, above[1], above[2] + 1, above[3])
            left = rows[i][j - 1]
            insertion = (left[0] + 1, left[1], left[2], left[3] + 1)
            rows[i][j] = min(substitution, deletion, insertion, key=lambda item: item[0])
    _, substitutions, deletions, insertions = rows[-1][-1]
    return EditCounts(substitutions, deletions, insertions, len(reference))


def first_per_speaker_indices(
    utterances: Sequence[ZerothUtterance], n_utterances: int
) -> list[int]:
    """Select before inference, using only file order and speaker id."""
    indices: list[int] = []
    speakers: set[int] = set()
    for index, utterance in enumerate(utterances):
        if utterance.speaker_id in speakers:
            continue
        speakers.add(utterance.speaker_id)
        indices.append(index)
        if len(indices) == n_utterances:
            return indices
    raise ValueError(
        f"requested {n_utterances} speakers but the test split contains only {len(indices)}"
    )


def aggregate_asr_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    word = EditCounts()
    char = EditCounts()
    for row in rows:
        word += EditCounts(**row["word_counts"])
        char += EditCounts(**row["char_counts"])
    n = len(rows)
    duration = sum(float(row["duration_s"]) for row in rows)
    runtime = sum(float(row["runtime_s"]) for row in rows)
    return {
        "n_utterances": n,
        "n_speakers": len({int(row["speaker_id"]) for row in rows}),
        "wer": word.to_dict(),
        "cer_no_spaces": char.to_dict(),
        "exact_match_rate": (
            sum(row["reference_normalized"] == row["hypothesis_normalized"] for row in rows) / n
            if n
            else 0.0
        ),
        "audio_duration_s": duration,
        "runtime_s": runtime,
        "real_time_factor": runtime / duration if duration else 0.0,
        "n_tokens": sum(int(row["n_tokens"]) for row in rows),
        "n_tokens_with_confidence": sum(int(row["n_tokens_with_confidence"]) for row in rows),
        "utterances_with_timing_problems": sum(bool(row["timing_problems"]) for row in rows),
        "n_timing_problems": sum(len(row["timing_problems"]) for row in rows),
    }


def evaluate_pair(
    utterance: ZerothUtterance, transcript: Transcript, *, runtime_s: float
) -> dict[str, Any]:
    reference = normalize_korean_text(utterance.text)
    hypothesis = normalize_korean_text(transcript.text)
    word = edit_counts(reference.split(), hypothesis.split())
    char = edit_counts(list(reference.replace(" ", "")), list(hypothesis.replace(" ", "")))
    return {
        "utterance_id": utterance.utterance_id,
        "speaker_id": utterance.speaker_id,
        "reference": utterance.text,
        "hypothesis": transcript.text,
        "reference_normalized": reference,
        "hypothesis_normalized": hypothesis,
        "word_counts": asdict(word),
        "char_counts": asdict(char),
        "duration_s": transcript.duration_s,
        "runtime_s": runtime_s,
        "n_tokens": len(transcript.tokens),
        "n_tokens_with_confidence": sum(t.confidence is not None for t in transcript.tokens),
        "timing_problems": transcript.timing_problems(),
        "transcript_provenance": transcript.provenance,
    }


def run_asr_evaluation(
    cfg: ASREvalConfig,
    *,
    backend: ASRBackend | None = None,
    record: RunRecord | None = None,
) -> dict[str, Any]:
    """Run actual ASR on the pinned subset and record aggregate and per-item artifacts."""
    metadata = load_zeroth_utterances(with_audio=False)
    indices = first_per_speaker_indices(metadata, cfg.n_utterances)
    utterances = load_zeroth_utterances(indices=indices, with_audio=True)
    active_backend = backend or FasterWhisperBackend(
        model_id=cfg.model_id,
        model_revision=cfg.model_revision,
        device=cfg.device,
        compute_type=cfg.compute_type,
    )
    rec = record or new_run(
        cfg.name,
        cfg.model_dump(mode="json") | {"selected_indices": indices},
        [],
        notes=cfg.description,
    )
    try:
        import soundfile as sf

        rows: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory(prefix="audire-asr-eval-") as temp:
            work = Path(temp)
            for utterance in utterances:
                if utterance.audio is None:
                    raise ValueError(f"audio was not decoded for {utterance.utterance_id}")
                media = work / f"{utterance.utterance_id}.wav"
                sf.write(media, utterance.audio, utterance.sample_rate)
                started = time.perf_counter()
                transcript = active_backend.transcribe(media, language=cfg.language)
                runtime_s = time.perf_counter() - started
                rows.append(evaluate_pair(utterance, transcript, runtime_s=runtime_s))

        metrics = aggregate_asr_metrics(rows)
        summary = {
            "schema": "audire.asr-eval.v1",
            "experiment": cfg.name,
            "source_id": cfg.source_id,
            "selection": cfg.selection,
            "selected_indices": indices,
            "selected_utterance_ids": [row["utterance_id"] for row in rows],
            "normalization": cfg.normalization,
            "backend": active_backend.describe(),
            "metrics": metrics,
            "is_synthetic": False,
            "claim_scope": (
                "Small pinned public-corpus regression baseline; not a clinical result and "
                "not an estimate of listener-specific mishearing risk."
            ),
        }
        save_artifact(rec, "asr_evaluation_items.json", rows)
        save_artifact(rec, "summary.json", summary)
        finish_run(rec, metrics)
        return {"run_id": rec.run_id, "summary": summary, "artifacts": list(rec.artifacts)}
    except Exception as exc:
        fail_run(rec, f"{type(exc).__name__}: {exc}")
        raise
