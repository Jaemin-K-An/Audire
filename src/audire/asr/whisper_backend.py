"""``faster-whisper`` backend (default), verified against the library's current API.

Confidence handling
-------------------
``faster-whisper`` exposes a per-word ``probability`` when ``word_timestamps=True``. It is
passed through unchanged as :attr:`~audire.asr.base.Token.confidence`. When a build does
not provide it, the field is ``None`` — never a fabricated default, because a fabricated
confidence would make the ASR-versus-listener-risk separation meaningless.

Determinism
-----------
``beam_size`` and ``temperature`` are pinned and recorded in provenance. Greedy decoding
with a fixed beam is deterministic on a given build and device; the device and compute
type are recorded because they can change numerics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from audire.asr.base import ASRBackend, ASRUnavailable, Token, Transcript
from audire.config.logging import get_logger
from audire.config.paths import models_dir

log = get_logger(__name__)

#: Default model. ``large-v3`` is the most accurate multilingual Whisper checkpoint, but it
#: is a multi-GB download; ``small`` is the CPU-friendly default so that a fresh evaluator
#: can run the end-to-end path without a long download. Override per deployment.
DEFAULT_MODEL_ID = "small"

#: Pinned decode options, recorded in provenance.
DEFAULT_DECODE_OPTIONS: dict[str, Any] = {
    "beam_size": 5,
    "temperature": 0.0,
    "condition_on_previous_text": False,
    "vad_filter": True,
}


class FasterWhisperBackend(ASRBackend):
    """Korean ASR with word timestamps via ``faster-whisper`` / CTranslate2."""

    name = "faster-whisper"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        download_root: Path | None = None,
        decode_options: dict[str, Any] | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root or models_dir()
        self.decode_options = {**DEFAULT_DECODE_OPTIONS, **(decode_options or {})}
        self._model: Any = None

    # ------------------------------------------------------------------ availability

    def is_available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False
        return True

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ASRUnavailable(
                "faster-whisper is not installed. Install the ASR extra with:\n"
                "    make bootstrap-asr\n"
                "or run AUDIRE with a different backend."
            ) from exc

        self.download_root.mkdir(parents=True, exist_ok=True)
        log.info(
            "asr.load",
            backend=self.name,
            model_id=self.model_id,
            device=self.device,
            compute_type=self.compute_type,
        )
        try:
            self._model = WhisperModel(
                self.model_id,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(self.download_root),
            )
        except Exception as exc:
            raise ASRUnavailable(
                f"could not load faster-whisper model {self.model_id!r} on device "
                f"{self.device!r} with compute type {self.compute_type!r}: {exc}"
            ) from exc
        return self._model

    # ------------------------------------------------------------------ transcription

    def transcribe(self, media: Path, *, language: str = "ko") -> Transcript:
        if not media.exists():
            raise FileNotFoundError(f"media file not found: {media}")

        model = self._load()
        segments, info = model.transcribe(
            str(media),
            language=language,
            word_timestamps=True,
            **self.decode_options,
        )

        tokens: list[Token] = []
        # `segments` is a generator: transcription only runs as it is consumed.
        for segment in segments:
            words = getattr(segment, "words", None)
            if not words:
                # A segment without word timings still carries information; keep it as a
                # single token spanning the segment rather than discarding the audio.
                text = str(getattr(segment, "text", "")).strip()
                if text:
                    tokens.append(
                        Token(
                            text=text,
                            start_s=float(segment.start),
                            end_s=float(segment.end),
                            confidence=None,
                        )
                    )
                continue
            for w in words:
                text = str(w.word).strip()
                if not text:
                    continue
                prob = getattr(w, "probability", None)
                tokens.append(
                    Token(
                        text=text,
                        start_s=float(w.start),
                        end_s=max(float(w.end), float(w.start)),
                        confidence=None if prob is None else float(min(max(prob, 0.0), 1.0)),
                    )
                )

        transcript = Transcript(
            tokens=tuple(tokens),
            language=str(getattr(info, "language", language)),
            language_probability=_opt_float(getattr(info, "language_probability", None)),
            duration_s=float(getattr(info, "duration", tokens[-1].end_s if tokens else 0.0)),
            backend=self.name,
            model_id=self.model_id,
            provenance=self.describe() | {"media": media.name},
        )
        problems = transcript.timing_problems()
        if problems:
            log.warning("asr.timing_problems", n=len(problems), examples=problems[:3])
        log.info(
            "asr.done",
            backend=self.name,
            n_tokens=len(transcript),
            n_hangul=len(transcript.hangul_tokens),
            duration_s=transcript.duration_s,
        )
        return transcript

    def describe(self) -> dict[str, Any]:
        import importlib.metadata as md

        try:
            version = md.version("faster-whisper")
        except md.PackageNotFoundError:  # pragma: no cover - not installed
            version = "not-installed"
        return {
            "backend": self.name,
            "library_version": version,
            "model_id": self.model_id,
            "device": self.device,
            "compute_type": self.compute_type,
            "decode_options": dict(self.decode_options),
        }


def _opt_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None
