"""Replay backend: re-serves a *previously recorded* transcript.

This is **not** a mock recogniser. It performs no inference and invents nothing; it loads a
transcript that a real backend produced and that was saved to disk, exactly as a cached
ASR result would be. That makes deterministic regression tests and CI runs possible
without downloading multi-gigabyte weights.

Two guards keep it from being mistaken for real recognition:

* the transcript's ``backend`` field keeps the **original** recogniser's name, and the
  provenance records that this run replayed rather than recognised;
* :meth:`ReplayBackend.transcribe` refuses to serve a transcript recorded for a different
  media file unless the caller explicitly allows it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audire.asr.base import ASRBackend, Token, Transcript


class ReplayBackend(ASRBackend):
    """Serve a recorded transcript from disk."""

    name = "replay"

    def __init__(self, transcript_path: Path, *, allow_media_mismatch: bool = False) -> None:
        self.transcript_path = Path(transcript_path)
        self.allow_media_mismatch = allow_media_mismatch
        if not self.transcript_path.exists():
            raise FileNotFoundError(f"recorded transcript not found: {self.transcript_path}")

    def transcribe(self, media: Path, *, language: str = "ko") -> Transcript:
        # `language` is part of the backend contract but is inert here: a recorded
        # transcript already carries the language it was produced with, and silently
        # relabelling it would corrupt provenance.
        del language
        payload = json.loads(self.transcript_path.read_text(encoding="utf-8"))
        recorded_media = (payload.get("provenance") or {}).get("media")
        if recorded_media and media.name != recorded_media and not self.allow_media_mismatch:
            raise ValueError(
                f"recorded transcript was produced from {recorded_media!r} but "
                f"{media.name!r} was supplied. Pass allow_media_mismatch=True only if you "
                f"intend to reuse it."
            )
        return transcript_from_dict(payload, replayed_from=self.transcript_path)

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "transcript_path": str(self.transcript_path),
            "note": "replays a recorded transcript; performs no inference",
        }


def transcript_from_dict(
    payload: dict[str, Any], *, replayed_from: Path | None = None
) -> Transcript:
    """Rebuild a :class:`~audire.asr.base.Transcript` from its serialised form."""
    tokens = tuple(
        Token(
            text=t["text"],
            start_s=float(t["start_s"]),
            end_s=float(t["end_s"]),
            confidence=None if t.get("confidence") is None else float(t["confidence"]),
        )
        for t in payload["tokens"]
    )
    provenance = dict(payload.get("provenance") or {})
    if replayed_from is not None:
        provenance["replayed_from"] = str(replayed_from)
        provenance["replayed"] = True
    return Transcript(
        tokens=tokens,
        language=payload.get("language", "ko"),
        language_probability=payload.get("language_probability"),
        duration_s=float(payload.get("duration_s", tokens[-1].end_s if tokens else 0.0)),
        # The ORIGINAL backend is preserved: a replayed result must not claim to have been
        # produced by the replay mechanism.
        backend=payload.get("backend", "unknown"),
        model_id=payload.get("model_id", "unknown"),
        provenance=provenance,
    )


def save_transcript(transcript: Transcript, path: Path) -> Path:
    """Persist a transcript so it can be replayed deterministically later."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path
