"""Replaceable Korean ASR adapter with word timestamps.

ASR confidence and listener mishearing risk are separate quantities throughout
(ADR-0010): a word the recogniser was unsure about is not a word the listener would
mishear, and the risk model never sees the recogniser's confidence.
"""

from audire.asr.base import ASRBackend, ASRUnavailable, Token, Transcript
from audire.asr.pipeline import (
    CaptionResult,
    IncompleteProfile,
    caption_media,
    check_ready,
    score_transcript,
)
from audire.asr.replay import ReplayBackend, save_transcript, transcript_from_dict
from audire.asr.whisper_backend import (
    DEFAULT_DECODE_OPTIONS,
    DEFAULT_MODEL_ID,
    FasterWhisperBackend,
)

#: Registry of selectable backends. `replay` is constructed with a transcript path.
BACKENDS = {
    "faster-whisper": FasterWhisperBackend,
    "replay": ReplayBackend,
}

__all__ = [
    "BACKENDS",
    "DEFAULT_DECODE_OPTIONS",
    "DEFAULT_MODEL_ID",
    "ASRBackend",
    "ASRUnavailable",
    "CaptionResult",
    "FasterWhisperBackend",
    "IncompleteProfile",
    "ReplayBackend",
    "Token",
    "Transcript",
    "caption_media",
    "check_ready",
    "save_transcript",
    "score_transcript",
    "transcript_from_dict",
]
