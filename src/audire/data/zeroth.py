"""Access to the locally fetched Zeroth-Korean test split (CC BY 4.0).

Only the transcripts and a small pinned audio subset are exposed. Audio is read from the
local parquet shard and never copied into the repository.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from audire.data.fetch import local_path_for
from audire.data.manifest import require_verified
from audire.data.sources import registry

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class ZerothUtterance:
    """One Zeroth-Korean test utterance."""

    utterance_id: str
    speaker_id: int
    text: str
    sample_rate: int
    audio: FloatArray | None = None


def _parquet_path(*, verify: bool = True) -> Path:
    """Locate the pinned shard, verifying its manifest before it is read.

    Verification happens here rather than only at fetch time so that a file edited after
    download cannot silently become model input (P0.2).
    """
    src = registry().get("zeroth_korean_test")
    if verify:
        require_verified(src.id)
    rel = str(src.expected.get("parquet_file", "data/test-00000-of-00001.parquet"))
    path = local_path_for(src) / rel
    if not path.exists():
        raise FileNotFoundError(
            f"Zeroth-Korean test shard not found at {path}. "
            f"Run `python scripts/fetch_data.py zeroth_korean_test` first."
        )
    return path


def load_zeroth_transcripts() -> list[str]:
    """Return every transcript in the pinned test split."""
    import pyarrow.parquet as pq

    table = pq.read_table(_parquet_path(), columns=["text"])
    return [str(t) for t in table.column("text").to_pylist()]


def load_zeroth_utterances(
    limit: int | None = None,
    *,
    indices: Sequence[int] | None = None,
    with_audio: bool = False,
) -> list[ZerothUtterance]:
    """Load utterances from the pinned test split in file order (deterministic).

    ``with_audio=True`` decodes the embedded audio, which is slower and much larger; use
    it only for the small pinned subsets that end-to-end tests need.
    """
    import pyarrow.parquet as pq

    if limit is not None and indices is not None:
        raise ValueError("specify either limit or indices, not both")
    columns = ["id", "speaker_id", "text"] + (["audio"] if with_audio else [])
    table = pq.read_table(_parquet_path(), columns=columns)
    if limit is not None:
        table = table.slice(0, limit)
    elif indices is not None:
        if any(index < 0 or index >= len(table) for index in indices):
            raise IndexError(f"utterance indices must be within 0..{len(table) - 1}")
        table = table.take(list(indices))

    ids = table.column("id").to_pylist()
    speakers = table.column("speaker_id").to_pylist()
    texts = table.column("text").to_pylist()
    audios: list[Any] = table.column("audio").to_pylist() if with_audio else [None] * len(ids)

    out: list[ZerothUtterance] = []
    for uid, spk, txt, aud in zip(ids, speakers, texts, audios, strict=True):
        samples: FloatArray | None = None
        rate = 16000
        if aud is not None:
            samples, rate = _decode_audio(aud)
        out.append(
            ZerothUtterance(
                utterance_id=str(uid),
                speaker_id=int(spk),
                text=str(txt),
                sample_rate=rate,
                audio=samples,
            )
        )
    return out


def _decode_audio(payload: dict[str, Any]) -> tuple[FloatArray, int]:
    """Decode a Hugging Face ``Audio`` struct into float32 samples."""
    import io

    import soundfile as sf

    raw = payload.get("bytes")
    if raw is None:
        path = payload.get("path")
        if not path:
            raise ValueError("audio struct carries neither bytes nor a path")
        data, rate = sf.read(path, dtype="float32", always_2d=False)
    else:
        data, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1).astype(np.float32)
    return arr, int(rate)
