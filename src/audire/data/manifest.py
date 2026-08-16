"""Data manifests: the record that ties every reported result to specific bytes.

A manifest is written whenever data is acquired and is validated whenever data is used.
It records the source id, the pinned revision, the license, the retrieval time, and a
SHA-256 for every file, so that "which data produced this number?" always has an answer.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from audire.config.paths import manifests_dir

#: Files that are never checksummed (caches / VCS metadata inside a snapshot download).
_SKIP_DIRS = frozenset({".git", ".cache", "__pycache__", ".huggingface"})

MANIFEST_SCHEMA_VERSION = 1


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Stream a SHA-256 digest of ``path``."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and not any(part in _SKIP_DIRS for part in p.relative_to(root).parts):
            yield p


@dataclass(frozen=True, slots=True)
class FileRecord:
    """One file's identity inside a manifest."""

    path: str
    sha256: str
    bytes: int


@dataclass(slots=True)
class Manifest:
    """Provenance record for one acquired dataset."""

    source_id: str
    license: str
    homepage: str | None
    revision: str | None
    retrieved_at_utc: str
    local_path: str
    files: list[FileRecord]
    expected: dict[str, Any] = field(default_factory=dict)
    checks: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    schema_version: int = MANIFEST_SCHEMA_VERSION

    # ------------------------------------------------------------------ derived

    @property
    def n_files(self) -> int:
        return len(self.files)

    @property
    def total_bytes(self) -> int:
        return sum(f.bytes for f in self.files)

    @property
    def content_digest(self) -> str:
        """A single digest over the whole file set.

        Stable across machines: derived from sorted relative paths and their digests, so
        it identifies *content*, not download order or absolute location.
        """
        h = hashlib.sha256()
        for rec in sorted(self.files, key=lambda r: r.path):
            h.update(rec.path.encode("utf-8"))
            h.update(b"\0")
            h.update(rec.sha256.encode("ascii"))
            h.update(b"\n")
        return h.hexdigest()

    # ------------------------------------------------------------------ build

    @classmethod
    def build(
        cls,
        *,
        source_id: str,
        license: str,  # noqa: A002 - the field is genuinely called "license"
        local_path: Path,
        homepage: str | None = None,
        revision: str | None = None,
        expected: dict[str, Any] | None = None,
        checks: dict[str, Any] | None = None,
        notes: str = "",
    ) -> Self:
        """Checksum everything under ``local_path`` and assemble a manifest."""
        root = local_path.resolve()
        files = [
            FileRecord(
                path=str(p.relative_to(root)),
                sha256=sha256_file(p),
                bytes=p.stat().st_size,
            )
            for p in _iter_files(root)
        ]
        return cls(
            source_id=source_id,
            license=license,
            homepage=homepage,
            revision=revision,
            retrieved_at_utc=datetime.now(UTC).isoformat(),
            local_path=str(root),
            files=files,
            expected=dict(expected or {}),
            checks=dict(checks or {}),
            notes=notes,
        )

    # ------------------------------------------------------------------ io

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["n_files"] = self.n_files
        payload["total_bytes"] = self.total_bytes
        payload["content_digest"] = self.content_digest
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Self:
        data = {k: v for k, v in payload.items() if k in cls.__dataclass_fields__}
        data["files"] = [FileRecord(**f) for f in payload.get("files", [])]
        return cls(**data)

    def path(self) -> Path:
        return manifests_dir() / f"{self.source_id}.json"

    def save(self, path: Path | None = None) -> Path:
        target = path or self.path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=False),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, source_id: str, path: Path | None = None) -> Self:
        target = path or (manifests_dir() / f"{source_id}.json")
        if not target.exists():
            raise FileNotFoundError(
                f"no manifest for source {source_id!r} at {target}. "
                f"Run `make data` (or `python scripts/fetch_data.py {source_id}`) first."
            )
        return cls.from_dict(json.loads(target.read_text(encoding="utf-8")))

    # ------------------------------------------------------------------ validation

    def verify(self, *, deep: bool = True) -> list[str]:
        """Re-check the manifest against the filesystem.

        Returns a list of human-readable problems; an empty list means the local data
        still matches the manifest exactly.

        ``deep=False`` checks presence and size only, which is enough for a fast CI gate
        on large corpora.
        """
        problems: list[str] = []
        root = Path(self.local_path)
        if not root.exists():
            return [f"local path is missing: {root}"]

        recorded = {rec.path: rec for rec in self.files}
        present = {str(p.relative_to(root)) for p in _iter_files(root)}

        for missing in sorted(set(recorded) - present):
            problems.append(f"missing file: {missing}")
        for extra in sorted(present - set(recorded)):
            problems.append(f"unrecorded file present: {extra}")

        for rel in sorted(set(recorded) & present):
            rec = recorded[rel]
            fp = root / rel
            size = fp.stat().st_size
            if size != rec.bytes:
                problems.append(f"size mismatch: {rel} ({size} != {rec.bytes})")
                continue
            if deep and sha256_file(fp) != rec.sha256:
                problems.append(f"checksum mismatch: {rel}")
        return problems
