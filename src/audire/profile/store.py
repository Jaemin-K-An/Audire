"""Local, git-ignored persistence for hearing and confusion profiles.

Real listener data **never** enters the repository. This store writes to
``private/profiles/`` (see :func:`audire.config.paths.private_dir`), which ``.gitignore``
blocks and ``scripts/check_repo_hygiene.py`` enforces in CI.

Synthetic profiles may be written anywhere, but are still tagged so that a synthetic
profile can never be mistaken for an observed one.
"""

from __future__ import annotations

import json
import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audire.config.paths import private_dir
from audire.confusion.profile import ConfusionProfile
from audire.identity import validate_listener_id as _validate_listener_id
from audire.profile.schema import HearingProfile

#: Owner-only permissions for real listener data on POSIX.
_DIR_MODE = 0o700
_FILE_MODE = 0o600


def _secure_dir(path: Path) -> Path:
    """Create ``path`` owner-only, tightening an existing directory if needed."""
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        with suppress(OSError):
            path.chmod(_DIR_MODE)
    return path


def _atomic_write(path: Path, text: str) -> Path:
    """Write ``text`` to ``path`` atomically, owner-only.

    A partial overwrite of a hearing profile or a calibration response file is worse than
    a failed write: the listener's record would be silently corrupted. The payload is
    written to a temporary file in the same directory, given owner-only permissions, then
    moved into place with :func:`os.replace`, which is atomic on POSIX and Windows.
    """
    _secure_dir(path.parent)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        if os.name == "posix":
            with suppress(OSError):
                tmp.chmod(_FILE_MODE)
        # `os.replace` (not `Path.replace`) so that the atomicity failure path is
        # reachable in tests by patching one well-known symbol.
        os.replace(tmp, path)  # noqa: PTH105
    finally:
        tmp.unlink(missing_ok=True)
    return path


class ProfileStoreError(RuntimeError):
    """Raised on invalid ids, missing profiles or refused writes."""


def validate_listener_id(listener_id: str) -> str:
    """Shared identifier rule, re-raised as a store error for callers of this module.

    The rule itself lives in :mod:`audire.identity` so that the schema, the
    store, the scoring boundary and the API cannot drift apart — previously the store
    enforced a safe alphabet while the schema checked only the length, so a profile the
    schema accepted could be rejected on save.
    """
    try:
        return _validate_listener_id(listener_id)
    except ValueError as exc:
        raise ProfileStoreError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class StoredProfile:
    """A hearing profile and, when calibration has been performed, its confusion profile."""

    hearing: HearingProfile
    confusion: ConfusionProfile | None = None

    @property
    def has_calibration(self) -> bool:
        return self.confusion is not None and self.confusion.n_trials > 0


class ProfileStore:
    """Filesystem-backed profile storage.

    Parameters
    ----------
    root:
        Directory to store under. Defaults to ``private/profiles`` so that the safe
        location is the one you get by doing nothing.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or (private_dir() / "profiles")).resolve()

    # ------------------------------------------------------------------ paths

    def _dir(self, listener_id: str) -> Path:
        return self.root / validate_listener_id(listener_id)

    def hearing_path(self, listener_id: str) -> Path:
        return self._dir(listener_id) / "hearing_profile.json"

    def confusion_path(self, listener_id: str) -> Path:
        return self._dir(listener_id) / "confusion_profile.json"

    def responses_path(self, listener_id: str) -> Path:
        """Raw calibration responses. The most sensitive file in the store."""
        return self._dir(listener_id) / "calibration_responses.jsonl"

    # ------------------------------------------------------------------ write

    def save_hearing(self, profile: HearingProfile) -> Path:
        return _atomic_write(
            self.hearing_path(profile.listener_id), profile.model_dump_json(indent=2)
        )

    def save_confusion(self, profile: ConfusionProfile) -> Path:
        return _atomic_write(
            self.confusion_path(profile.listener_id),
            json.dumps(profile.to_dict(), ensure_ascii=False, indent=2),
        )

    def append_responses(self, listener_id: str, rows: list[dict[str, Any]]) -> Path:
        """Append raw calibration responses as JSON lines (append-only audit trail)."""
        path = self.responses_path(listener_id)
        _secure_dir(path.parent)
        # Append-only audit trail, so this is a genuine append rather than a replacement.
        # The mode is applied on creation and re-asserted cheaply afterwards.
        existed = path.exists()
        with path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if os.name == "posix" and not existed:
            with suppress(OSError):
                path.chmod(_FILE_MODE)
        return path

    # ------------------------------------------------------------------ read

    def load(self, listener_id: str) -> StoredProfile:
        hp = self.hearing_path(listener_id)
        if not hp.exists():
            raise ProfileStoreError(f"no hearing profile for listener {listener_id!r} at {hp}")
        hearing = HearingProfile.model_validate_json(hp.read_text(encoding="utf-8"))
        cp = self.confusion_path(listener_id)
        confusion = ConfusionProfile.load_json(cp) if cp.exists() else None
        return StoredProfile(hearing=hearing, confusion=confusion)

    def load_responses(self, listener_id: str) -> list[dict[str, Any]]:
        path = self.responses_path(listener_id)
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    def exists(self, listener_id: str) -> bool:
        return self.hearing_path(listener_id).exists()

    def list_ids(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(
            d.name
            for d in self.root.iterdir()
            if d.is_dir() and (d / "hearing_profile.json").exists()
        )

    # ------------------------------------------------------------------ export / delete

    def export(self, listener_id: str) -> dict[str, Any]:
        """Return everything stored for a listener, for a data-subject export request."""
        stored = self.load(listener_id)
        return {
            "listener_id": listener_id,
            "hearing_profile": stored.hearing.model_dump(mode="json"),
            "confusion_profile": stored.confusion.to_dict() if stored.confusion else None,
            "calibration_responses": self.load_responses(listener_id),
        }

    def delete(self, listener_id: str) -> list[str]:
        """Irreversibly delete everything stored for a listener.

        Returns the relative paths removed. Deleting a listener that does not exist is an
        error rather than a silent success, so that a failed erasure request is visible.
        """
        d = self._dir(listener_id)
        if not d.exists():
            raise ProfileStoreError(f"nothing stored for listener {listener_id!r}")
        removed: list[str] = []
        for p in sorted(d.rglob("*"), reverse=True):
            if p.is_file():
                removed.append(str(p.relative_to(self.root)))
                p.unlink()
            elif p.is_dir():
                p.rmdir()
        d.rmdir()
        return removed
