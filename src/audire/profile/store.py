"""Local, git-ignored persistence for hearing and confusion profiles.

Real listener data **never** enters the repository. This store writes to
``private/profiles/`` (see :func:`audire.config.paths.private_dir`), which ``.gitignore``
blocks and ``scripts/check_repo_hygiene.py`` enforces in CI.

Synthetic profiles may be written anywhere, but are still tagged so that a synthetic
profile can never be mistaken for an observed one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audire.config.paths import private_dir
from audire.confusion.profile import ConfusionProfile
from audire.profile.schema import HearingProfile

#: Listener ids are used as filenames, so they are restricted to a safe alphabet. This
#: also discourages putting a person's name in the id.
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class ProfileStoreError(RuntimeError):
    """Raised on invalid ids, missing profiles or refused writes."""


def validate_listener_id(listener_id: str) -> str:
    if not _SAFE_ID.match(listener_id):
        raise ProfileStoreError(
            f"invalid listener id {listener_id!r}: use 1-64 characters from "
            f"[A-Za-z0-9._-] starting alphanumeric. Do not use names or other direct "
            f"identifiers."
        )
    return listener_id


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
        path = self.hearing_path(profile.listener_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
        return path

    def save_confusion(self, profile: ConfusionProfile) -> Path:
        path = self.confusion_path(profile.listener_id)
        profile.save_json(path)
        return path

    def append_responses(self, listener_id: str, rows: list[dict[str, Any]]) -> Path:
        """Append raw calibration responses as JSON lines (append-only audit trail)."""
        path = self.responses_path(listener_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
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
