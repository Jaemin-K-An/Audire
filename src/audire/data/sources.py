"""Typed access to ``data/sources.yaml``.

Nothing may be downloaded that is not registered here, and every registered source
carries the permitted/prohibited-use statements that the rest of the system enforces.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from audire.config.paths import sources_file


class SourceUseViolation(RuntimeError):
    """Raised when code attempts a use a source's registry entry forbids."""


class AcknowledgementRequired(RuntimeError):
    """Raised when a source requires a human step that has not been acknowledged."""


@dataclass(frozen=True, slots=True)
class Source:
    """One registered external data source."""

    id: str
    role: str
    title: str
    kind: str
    license: str
    verified_at: str
    homepage: str | None = None
    repo_id: str | None = None
    repo_type: str | None = None
    revision: str | None = None
    record_id: str | None = None
    doi: str | None = None
    mirror: str | None = None
    license_source: str | None = None
    creator_contact: str | None = None
    requires_human_acknowledgement: bool = False
    acknowledgement_env: str | None = None
    expected: dict[str, Any] = field(default_factory=dict)
    permitted_uses: tuple[str, ...] = ()
    prohibited_uses: tuple[str, ...] = ()

    # ------------------------------------------------------------------ policy

    def acknowledgement_satisfied(self) -> bool:
        """Whether the required human step has been acknowledged in this environment."""
        if not self.requires_human_acknowledgement:
            return True
        if not self.acknowledgement_env:
            return False
        return os.environ.get(self.acknowledgement_env) == "1"

    def require_acknowledgement(self) -> None:
        """Raise unless the human acknowledgement step has been performed.

        AUDIRE never performs the human step itself -- it does not send email and does
        not accept terms on anyone's behalf.
        """
        if self.acknowledgement_satisfied():
            return
        raise AcknowledgementRequired(
            f"Source '{self.id}' ({self.license}) requires a human step before download.\n"
            f"  Dataset card requirement: inform the creator of intended use and scope.\n"
            f"  Creator contact: {self.creator_contact or 'see the dataset card'}\n"
            f"  AUDIRE will not send that message for you.\n"
            f"  After you have handled it, set {self.acknowledgement_env}=1 and rerun."
        )

    def assert_permits(self, intended_use: str) -> None:
        """Raise if ``intended_use`` matches a prohibited-use statement.

        Matching is substring-based and deliberately conservative: this is a tripwire
        that keeps a prohibited use from being added silently, not a legal analysis.
        """
        low = intended_use.lower()
        for prohibited in self.prohibited_uses:
            key = prohibited.lower()
            if key in low or any(tok in low for tok in _significant_tokens(key)):
                raise SourceUseViolation(
                    f"Source '{self.id}' ({self.license}) prohibits: {prohibited!r}. "
                    f"Refusing intended use {intended_use!r}."
                )

    @property
    def redistribution_allowed(self) -> bool:
        """Whether derivative works may be produced and shared (i.e. no ND clause)."""
        return "ND" not in self.license.upper().split("-")


_STOPWORDS = frozenset(
    {"the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "this", "that", "its", "as"}
)


def _significant_tokens(text: str) -> list[str]:
    return [t for t in text.replace("/", " ").split() if len(t) > 4 and t not in _STOPWORDS]


@dataclass(frozen=True, slots=True)
class LiteratureRef:
    """A cited publication. Never downloaded; used for provenance of transcribed values."""

    id: str
    doi: str
    title: str
    authors: str
    journal: str
    year: int
    volume_issue_pages: str
    url: str
    accessed: str
    use: str
    do_not: str


@dataclass(frozen=True, slots=True)
class SourceRegistry:
    """The parsed contents of ``data/sources.yaml``."""

    schema_version: int
    sources: dict[str, Source]
    literature: dict[str, LiteratureRef]

    def get(self, source_id: str) -> Source:
        try:
            return self.sources[source_id]
        except KeyError as exc:
            raise KeyError(
                f"unregistered source {source_id!r}; known: {sorted(self.sources)}"
            ) from exc

    def cite(self, literature_id: str) -> LiteratureRef:
        try:
            return self.literature[literature_id]
        except KeyError as exc:
            raise KeyError(
                f"unregistered literature {literature_id!r}; known: {sorted(self.literature)}"
            ) from exc


def load_registry(path: Path | None = None) -> SourceRegistry:
    """Parse the source registry from ``path`` (defaults to ``data/sources.yaml``)."""
    p = path or sources_file()
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    sources = {}
    for entry in raw.get("sources", []):
        payload = dict(entry)
        payload["permitted_uses"] = tuple(payload.get("permitted_uses") or ())
        payload["prohibited_uses"] = tuple(payload.get("prohibited_uses") or ())
        payload["verified_at"] = str(payload["verified_at"])
        known = set(Source.__dataclass_fields__)
        sources[payload["id"]] = Source(**{k: v for k, v in payload.items() if k in known})
    literature = {}
    for entry in raw.get("literature", []):
        payload = dict(entry)
        payload["doi"] = str(payload["doi"])
        payload["accessed"] = str(payload["accessed"])
        known_lit = set(LiteratureRef.__dataclass_fields__)
        literature[payload["id"]] = LiteratureRef(
            **{k: v for k, v in payload.items() if k in known_lit}
        )
    return SourceRegistry(
        schema_version=int(raw["schema_version"]), sources=sources, literature=literature
    )


@lru_cache(maxsize=1)
def registry() -> SourceRegistry:
    """Cached default registry."""
    return load_registry()
