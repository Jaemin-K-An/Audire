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


class SourceRegistryError(RuntimeError):
    """Raised when ``data/sources.yaml`` is malformed.

    Registry problems are fatal rather than tolerated. A silently dropped
    ``prohibited_uses`` key removes a compliance tripwire without anyone noticing.
    """


#: Licence tokens that forbid derivative works. Compared against the normalised token
#: set, never against a raw substring: "ODbL" contains the letters "nd" and must not be
#: mistaken for a No-Derivatives clause.
_NO_DERIVATIVES_TOKENS: frozenset[str] = frozenset({"ND"})


def normalise_license(license_text: str) -> str:
    """Canonicalise a licence string to hyphen-separated upper case.

    ``CC BY-NC-ND 4.0``, ``cc_by_nc_nd_4.0`` and ``CC-BY-NC-ND-4.0`` all denote the same
    licence. The previous check split on hyphens only, so the space- and underscore-
    separated spellings reported ``redistribution_allowed=True`` for an ND licence —
    the exact opposite of the truth.
    """
    collapsed = license_text.strip().replace("_", "-").replace(" ", "-").upper()
    while "--" in collapsed:
        collapsed = collapsed.replace("--", "-")
    return collapsed.strip("-")


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
    #: Korean-language title where the source has one. Present in the shipped registry;
    #: strict loading exposed that it was being silently discarded before.
    korean_title: str | None = None
    repo_id: str | None = None
    repo_type: str | None = None
    revision: str | None = None
    record_id: str | None = None
    doi: str | None = None
    #: Upstream publication date where the source records one (Zenodo). Also silently
    #: discarded before strict loading.
    publication_date: str | None = None
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
    def license_tokens(self) -> tuple[str, ...]:
        """Normalised licence tokens, e.g. ``("CC", "BY", "NC", "ND", "4.0")``."""
        return tuple(normalise_license(self.license).split("-"))

    @property
    def redistribution_allowed(self) -> bool:
        """Whether derivative works may be produced and shared (i.e. no ND clause).

        A compliance tripwire, not legal advice. Judged on the normalised token set so
        that spelling variants cannot flip the answer.
        """
        return not (_NO_DERIVATIVES_TOKENS & set(self.license_tokens))


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


#: Fields every registered source must declare.
_REQUIRED_SOURCE_FIELDS: frozenset[str] = frozenset(
    {"id", "role", "title", "kind", "license", "verified_at"}
)


def load_registry(path: Path | None = None) -> SourceRegistry:
    """Parse the source registry strictly.

    Unknown keys are errors. The previous loader filtered the payload down to known
    dataclass fields, so a typo such as ``prohibited_use`` for ``prohibited_uses``
    silently removed the entire prohibited-use list and disarmed the tripwire that
    :meth:`Source.assert_permits` depends on.
    """
    p = path or sources_file()
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    known = set(Source.__dataclass_fields__)
    sources = {}
    for index, entry in enumerate(raw.get("sources", [])):
        payload = dict(entry)
        where = f"sources[{index}]" + (f" (id={payload['id']!r})" if "id" in payload else "")

        unknown = sorted(set(payload) - known)
        if unknown:
            raise SourceRegistryError(
                f"{where}: 알 수 없는 키 {unknown}. 오타라면 금지 목록 같은 정책이 "
                f"조용히 사라집니다. 알려진 키: {sorted(known)}"
            )
        missing = sorted(_REQUIRED_SOURCE_FIELDS - set(payload))
        if missing:
            raise SourceRegistryError(f"{where}: 필수 필드 누락 {missing}")
        if payload["id"] in sources:
            raise SourceRegistryError(f"{where}: 중복(duplicate) source id {payload['id']!r}")
        if payload.get("requires_human_acknowledgement") and not payload.get("acknowledgement_env"):
            raise SourceRegistryError(
                f"{where}: requires_human_acknowledgement 인데 acknowledgement_env 가 "
                f"없습니다. 승인 방법이 없으면 게이트가 무의미합니다."
            )

        payload["permitted_uses"] = tuple(payload.get("permitted_uses") or ())
        payload["prohibited_uses"] = tuple(payload.get("prohibited_uses") or ())
        payload["verified_at"] = str(payload["verified_at"])
        sources[payload["id"]] = Source(**payload)
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
