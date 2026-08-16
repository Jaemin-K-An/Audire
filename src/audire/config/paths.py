"""Filesystem layout. Every path in AUDIRE resolves through this module.

Repository-root detection walks upward from this file looking for the marker files that
only the AUDIRE repository has. All locations can be overridden with environment
variables so that the package works when installed outside a source checkout.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Final

_MARKERS: Final[tuple[str, ...]] = ("pyproject.toml", "AGENTS.md")


def _repo_root_or_none() -> Path | None:
    """The repository root, or ``None`` when not running inside a source checkout."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if all((parent / m).exists() for m in _MARKERS):
            return parent
    return None


def user_data_dir() -> Path:
    """OS-appropriate per-user application data directory.

    Used for private listener data when AUDIRE is installed outside a source checkout.
    Writing participant data into whatever directory the process happens to be started
    from is not acceptable, and that is what ``cwd()/private`` amounted to.
    """
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":  # pragma: no cover - not exercised on this host
        base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))
    return (base / "audire").resolve()


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Locate the repository root.

    ``AUDIRE_ROOT`` overrides detection. Otherwise the first ancestor directory
    containing both marker files wins; if none is found, the current working directory
    is used so that an installed package still has a usable workspace.
    """
    override = os.environ.get("AUDIRE_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    found = _repo_root_or_none()
    return found if found is not None else Path.cwd().resolve()


def _env_path(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    return Path(raw).expanduser().resolve() if raw else default


def data_dir() -> Path:
    """Root of the data tree. Contains ``sources.yaml`` and ``manifests/``."""
    return _env_path("AUDIRE_DATA_DIR", repo_root() / "data")


def raw_dir() -> Path:
    """Downloaded third-party data. Never tracked by Git."""
    return _env_path("AUDIRE_RAW_DIR", data_dir() / "raw")


def processed_dir() -> Path:
    """Derived datasets. Never tracked by Git."""
    return _env_path("AUDIRE_PROCESSED_DIR", data_dir() / "processed")


def manifests_dir() -> Path:
    """Machine-readable provenance manifests. Tracked by Git."""
    return _env_path("AUDIRE_MANIFESTS_DIR", data_dir() / "manifests")


def literature_dir() -> Path:
    """Hand-transcribed aggregate values from published tables, with provenance."""
    return _env_path("AUDIRE_LITERATURE_DIR", data_dir() / "literature")


def sources_file() -> Path:
    """The external source registry."""
    return _env_path("AUDIRE_SOURCES_FILE", data_dir() / "sources.yaml")


def experiments_dir() -> Path:
    return _env_path("AUDIRE_EXPERIMENTS_DIR", repo_root() / "experiments")


def experiment_configs_dir() -> Path:
    return experiments_dir() / "configs"


def artifacts_dir() -> Path:
    """Regenerable experiment output. Never tracked by Git."""
    return _env_path("AUDIRE_ARTIFACTS_DIR", experiments_dir() / "artifacts")


def private_dir() -> Path:
    """Local storage for real listener profiles and calibration responses.

    Resolution order:

    1. ``AUDIRE_PRIVATE_DIR`` — explicit override, always wins.
    2. ``<repo>/private`` when running inside a source checkout, which is git-ignored and
       is what a developer expects.
    3. :func:`user_data_dir` otherwise. An installed package must never fall back to the
       current working directory: participant data would land wherever the process
       happened to be launched from.

    Nothing under this directory may ever be committed. See docs/SYSTEM_SPEC.md §Privacy.
    """
    override = os.environ.get("AUDIRE_PRIVATE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    checkout = _repo_root_or_none()
    return (checkout / "private").resolve() if checkout else user_data_dir() / "private"


def models_dir() -> Path:
    """Cache directory for downloaded ASR model weights."""
    return _env_path("AUDIRE_MODELS_DIR", repo_root() / "models")


def ensure_runtime_dirs() -> None:
    """Create the directories AUDIRE writes to at runtime."""
    for d in (raw_dir(), processed_dir(), manifests_dir(), artifacts_dir(), private_dir()):
        d.mkdir(parents=True, exist_ok=True)
