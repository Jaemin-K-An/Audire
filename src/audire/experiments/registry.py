"""Experiment provenance: the record that makes a reported number reproducible.

Every run appends an entry to ``experiments/registry.yaml`` containing the git SHA, the
dependency lock hash, the exact config, the seed list, the data manifest ids in use, the
metric artifact paths and the run status. A number that is not traceable to an entry here
must not appear in ``docs/RESULTS.md``.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import traceback
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from audire.config.paths import artifacts_dir, experiments_dir, manifests_dir, repo_root
from audire.data.manifest import accessed_sources, reset_accessed

REGISTRY_SCHEMA_VERSION = 1


def _git(*args: str) -> str | None:
    """Run a git command, returning ``None`` when git state cannot be determined."""
    try:
        out = subprocess.run(
            ["git", *args], cwd=repo_root(), capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def git_sha(short: bool = False) -> str:
    """Current commit SHA, or ``"unknown"`` when git state cannot be determined."""
    out = _git("rev-parse", *(["--short"] if short else []), "HEAD")
    return out if out else "unknown"


def git_is_dirty() -> bool | str:
    """Whether the working tree has uncommitted changes, or ``"unknown"``.

    Returns the string ``"unknown"`` rather than ``False`` when git cannot be queried.
    Recording a clean tree we never verified would be a false provenance claim, and a
    result produced from a dirty tree is not reproducible from its SHA alone.
    """
    out = _git("status", "--porcelain")
    return "unknown" if out is None else bool(out)


def lock_hash() -> str:
    """SHA-256 of ``requirements.lock``, identifying the dependency set exactly."""
    lock = repo_root() / "requirements.lock"
    if not lock.exists():
        return "no-lockfile"
    return hashlib.sha256(lock.read_bytes()).hexdigest()[:16]


def _installed_distributions() -> dict[str, str]:
    """Name -> version for every distribution actually importable right now."""
    import importlib.metadata as md

    out: dict[str, str] = {}
    for dist in md.distributions():
        name = dist.metadata["Name"]
        if name:
            out[name.lower().replace("_", "-")] = dist.version or "unknown"
    return out


def environment_fingerprint() -> dict[str, Any]:
    """Fingerprint the environment that is **actually installed**, not a declared file.

    Hashing ``requirements.lock`` alone proves nothing: the file may bear no relation to
    the interpreter running the experiment. This reads the live distribution metadata, so
    the recorded digest changes whenever the environment a result was produced in changes.
    """
    dists = _installed_distributions()
    h = hashlib.sha256()
    h.update(sys.version.encode("utf-8"))
    h.update(b"\0")
    for name in sorted(dists):
        h.update(f"{name}=={dists[name]}\n".encode())
    return {
        "digest": h.hexdigest(),
        "python": sys.version.split()[0],
        "n_distributions": len(dists),
        "distributions": dists,
    }


def environment_matches_lock() -> dict[str, Any]:
    """Compare the declared lockfile against the installed environment.

    Reports ``match`` / ``mismatch`` / ``no_lockfile`` / ``unknown`` plus the specific
    differences, so a run recorded against a lockfile it did not actually use is visible
    instead of implied.
    """
    lock = repo_root() / "requirements.lock"
    if not lock.exists():
        return {"status": "no_lockfile", "differences": []}
    try:
        text = lock.read_text(encoding="utf-8")
    except OSError:  # pragma: no cover - unreadable lockfile
        return {"status": "unknown", "differences": []}

    declared: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, _, version = line.partition("==")
        declared[name.strip().lower().replace("_", "-")] = version.strip().split(" ")[0]

    installed = _installed_distributions()
    differences = [
        {"package": name, "declared": version, "installed": installed.get(name, "MISSING")}
        for name, version in sorted(declared.items())
        if installed.get(name) != version
    ]
    return {
        "status": "match" if not differences else "mismatch",
        "n_declared": len(declared),
        "n_differences": len(differences),
        # Truncated: a wholesale mismatch should not bloat every registry entry.
        "differences": differences[:20],
    }


def data_manifest_ids(only: Iterable[str] | None = None) -> dict[str, str]:
    """Content digests of data manifests.

    ``only`` restricts the result to sources an experiment actually consumed. Recording
    every manifest that happens to sit on disk would attribute datasets to a run that
    never read them.
    """
    out: dict[str, str] = {}
    d = manifests_dir()
    if not d.exists():
        return out
    wanted = None if only is None else set(only)
    for path in sorted(d.glob("*.json")):
        if wanted is not None and path.stem not in wanted:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):  # pragma: no cover - corrupt manifest
            continue
        out[path.stem] = str(payload.get("content_digest", "unknown"))
    return out


@dataclass(slots=True)
class RunRecord:
    """One experiment run."""

    run_id: str
    experiment: str
    started_at_utc: str
    git_sha: str
    git_dirty: bool | str
    lock_hash: str
    #: Digest of the environment that is actually installed (P0.3).
    env_fingerprint: str
    #: Whether the declared lockfile matches that environment.
    env_matches_lock: str
    python: str
    platform: str
    seeds: list[int]
    config: dict[str, Any]
    data_manifests: dict[str, str] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    status: str = "running"
    finished_at_utc: str | None = None
    error: str | None = None
    #: Traceback of the failure. An error string alone does not say where it happened,
    #: which is exactly what someone re-running a failed experiment needs to know.
    error_traceback: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def new_run(
    experiment: str, config: dict[str, Any], seeds: list[int], notes: str = ""
) -> RunRecord:
    """Start a run record. The run id encodes the experiment, time and commit."""
    now = datetime.now(UTC)
    sha = git_sha(short=True)
    return RunRecord(
        run_id=f"{experiment}-{now:%Y%m%dT%H%M%SZ}-{sha}",
        experiment=experiment,
        started_at_utc=now.isoformat(),
        git_sha=git_sha(),
        git_dirty=git_is_dirty(),
        lock_hash=lock_hash(),
        env_fingerprint=environment_fingerprint()["digest"],
        env_matches_lock=str(environment_matches_lock()["status"]),
        python=sys.version.split()[0],
        platform=f"{platform.system()} {platform.machine()}",
        seeds=list(seeds),
        config=config,
        # Filled in when the run ends, from the sources it actually consumed.
        data_manifests={},
        notes=notes,
    )


def registry_path() -> Path:
    return experiments_dir() / "registry.yaml"


def append_run(record: RunRecord) -> Path:
    """Append (or replace by ``run_id``) a run record in the registry."""
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"schema_version": REGISTRY_SCHEMA_VERSION, "runs": []}
    if path.exists():
        existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        payload["runs"] = [r for r in existing.get("runs", []) if r.get("run_id") != record.run_id]
    payload["runs"].append(record.to_dict())
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
    )
    return path


def load_runs() -> list[dict[str, Any]]:
    path = registry_path()
    if not path.exists():
        return []
    return list((yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("runs", []))


def run_artifact_dir(record: RunRecord) -> Path:
    """Directory for this run's artifacts. Regenerable, never committed."""
    d = artifacts_dir() / record.run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_artifact(record: RunRecord, name: str, payload: Any) -> Path:
    """Write a JSON artifact and register its path on the run record."""
    path = run_artifact_dir(record) / name
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    rel = str(path.relative_to(repo_root())) if path.is_relative_to(repo_root()) else str(path)
    if rel not in record.artifacts:
        record.artifacts.append(rel)
    return path


def _json_default(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"not JSON-serialisable: {type(obj).__name__}")


def finish_run(record: RunRecord, metrics: dict[str, Any], status: str = "completed") -> Path:
    record.metrics = metrics
    record.status = status
    record.finished_at_utc = datetime.now(UTC).isoformat()
    return append_run(record)


def fail_run(record: RunRecord, error: str, traceback_text: str | None = None) -> Path:
    """Record a failed run rather than leaving a silent gap in the registry."""
    record.status = "failed"
    record.error = error
    record.error_traceback = traceback_text
    record.finished_at_utc = datetime.now(UTC).isoformat()
    return append_run(record)


@contextmanager
def tracked_run(
    experiment: str,
    config: dict[str, Any],
    seeds: list[int],
    notes: str = "",
) -> Iterator[RunRecord]:
    """Run an experiment under a lifecycle that the registry always observes.

    The registry entry is written **on entry**, with ``status="running"``, so an
    experiment that is still executing — or that was killed without unwinding — is
    visible rather than invisible.

    Previously the two main runners called ``new_run()`` and then ``finish_run()`` with
    no exception handling. ``new_run()`` only *constructs* a record; nothing reached the
    registry until ``finish_run``. A run that raised part-way through therefore left **no
    trace at all**, which is worse than being recorded as failed: a reader could not tell
    the run had ever been attempted.

    On any exception — including :class:`BaseException` such as ``KeyboardInterrupt``,
    because a long sweep is routinely interrupted — the record is marked ``failed`` with
    the error and its traceback, and the exception is re-raised unchanged.
    """
    reset_accessed()
    record = new_run(experiment, config, seeds, notes=notes)
    append_run(record)  # visible as `running` from this moment on
    try:
        yield record
    except BaseException as exc:
        record.data_manifests = data_manifest_ids(only=accessed_sources())
        fail_run(
            record,
            f"{type(exc).__name__}: {exc}",
            traceback_text="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )
        raise
    else:
        # Only the datasets this run actually verified and read are attributed to it.
        record.data_manifests = data_manifest_ids(only=accessed_sources())
        # A runner that already called finish_run keeps its own metrics and status.
        finish_run(record, record.metrics) if record.status == "running" else append_run(record)
