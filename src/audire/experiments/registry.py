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
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from audire.config.paths import artifacts_dir, experiments_dir, manifests_dir, repo_root

REGISTRY_SCHEMA_VERSION = 1


def git_sha(short: bool = False) -> str:
    """Current commit SHA, or ``"unknown"`` outside a repository."""
    try:
        args = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
        out = subprocess.run(args, cwd=repo_root(), capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def git_is_dirty() -> bool:
    """Whether the working tree has uncommitted changes.

    Recorded per run: a result produced from a dirty tree is not reproducible from its
    SHA alone, and saying so is better than pretending otherwise.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def lock_hash() -> str:
    """SHA-256 of ``requirements.lock``, identifying the dependency set exactly."""
    lock = repo_root() / "requirements.lock"
    if not lock.exists():
        return "no-lockfile"
    return hashlib.sha256(lock.read_bytes()).hexdigest()[:16]


def data_manifest_ids() -> dict[str, str]:
    """Content digests of every data manifest currently on disk."""
    out: dict[str, str] = {}
    d = manifests_dir()
    if not d.exists():
        return out
    for path in sorted(d.glob("*.json")):
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
    git_dirty: bool
    lock_hash: str
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
        python=sys.version.split()[0],
        platform=f"{platform.system()} {platform.machine()}",
        seeds=list(seeds),
        config=config,
        data_manifests=data_manifest_ids(),
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
    record = new_run(experiment, config, seeds, notes=notes)
    append_run(record)  # visible as `running` from this moment on
    try:
        yield record
    except BaseException as exc:
        fail_run(
            record,
            f"{type(exc).__name__}: {exc}",
            traceback_text="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )
        raise
    else:
        # A runner that already called finish_run keeps its own metrics and status.
        if record.status == "running":
            finish_run(record, record.metrics)
