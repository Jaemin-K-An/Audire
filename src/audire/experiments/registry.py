"""Experiment provenance: the record that makes a reported number reproducible.

Every run appends an entry to ``experiments/registry.yaml`` containing the git SHA, the
dependency lock hash, the exact config, the seed list, the data manifest ids in use, the
metric artifact paths and the run status. A number that is not traceable to an entry here
must not appear in ``docs/RESULTS.md``.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import secrets
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


class RegistryCollision(Exception):
    """Two different runs claimed the same ``run_id``."""


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
    #: SHA-256 of each artifact at the moment the run wrote it, keyed by the same relative
    #: path that appears in :attr:`artifacts`. Without it the registry records only that a
    #: file was produced, not what was in it, so a results file edited after the fact is
    #: indistinguishable from the one the run actually generated. Kept as a separate field
    #: rather than restructuring ``artifacts`` so that runs recorded before this existed
    #: still load and stay reproducible.
    artifact_digests: dict[str, str] = field(default_factory=dict)
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
    # The random suffix is what makes the id unique. Experiment name + second-resolution
    # timestamp + commit collide whenever two runs of the same experiment start in the
    # same second — easy to hit with fast configs, loops, or parallel invocations — and
    # `append_run` replaces by run_id, so the earlier run's record was silently destroyed.
    token = secrets.token_hex(3)
    return RunRecord(
        run_id=f"{experiment}-{now:%Y%m%dT%H%M%SZ}-{sha}-{token}",
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
    """Append, or update in place, a run record in the registry.

    Replacing by ``run_id`` is how a run's own lifecycle works: ``new_run`` writes a
    ``running`` record and ``finish_run``/``fail_run`` replace it with the final one. That
    is only safe while a ``run_id`` identifies exactly one run, so a record whose
    ``started_at_utc`` disagrees with the stored one is treated as a collision and
    refused. Overwriting it would delete a completed experiment without a trace.
    """
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"schema_version": REGISTRY_SCHEMA_VERSION, "runs": []}
    if path.exists():
        existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for stored in existing.get("runs", []):
            if stored.get("run_id") != record.run_id:
                payload["runs"].append(stored)
                continue
            if stored.get("started_at_utc") != record.started_at_utc:
                raise RegistryCollision(
                    f"run_id {record.run_id!r} 가 이미 다른 실행에 쓰이고 있습니다 "
                    f"(기록된 시작 시각 {stored.get('started_at_utc')}, "
                    f"덮어쓰려는 실행 {record.started_at_utc}). 기존 기록을 덮어쓰지 않습니다."
                )
    payload["runs"].append(record.to_dict())
    _atomic_write(path, yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=100))
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
    """Write a JSON artifact, then register its path **and its digest**."""
    path = run_artifact_dir(record) / name
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
    _atomic_write(path, text)
    rel = str(path.relative_to(repo_root())) if path.is_relative_to(repo_root()) else str(path)
    if rel not in record.artifacts:
        record.artifacts.append(rel)
    record.artifact_digests[rel] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return path


def verify_artifacts(run_id: str) -> dict[str, str]:
    """Re-hash a recorded run's artifacts and report the state of each one.

    Values are ``"match"``, ``"modified"``, ``"missing"``, or ``"not_recorded"`` for
    artifacts written before digests existed. Reporting ``"not_recorded"`` rather than
    quietly passing keeps the difference between "verified" and "unverifiable" visible.
    """
    record = next((r for r in load_runs() if r.get("run_id") == run_id), None)
    if record is None:
        raise KeyError(f"기록되지 않은 run_id: {run_id!r}")

    digests: dict[str, str] = record.get("artifact_digests") or {}
    out: dict[str, str] = {}
    for rel in record.get("artifacts", []):
        path = repo_root() / rel
        if not path.exists():
            out[rel] = "missing"
        elif rel not in digests:
            out[rel] = "not_recorded"
        else:
            actual = hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
            out[rel] = "match" if actual == digests[rel] else "modified"
    return out


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temporary file and :func:`os.replace`.

    A plain ``write_text`` truncates the destination first, so a crash or a full disk
    part-way through leaves a truncated file. For the registry that means losing the
    record of every run, not just the one being written.
    """
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)  # noqa: PTH105 - patched in tests to simulate a failed replace
    finally:
        tmp.unlink(missing_ok=True)


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
