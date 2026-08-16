"""Idempotent, restartable acquisition of the registered external datasets.

Rules enforced here
-------------------
* Only sources present in ``data/sources.yaml`` can be fetched.
* A source whose registry entry sets ``requires_human_acknowledgement`` refuses to
  download until the named environment variable is set. AUDIRE never performs the human
  step (it does not send email and does not accept terms on anyone's behalf).
* Every successful fetch writes a manifest with a SHA-256 for every file.
* Declared expectations (row counts, file names) are checked and a mismatch is a hard
  error, because a silently changed upstream revision would invalidate every result
  derived from it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audire.config.logging import get_logger
from audire.config.paths import raw_dir
from audire.data.manifest import Manifest
from audire.data.sources import Source, SourceRegistry, registry

log = get_logger(__name__)


class FetchError(RuntimeError):
    """Raised when acquisition fails or the acquired data does not match expectations."""


def local_path_for(source: Source) -> Path:
    """Where ``source`` is stored locally."""
    return raw_dir() / source.id


# --------------------------------------------------------------------------- backends


def _fetch_huggingface(source: Source, dest: Path) -> dict[str, Any]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise FetchError(
            "huggingface_hub is required for this source. Install with: pip install -e '.[data]'"
        ) from exc

    if not source.repo_id:
        raise FetchError(f"source {source.id!r} is kind=huggingface_dataset but has no repo_id")

    allow_patterns = source.expected.get("allow_patterns")
    log.info(
        "fetch.huggingface.start",
        source=source.id,
        repo=source.repo_id,
        revision=source.revision,
        dest=str(dest),
    )
    path = snapshot_download(
        repo_id=source.repo_id,
        repo_type=source.repo_type or "dataset",
        revision=source.revision,
        local_dir=str(dest),
        allow_patterns=allow_patterns,
        # snapshot_download resumes partial downloads and skips files already present,
        # which is what makes repeated invocations idempotent and restartable.
    )
    return {"backend": "huggingface_hub.snapshot_download", "resolved_path": str(path)}


def _fetch_zenodo(source: Source, dest: Path) -> dict[str, Any]:
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise FetchError(
            "requests is required for this source. Install with: pip install -e '.[data]'"
        ) from exc

    if not source.record_id:
        raise FetchError(f"source {source.id!r} is kind=zenodo_record but has no record_id")

    api = f"https://zenodo.org/api/records/{source.record_id}"
    log.info("fetch.zenodo.start", source=source.id, api=api)
    resp = requests.get(api, timeout=60)
    resp.raise_for_status()
    meta = resp.json()

    dest.mkdir(parents=True, exist_ok=True)
    (dest / "record.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Verify the live licence still matches what the registry claims. A silent upstream
    # licence change must not be inherited without review.
    live_license = ((meta.get("metadata") or {}).get("license") or {}).get("id")
    if live_license and live_license.upper() != source.license.upper():
        raise FetchError(
            f"licence mismatch for {source.id}: registry says {source.license}, "
            f"Zenodo record now says {live_license}. Re-verify data/sources.yaml before use."
        )

    for item in meta.get("files", []):
        key = item["key"]
        url = (item.get("links") or {}).get("content") or (item.get("links") or {}).get("self")
        if not url:
            continue
        out = dest / key
        if out.exists() and item.get("size") and out.stat().st_size == int(item["size"]):
            log.info("fetch.zenodo.skip_existing", file=key)
            continue
        log.info("fetch.zenodo.file", file=key, bytes=item.get("size"))
        tmp = out.with_suffix(out.suffix + ".part")
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_content(1 << 20):
                    if chunk:
                        f.write(chunk)
        tmp.replace(out)

    return {"backend": "zenodo REST API", "api": api, "live_license": live_license}


_BACKENDS = {
    "huggingface_dataset": _fetch_huggingface,
    "zenodo_record": _fetch_zenodo,
}


# --------------------------------------------------------------------------- checks


def _run_expectation_checks(source: Source, dest: Path) -> dict[str, Any]:
    """Validate declared expectations. Raises :class:`FetchError` on mismatch."""
    checks: dict[str, Any] = {}
    exp = source.expected

    if "metadata_file" in exp:
        meta = dest / str(exp["metadata_file"])
        checks["metadata_file_present"] = meta.exists()
        if not meta.exists():
            raise FetchError(f"{source.id}: expected metadata file {meta} is missing")

    if "audio_dir" in exp:
        audio = dest / str(exp["audio_dir"])
        n_audio = sum(1 for _ in audio.glob("*.wav")) if audio.is_dir() else 0
        checks["n_audio_files"] = n_audio
        expected_n = exp.get("n_utterances")
        if expected_n is not None and n_audio != int(expected_n):
            raise FetchError(
                f"{source.id}: expected {expected_n} audio files in {audio}, found {n_audio}"
            )

    if "parquet_file" in exp:
        pq = dest / str(exp["parquet_file"])
        checks["parquet_present"] = pq.exists()
        if not pq.exists():
            raise FetchError(f"{source.id}: expected parquet shard {pq} is missing")
        n_rows = _parquet_rows(pq)
        checks["n_rows"] = n_rows
        expected_rows = exp.get("n_rows")
        if expected_rows is not None and n_rows != int(expected_rows):
            raise FetchError(
                f"{source.id}: expected {expected_rows} rows in {pq.name}, found {n_rows}"
            )

    for name in exp.get("files", []) or []:
        present = (dest / str(name)).exists()
        checks[f"file:{name}"] = present
        if not present:
            raise FetchError(f"{source.id}: expected file {name!r} is missing")

    return checks


def _parquet_rows(path: Path) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise FetchError("pyarrow is required to validate parquet row counts") from exc
    return int(pq.ParquetFile(path).metadata.num_rows)


# --------------------------------------------------------------------------- entry points


def fetch_source(
    source_id: str,
    *,
    reg: SourceRegistry | None = None,
    force: bool = False,
    deep_verify: bool = True,
) -> Manifest:
    """Fetch one registered source and write its manifest.

    Idempotent: if a manifest exists and still verifies, the download is skipped unless
    ``force`` is set.
    """
    reg = reg or registry()
    source = reg.get(source_id)
    source.require_acknowledgement()

    dest = local_path_for(source)
    manifest_path = Manifest(
        source_id=source.id,
        license=source.license,
        homepage=source.homepage,
        revision=source.revision,
        retrieved_at_utc="",
        local_path=str(dest),
        files=[],
    ).path()

    if manifest_path.exists() and not force:
        existing = Manifest.load(source.id)
        problems = existing.verify(deep=deep_verify)
        if existing.revision != source.revision:
            problems.append(
                f"registry revision changed: manifest={existing.revision!r}, "
                f"registry={source.revision!r}"
            )
        if existing.expected != dict(source.expected):
            problems.append("registry expectations changed since the manifest was written")
        if not problems:
            log.info("fetch.skip_verified", source=source.id, n_files=existing.n_files)
            return existing
        log.warning("fetch.manifest_stale", source=source.id, problems=problems[:5])

    backend = _BACKENDS.get(source.kind)
    if backend is None:
        raise FetchError(f"no fetch backend for source kind {source.kind!r}")

    dest.mkdir(parents=True, exist_ok=True)
    backend_info = backend(source, dest)
    checks = _run_expectation_checks(source, dest)

    manifest = Manifest.build(
        source_id=source.id,
        license=source.license,
        local_path=dest,
        homepage=source.homepage,
        revision=source.revision,
        expected=dict(source.expected),
        checks={**checks, **backend_info},
        notes=(
            f"Permitted: {'; '.join(source.permitted_uses)}. "
            f"Prohibited: {'; '.join(source.prohibited_uses)}."
        ),
    )
    written = manifest.save()
    log.info(
        "fetch.complete",
        source=source.id,
        n_files=manifest.n_files,
        total_bytes=manifest.total_bytes,
        content_digest=manifest.content_digest[:16],
        manifest=str(written),
    )
    return manifest


def fetch_all_permitted(
    *, reg: SourceRegistry | None = None, force: bool = False
) -> list[Manifest]:
    """Fetch every source that does not require an outstanding human acknowledgement.

    Sources needing a human step are reported and skipped rather than blocking the run,
    so that a fresh evaluator can still prepare everything else with one command.
    """
    reg = reg or registry()
    out: list[Manifest] = []
    for source_id, source in reg.sources.items():
        if not source.acknowledgement_satisfied():
            log.warning(
                "fetch.skipped_needs_human_step",
                source=source_id,
                env=source.acknowledgement_env,
                contact=source.creator_contact,
            )
            continue
        out.append(fetch_source(source_id, reg=reg, force=force))
    return out


def verify_all(*, reg: SourceRegistry | None = None, deep: bool = True) -> dict[str, list[str]]:
    """Verify every manifest that exists on disk. Returns ``{source_id: problems}``."""
    reg = reg or registry()
    results: dict[str, list[str]] = {}
    for source_id in reg.sources:
        try:
            manifest = Manifest.load(source_id)
        except FileNotFoundError:
            continue
        results[source_id] = manifest.verify(deep=deep)
    return results
