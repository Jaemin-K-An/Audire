#!/usr/bin/env python3
"""Reproducible data fetcher for AUDIRE.

Raw data is kept outside Git. This script does not bypass source terms: the primary
CC BY-NC-ND corpus requires the caller to confirm that the dataset card's
creator-notification requirement has been handled, and AUDIRE never sends that message.

Usage
-----
    python scripts/fetch_data.py list
    python scripts/fetch_data.py all-permitted
    python scripts/fetch_data.py zeroth_korean_test
    python scripts/fetch_data.py korean_monosyllabic_speech   # needs the ack env var
    python scripts/fetch_data.py verify
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:  # allow running without an editable install
    sys.path.insert(0, str(_SRC))

from audire.config.logging import configure_logging  # noqa: E402
from audire.config.paths import ensure_runtime_dirs  # noqa: E402
from audire.data.fetch import fetch_all_permitted, fetch_source, verify_all  # noqa: E402
from audire.data.sources import AcknowledgementRequired, registry  # noqa: E402


def cmd_list() -> int:
    reg = registry()
    print(f"{'source id':<32} {'license':<18} {'ack?':<6} status")
    print("-" * 88)
    for sid, src in reg.sources.items():
        ok = src.acknowledgement_satisfied()
        need = "yes" if src.requires_human_acknowledgement else "no"
        status = "ready" if ok else f"BLOCKED (set {src.acknowledgement_env}=1)"
        print(f"{sid:<32} {src.license:<18} {need:<6} {status}")
    print("\nLiterature references (not downloaded):")
    for lid, ref in reg.literature.items():
        print(f"  {lid:<28} {ref.doi}  {ref.journal} {ref.year}")
    return 0


def cmd_verify(deep: bool) -> int:
    results = verify_all(deep=deep)
    if not results:
        print("no manifests found; nothing fetched yet")
        return 0
    failed = 0
    for sid, problems in results.items():
        if problems:
            failed += 1
            print(f"FAIL {sid}: {len(problems)} problem(s)")
            for p in problems[:10]:
                print(f"       - {p}")
        else:
            print(f"OK   {sid}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    ensure_runtime_dirs()

    reg = registry()
    choices = [*reg.sources.keys(), "all-permitted", "list", "verify"]

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("target", choices=choices, help="source id, or a meta-command")
    parser.add_argument(
        "--force", action="store_true", help="re-download even if the manifest verifies"
    )
    parser.add_argument("--shallow", action="store_true", help="verify sizes only, skip checksums")
    args = parser.parse_args(argv)

    if args.target == "list":
        return cmd_list()
    if args.target == "verify":
        return cmd_verify(deep=not args.shallow)

    try:
        if args.target == "all-permitted":
            manifests = fetch_all_permitted(force=args.force)
            if not manifests:
                print(
                    "nothing fetched: all sources need a human acknowledgement step",
                    file=sys.stderr,
                )
                return 0
            for m in manifests:
                print(f"OK {m.source_id}: {m.n_files} files, digest {m.content_digest[:16]}")
            return 0

        manifest = fetch_source(args.target, force=args.force, deep_verify=not args.shallow)
        digest = manifest.content_digest[:16]
        print(f"OK {manifest.source_id}: {manifest.n_files} files, digest {digest}")
        return 0
    except AcknowledgementRequired as exc:
        print(f"\nBLOCKED — human step required\n\n{exc}\n", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
