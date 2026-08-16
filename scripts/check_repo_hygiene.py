#!/usr/bin/env python3
"""Privacy and licence hygiene gate.

Fails if the repository tracks anything it must never track: participant-level hearing
data, raw third-party corpora, model weights, or secrets. Run in CI and available as a
pre-commit check.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Path prefixes that must never appear in the Git index.
FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "data/raw/",
    "data/processed/",
    "data/stimuli/",
    "private/",
    "profiles_local/",
    "models/",
    "experiments/artifacts/",
)

#: Extensions that indicate bundled media or weights.
FORBIDDEN_SUFFIXES: tuple[str, ...] = (
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".mp4",
    ".mkv",
    ".mov",
    ".bin",
    ".ct2",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
)

#: Filenames that indicate participant-level data regardless of location.
FORBIDDEN_NAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\.participant\.(json|csv|ya?ml)$"),
    re.compile(r"^audiogram.*\.(csv|json)$", re.IGNORECASE),
    re.compile(r"(^|/)consent.*\.(pdf|docx?)$", re.IGNORECASE),
)

#: Crude secret detection over tracked text files.
SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("AWS access key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("private key block", re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{34,}\b")),
    (
        "generic api key assignment",
        re.compile(r"(?i)\b(api[_-]?key|secret|password)\s*=\s*['\"][^'\"]{16,}['\"]"),
    ),
)

TEXT_SUFFIXES = frozenset(
    {
        ".py",
        ".yaml",
        ".yml",
        ".json",
        ".toml",
        ".md",
        ".txt",
        ".cfg",
        ".ini",
        ".html",
        ".js",
        ".css",
    }
)

#: This file necessarily contains the patterns it searches for.
SELF = "scripts/check_repo_hygiene.py"


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    return [line for line in out.stdout.splitlines() if line]


def main() -> int:
    try:
        files = tracked_files()
    except subprocess.CalledProcessError:
        print("not a git repository; hygiene gate skipped", file=sys.stderr)
        return 0

    problems: list[str] = []

    for rel in files:
        if any(rel.startswith(p) for p in FORBIDDEN_PREFIXES) and not rel.endswith(".gitkeep"):
            problems.append(f"tracked file in a forbidden location: {rel}")
        if rel.endswith(FORBIDDEN_SUFFIXES):
            problems.append(f"tracked media/weights file: {rel}")
        for pat in FORBIDDEN_NAME_PATTERNS:
            if pat.search(rel):
                problems.append(f"tracked file looks like participant data: {rel}")

    for rel in files:
        if rel == SELF or Path(rel).suffix not in TEXT_SUFFIXES:
            continue
        path = ROOT / rel
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for label, pat in SECRET_PATTERNS:
            if pat.search(text):
                problems.append(f"possible {label} in {rel}")

    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for required in ("data/raw/", "private/", "data/processed/"):
        if required not in gitignore:
            problems.append(f".gitignore is missing a required entry: {required}")

    if problems:
        print("REPOSITORY HYGIENE FAILURES:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    print(f"repository hygiene OK ({len(files)} tracked files checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
