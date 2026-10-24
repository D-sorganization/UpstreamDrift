#!/usr/bin/env python3
"""Ratchet: tracked files must not grow Windows home-path or local-denylist leaks.

Committed patterns live in ``scripts/config/private_path_guard_patterns.json``.
Optional extra regexes (fleet hostnames, etc.) may be listed one per line in
``scripts/config/fleet_hostname_denylist.local.txt`` (gitignored; never commit
real hostnames into the committed pattern file).

Exit 0 when every pattern count is at or below the baseline; exit 1 when any
count exceeds baseline; exit 2 on configuration errors.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PATTERNS_PATH = REPO_ROOT / "scripts" / "config" / "private_path_guard_patterns.json"
LOCAL_DENYLIST_PATH = (
    REPO_ROOT / "scripts" / "config" / "fleet_hostname_denylist.local.txt"
)
BASELINE_PATH = REPO_ROOT / "scripts" / "config" / "private_path_leaks_baseline.json"

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".txt",
    ".qmd",
    ".tex",
    ".m",
    ".csv",
    ".mdl",
    ".sh",
    ".ps1",
    ".html",
    ".js",
    ".ts",
    ".tsx",
    ".rs",
    ".ini",
    ".cfg",
}

SELF_EXEMPT = {
    REPO_ROOT / "scripts" / "ci" / "check_private_path_leaks.py",
    PATTERNS_PATH,
    BASELINE_PATH,
}


def _load_patterns() -> dict[str, re.Pattern[str]]:
    raw = json.loads(PATTERNS_PATH.read_text(encoding="utf-8"))
    patterns: dict[str, re.Pattern[str]] = {}
    for entry in raw.get("patterns", []):
        pattern_id = entry["id"]
        patterns[pattern_id] = re.compile(entry["regex"])
    if LOCAL_DENYLIST_PATH.is_file():
        for line in LOCAL_DENYLIST_PATH.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            local_id = f"local_{len(patterns)}"
            patterns[local_id] = re.compile(stripped)
    return patterns


def _tracked_text_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    paths: list[Path] = []
    for rel in result.stdout.split("\0"):
        if not rel:
            continue
        path = REPO_ROOT / rel
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path in SELF_EXEMPT:
            continue
        paths.append(path)
    return paths


def count_matches(patterns: dict[str, re.Pattern[str]]) -> dict[str, int]:
    counts = dict.fromkeys(patterns, 0)
    for path in _tracked_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            logger.warning("skipping unreadable file: %s", path)
            continue
        for pattern_id, compiled in patterns.items():
            counts[pattern_id] += len(compiled.findall(text))
    return counts


def _load_baseline() -> dict[str, int]:
    data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in data.get("counts", data).items()}


def _save_baseline(counts: dict[str, int]) -> None:
    payload = {"counts": counts}
    BASELINE_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Lower baseline counts to current totals (never raises).",
    )
    args = parser.parse_args(argv)

    if not PATTERNS_PATH.is_file():
        logger.error("missing patterns file: %s", PATTERNS_PATH)
        return 2
    if not BASELINE_PATH.is_file() and not args.update_baseline:
        logger.error("missing baseline: %s (run with --update-baseline once)", BASELINE_PATH)
        return 2

    patterns = _load_patterns()
    current = count_matches(patterns)

    if args.update_baseline:
        baseline = _load_baseline() if BASELINE_PATH.is_file() else {}
        merged = {k: min(baseline.get(k, current[k]), current[k]) for k in current}
        for key, value in baseline.items():
            merged.setdefault(key, value)
        _save_baseline(merged)
        logger.info("Updated baseline: %s", merged)
        return 0

    baseline = _load_baseline()
    violations: list[str] = []
    for pattern_id, total in current.items():
        allowed = baseline.get(pattern_id, 0)
        if total > allowed:
            violations.append(f"{pattern_id}: {total} > baseline {allowed}")
    if violations:
        logger.error("Private path leak ratchet failed:")
        for line in violations:
            logger.error("  %s", line)
        return 1
    logger.info("Private path leak ratchet: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
