#!/usr/bin/env python3
"""Freeze the deprecated upstream_drift_tools package to its shim surface.

Issue #5922 called out that the deprecated alias package still exists even
though imports should move to ``sidekick``. We cannot remove the package in one
wave because existing consumers still rely on its compatibility shims, but we
can stop the deprecated surface from growing.

This guard intentionally uses a subset ratchet:

- the package may shrink as cleanup work lands,
- the package may disappear entirely,
- but it may not gain new files beyond the current shim-only surface.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_ALIAS_ROOT = Path("src/shared/python/upstream_drift_tools")
ALLOWED_ALIAS_FILES = frozenset(
    {
        "src/shared/python/upstream_drift_tools/__init__.py",
        "src/shared/python/upstream_drift_tools/ui/tools_sidebar/__init__.py",
        "src/shared/python/upstream_drift_tools/ui/tools_sidebar/default_tabs.py",
    }
)
IGNORED_FILE_SUFFIXES = {".pyc", ".pyo"}
IGNORED_DIR_NAMES = {"__pycache__"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _iter_alias_files(repo_root: Path) -> list[Path]:
    alias_root = repo_root / DEFAULT_ALIAS_ROOT
    if not alias_root.exists():
        return []

    files: list[Path] = []
    for candidate in alias_root.rglob("*"):
        if not candidate.is_file():
            continue
        rel_parts = candidate.relative_to(repo_root).parts
        if any(part in IGNORED_DIR_NAMES for part in rel_parts):
            continue
        if candidate.suffix in IGNORED_FILE_SUFFIXES:
            continue
        files.append(candidate)
    return sorted(files)


def find_unexpected_alias_files(repo_root: Path) -> list[str]:
    """Return deprecated-alias files that fall outside the shim allowlist."""
    unexpected: list[str] = []
    for candidate in _iter_alias_files(repo_root):
        rel = candidate.relative_to(repo_root).as_posix()
        if rel not in ALLOWED_ALIAS_FILES:
            unexpected.append(rel)
    return unexpected


def main() -> int:
    repo_root = _repo_root()
    unexpected = find_unexpected_alias_files(repo_root)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if unexpected:
        logger.error(
            "FAIL: deprecated upstream_drift_tools shim surface grew unexpectedly:\n"
        )
        for rel in unexpected:
            logger.error("  %s", rel)
        logger.error(
            "\nMove new code under sidekick/ instead, or delete the deprecated alias file."
        )
        return 1

    logger.info("OK: deprecated upstream_drift_tools package is shim-only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
