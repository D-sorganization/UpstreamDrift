#!/usr/bin/env python3
"""Enforce TODO/FIXME discipline: every occurrence must reference a GitHub issue.

Every Python comment line in ``src/``, ``tests/``, and ``scripts/`` that
contains ``TODO`` or ``FIXME`` (matched as whole words, case-insensitively)
must also contain an issue reference of the form ``#<digits>`` on the same
line. Workflow YAML files under ``.github/workflows/`` also reject untracked
``TODO``, ``FIXME``, ``Future:``, and placeholder markers in comments or
placeholder-emitting shell lines.

Valid examples::

    # TODO(#5234): refactor after engine update
    # FIXME(#1234): temporary workaround
    raise NotImplementedError("not yet implemented TODO: #4963")

Invalid examples (each line below violates rule #5922 — missing issue reference)::

    # TODO: refactor this        <- rejected; needs e.g. #5922
    # FIXME - need to clean up   <- rejected; needs e.g. #5922

Exit codes:
    0 — no violations (CI passes)
    1 — at least one violation found (CI fails)

Usage::

    python3 scripts/ci/check_todo_discipline.py
    python3 scripts/ci/check_todo_discipline.py --fix  # no-op, reserved for future use

Filed as part of issue #5922 (TODO/FIXME discipline enforcement).
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_SCAN_ROOTS = (Path("src"), Path("tests"), Path("scripts"))
WORKFLOW_SCAN_ROOT = Path(".github/workflows")

# Match TODO or FIXME as whole words (not substrings like "autodoc"). #5922
_TODO_RE = re.compile(r"\b(?:TODO|FIXME)\b", re.IGNORECASE)
_WORKFLOW_MARKER_RE = re.compile(
    r"\b(?:TODO|FIXME|placeholder)\b|Future:",
    re.IGNORECASE,
)
_WORKFLOW_SHELL_MARKER_RE = re.compile(
    r"\becho\b.*\b(?:TODO|FIXME|placeholder)\b|\becho\b.*Future:",
    re.IGNORECASE,
)

# Issue reference: #<one-or-more-digits>
_ISSUE_RE = re.compile(r"#\d+")


def _is_workflow_file(path: Path) -> bool:
    """Return whether *path* is a GitHub Actions workflow file."""
    return path.suffix in {".yml", ".yaml"} and ".github" in path.parts


def _line_requires_issue(path: Path, raw_line: str) -> bool:
    """Return whether a line contains a tracked-debt marker for this file type."""
    stripped = raw_line.lstrip()
    if _is_workflow_file(path):
        if stripped.startswith("#") and _WORKFLOW_MARKER_RE.search(stripped):
            return True
        return _WORKFLOW_SHELL_MARKER_RE.search(stripped) is not None

    # A Python comment line with TODO/FIXME requires an issue reference. #5922
    return stripped.startswith("#") and _TODO_RE.search(stripped) is not None


def _scan_file(path: Path) -> list[tuple[int, str]]:
    """Return a list of (lineno, line) pairs that violate the discipline rule."""
    violations: list[tuple[int, str]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("skipping non-utf8 file: %s", path)
        return violations

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        if _line_requires_issue(path, raw_line) and not _ISSUE_RE.search(raw_line):
            violations.append((lineno, raw_line))

    return violations


def _iter_scanned_files(repo_root: Path) -> list[Path]:
    """Yield files covered by the TODO discipline gate."""
    paths: list[Path] = []
    for rel_root in PYTHON_SCAN_ROOTS:
        root = repo_root / rel_root
        if root.exists():
            paths.extend(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)

    workflow_root = repo_root / WORKFLOW_SCAN_ROOT
    if workflow_root.exists():
        paths.extend(workflow_root.glob("*.yml"))
        paths.extend(workflow_root.glob("*.yaml"))

    return sorted(paths)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the desired process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fix",
        action="store_true",
        help="No-op flag reserved for future automated fix support.",
    )
    args = parser.parse_args(argv)

    if args.fix:
        logger.info("--fix is a no-op in this version; run manually to add issue refs.")

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not any((REPO_ROOT / root).exists() for root in PYTHON_SCAN_ROOTS):
        logger.error("no configured source roots found under: %s", REPO_ROOT)
        return 1

    scanned_files = _iter_scanned_files(REPO_ROOT)
    all_violations: list[tuple[Path, int, str]] = []

    for path in scanned_files:
        for lineno, line in _scan_file(path):
            all_violations.append((path, lineno, line))

    logger.info("TODO discipline check")
    logger.info("=====================")

    if all_violations:
        for path, lineno, line in all_violations:
            rel = path.relative_to(REPO_ROOT)
            logger.info("%s:%d: %s", rel, lineno, line.rstrip())
        logger.info("")
        logger.error(
            "%d violation%s found. Add a GitHub issue reference (#N) to every tracked-debt marker.",
            len(all_violations),
            "s" if len(all_violations) != 1 else "",
        )
        logger.info(
            "\n%d file%s scanned, %d violation%s found.",
            len(scanned_files),
            "s" if len(scanned_files) != 1 else "",
            len(all_violations),
            "s" if len(all_violations) != 1 else "",
        )
        return 1

    logger.info(
        "%d file%s scanned, 0 violations found.",
        len(scanned_files),
        "s" if len(scanned_files) != 1 else "",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
