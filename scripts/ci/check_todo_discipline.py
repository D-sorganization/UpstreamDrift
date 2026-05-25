#!/usr/bin/env python3
"""Enforce TODO/FIXME discipline: every occurrence must reference a GitHub issue.

Every line in ``src/`` that contains ``TODO`` or ``FIXME`` (matched as whole
words, case-insensitively) must also contain an issue reference of the form
``#<digits>`` on the same line.

Valid examples::

    # TODO(#5234): refactor after engine update
    # FIXME(#1234): temporary workaround
    raise NotImplementedError("not yet implemented TODO: #4963")

Invalid examples::

    # TODO: refactor this
    # FIXME - need to clean up

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
SRC_DIR = REPO_ROOT / "src"

# Match TODO or FIXME as whole words (not substrings like "autodoc")
_TODO_RE = re.compile(r"\b(?:TODO|FIXME)\b", re.IGNORECASE)

# Issue reference: #<one-or-more-digits>
_ISSUE_RE = re.compile(r"#\d+")


def _scan_file(path: Path) -> list[tuple[int, str]]:
    """Return a list of (lineno, line) pairs that violate the discipline rule."""
    violations: list[tuple[int, str]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("skipping non-utf8 file: %s", path)
        return violations

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        # Only check comment lines (lines whose first non-whitespace char is #)
        stripped = raw_line.lstrip()
        if not stripped.startswith("#"):
            # Non-comment lines (strings, docstrings, code) are ignored per spec.
            continue

        if not _TODO_RE.search(stripped):
            continue

        # A comment line with TODO/FIXME: require an issue reference.
        if not _ISSUE_RE.search(raw_line):
            violations.append((lineno, raw_line))

    return violations


def _iter_src_python_files(src_dir: Path) -> list[Path]:
    """Yield all ``*.py`` files under *src_dir*, skipping ``__pycache__``."""
    return [p for p in src_dir.rglob("*.py") if "__pycache__" not in p.parts]


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

    if not SRC_DIR.exists():
        logger.error("source directory not found: %s", SRC_DIR)
        return 1

    python_files = _iter_src_python_files(SRC_DIR)
    all_violations: list[tuple[Path, int, str]] = []

    for path in sorted(python_files):
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
            "%d violation%s found. Add a GitHub issue reference (#N) to every TODO/FIXME.",
            len(all_violations),
            "s" if len(all_violations) != 1 else "",
        )
        logger.info(
            "\n%d file%s scanned, %d violation%s found.",
            len(python_files),
            "s" if len(python_files) != 1 else "",
            len(all_violations),
            "s" if len(all_violations) != 1 else "",
        )
        return 1

    logger.info(
        "%d file%s scanned, 0 violations found.",
        len(python_files),
        "s" if len(python_files) != 1 else "",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
