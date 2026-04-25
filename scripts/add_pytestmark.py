#!/usr/bin/env python3
"""
One-shot script to add pytestmark = pytest.mark.<suite> to test files that lack it.
Also adds `import pytest` if not already present.

Usage:
    python3 scripts/add_pytestmark.py
"""
from __future__ import annotations

import os
import re
import sys

SUITES = {
    "tests/unit": "unit",
    "tests/integration": "integration",
}


def process_file(path: str, marker: str) -> bool:
    """Return True if the file was modified."""
    with open(path, encoding="utf-8") as fh:
        content = fh.read()

    if "pytestmark" in content:
        return False  # already decorated

    lines = content.split("\n")

    has_pytest_import = any(
        re.match(r"^import pytest\b", line) or re.match(r"^from pytest\b", line)
        for line in lines
    )

    # Find insertion point: after the last top-level import line
    # We scan for lines starting with `import ` or `from ` that are not
    # inside a try/except block (heuristic: indentation == 0).
    insert_at = 0
    for i, line in enumerate(lines):
        if re.match(r"^(import |from )\S", line):
            insert_at = i + 1

    # Build the lines to insert
    to_insert: list[str] = []
    if not has_pytest_import:
        to_insert.append("import pytest")
    to_insert.append(f"pytestmark = pytest.mark.{marker}")
    to_insert.append("")  # blank line after

    new_lines = lines[:insert_at] + to_insert + lines[insert_at:]
    new_content = "\n".join(new_lines)

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(new_content)

    return True


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    total_modified = 0

    for suite_rel, marker in SUITES.items():
        suite_dir = os.path.join(repo_root, suite_rel)
        if not os.path.isdir(suite_dir):
            print(f"  [skip] {suite_rel} does not exist", file=sys.stderr)
            continue

        modified = 0
        for dirpath, _dirs, filenames in os.walk(suite_dir):
            for fname in filenames:
                if fname.startswith("test_") and fname.endswith(".py"):
                    fpath = os.path.join(dirpath, fname)
                    if process_file(fpath, marker):
                        modified += 1
                        rel = os.path.relpath(fpath, repo_root)
                        print(f"  + {rel}")

        print(f"[{suite_rel}] modified {modified} files")
        total_modified += modified

    print(f"\nTotal files modified: {total_modified}")


if __name__ == "__main__":
    main()
