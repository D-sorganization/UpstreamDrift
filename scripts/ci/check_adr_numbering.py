#!/usr/bin/env python3
"""CI check: detect duplicate ADR numbers in docs/adr/.

Scans all .md files whose names begin with a 4-digit prefix (e.g. 0005-...).
Exits 0 when every prefix is unique, exits 1 and lists every collision otherwise.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

# Files that do not follow the numbered-ADR naming convention.
_SKIP = {"ADR_TEMPLATE.md", "README.md", "api-versioning.md"}

_NUMBERED_RE = re.compile(r"^(\d{4})-")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def check_adr_numbering(adr_dir: Path) -> int:
    """Return the number of duplicate-number conflicts found.

    Prints a report to stdout.  Callers should exit with this value (capped at
    1 for shell exit-code purposes).
    """
    print("ADR numbering check")
    print("===================")

    by_number: dict[str, list[Path]] = defaultdict(list)

    for path in sorted(adr_dir.glob("*.md")):
        if path.name in _SKIP:
            continue
        match = _NUMBERED_RE.match(path.name)
        if match:
            by_number[match.group(1)].append(path)

    total_checked = sum(len(v) for v in by_number.values())
    conflicts = {num: paths for num, paths in by_number.items() if len(paths) > 1}

    for num, paths in sorted(conflicts.items()):
        print(f"\nCONFLICT: {num} used by both:")
        for p in sorted(paths):
            print(f"  {p}")

    print()
    if conflicts:
        print(
            f"{len(conflicts)} conflict{'s' if len(conflicts) != 1 else ''} found."
            " Renumber one of the conflicting ADRs."
        )
    print(f"{total_checked} ADRs checked, {len(conflicts)} conflicts found.")

    return len(conflicts)


def main() -> None:
    adr_dir = _repo_root() / "docs" / "adr"
    conflicts = check_adr_numbering(adr_dir)
    sys.exit(1 if conflicts else 0)


if __name__ == "__main__":
    main()
