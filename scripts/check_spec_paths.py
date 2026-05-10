#!/usr/bin/env python3
"""Fail if SPEC.md cites paths that don't exist on disk."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "SPEC.md"

# Match `path/like/this.py` or `path/like/this/` inside backticks.
PATH_PATTERN = re.compile(r"`([a-zA-Z_][a-zA-Z0-9_./-]+(?:\.py|/))`")

# These prefixes are the project's package roots. Anything matched against
# them must exist; everything else is treated as a non-path identifier.
REPO_ROOTS = {
    "src/",
    "tests/",
    "scripts/",
    "ui/",
    "rust_core/",
    "shared/",
    "vendor/",
    "docs/",
    "installer/",
    ".github/",
}


def main() -> int:
    text = SPEC.read_text(encoding='utf-8')
    bad: list[str] = []
    for match in PATH_PATTERN.finditer(text):
        path = match.group(1)
        if not any(path.startswith(prefix) for prefix in REPO_ROOTS):
            continue
        target = ROOT / path.rstrip("/")
        if not target.exists():
            line = text[: match.start()].count("\n") + 1
            bad.append(f"  SPEC.md:{line}: missing path {path!r}")
    if bad:
        print("FAIL: SPEC.md references paths that do not exist:")
        print("\n".join(bad))
        print("\nUpdate SPEC.md or add the file/directory it claims exists.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
