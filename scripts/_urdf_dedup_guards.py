#!/usr/bin/env python3
"""One-shot: deduplicate the AI-slop `if not (X is not None)` guards.

Finds consecutive identical blocks of:

    if not (X is not None):
        raise ValueError("X must be provided")
    if not (X is not None):
        raise ValueError("X must be provided")

and reduces them to a single occurrence (the duplicate is the bug — we
keep the first guard so behavior is identical).

Where the parameter is non-Optional in the function signature, the
remaining guard is also redundant, but removing those requires
type-aware analysis. We're deliberately conservative here and just
remove the literal duplicates. The remaining single guards can be
swept in a follow-up.

See issue #4532.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path("src/shared/python")
TARGETS = [
    ROOT / "humanoid_character_builder",
    ROOT / "model_generation",
]

# Match two consecutive identical guard blocks.
# Each guard is exactly:
#     <indent>if not (<NAME> is not None):
#     <indent>    raise ValueError("<MSG>")
PATTERN = re.compile(
    r"^([ \t]+)if not \((\w+) is not None\):\n"
    r"\1    raise ValueError\((\"[^\"]+\")\)\n"
    r"\1if not \(\2 is not None\):\n"
    r"\1    raise ValueError\(\3\)\n",
    re.MULTILINE,
)


def dedup_file(path: Path) -> int:
    src = path.read_text()
    new, n = PATTERN.subn(
        lambda m: (
            f"{m.group(1)}if not ({m.group(2)} is not None):\n"
            f"{m.group(1)}    raise ValueError({m.group(3)})\n"
        ),
        src,
    )
    if n:
        path.write_text(new)
    return n


def main() -> None:
    total = 0
    files_changed = 0
    for target in TARGETS:
        for p in target.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            n = dedup_file(p)
            if n:
                files_changed += 1
                total += n
                print(f"  {p}: -{n}")
    print(f"\nDeduplicated {total} guard blocks across {files_changed} files.")


if __name__ == "__main__":
    main()
