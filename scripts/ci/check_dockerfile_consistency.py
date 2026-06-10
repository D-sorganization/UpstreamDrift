#!/usr/bin/env python3
"""Assert that all ``pip==`` pins across Dockerfile* agree (issue #7161).

Two Dockerfile variants pinning different pip versions (26.1 vs 25.3) produce
divergent wheel selection between the canonical and modular images. This check
greps every ``Dockerfile*`` for ``pip==<version>`` pins and fails if more than
one distinct version is found.

Exit code 0 when consistent (or no pins found), 1 otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_PIP_PIN_RE = re.compile(r"\bpip==([0-9][0-9A-Za-z.\-]*)")


def find_pip_pins(repo_root: Path) -> dict[str, list[str]]:
    """Map each Dockerfile path to the list of pinned pip versions it declares."""
    pins: dict[str, list[str]] = {}
    for dockerfile in sorted(repo_root.glob("Dockerfile*")):
        if not dockerfile.is_file():
            continue
        versions = _PIP_PIN_RE.findall(dockerfile.read_text(encoding="utf-8"))
        if versions:
            pins[dockerfile.name] = versions
    return pins


def check(repo_root: Path) -> list[str]:
    """Return a list of error strings; empty when all pip pins agree."""
    pins = find_pip_pins(repo_root)
    distinct = {v for versions in pins.values() for v in versions}
    if len(distinct) <= 1:
        return []
    errors = [f"Inconsistent pip pins across Dockerfile* (found {sorted(distinct)}):"]
    for name, versions in pins.items():
        errors.append(f"  {name}: {sorted(set(versions))}")
    errors.append("Align all `pip==` pins on a single version (DRY).")
    return errors


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    errors = check(repo_root)
    if errors:
        print("\n".join(errors))
        return 1
    print("OK: all Dockerfile pip pins are consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
