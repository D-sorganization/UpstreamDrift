#!/usr/bin/env python3
"""Fail when tracked in-tree tests are outside pytest testpaths."""

from __future__ import annotations

import subprocess
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"


def _tracked_in_tree_tests() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "src/**/test*.py", "src/**/tests/*.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    paths: list[Path] = []
    for line in result.stdout.splitlines():
        path = Path(line)
        if path.name == "__init__.py":
            continue
        paths.append(path)
    return paths


def _configured_testpaths() -> list[Path]:
    with PYPROJECT.open("rb") as handle:
        data = tomllib.load(handle)
    values = data["tool"]["pytest"]["ini_options"]["testpaths"]
    return [Path(value) for value in values]


def _is_covered(test_file: Path, testpaths: list[Path]) -> bool:
    return any(test_file == path or path in test_file.parents for path in testpaths)


def main() -> int:
    testpaths = _configured_testpaths()
    missing = [
        str(path)
        for path in _tracked_in_tree_tests()
        if not _is_covered(path, testpaths)
    ]
    if missing:
        print("Tracked in-tree tests are not covered by pytest testpaths:")
        print("\n".join(missing))
        return 1
    print("All tracked in-tree tests are covered by pytest testpaths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
