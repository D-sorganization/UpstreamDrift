#!/usr/bin/env python3
"""Fail when tracked in-tree tests are outside pytest testpaths."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10 fallback
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


def find_missing_testpaths(repo_root: Path, testpaths: list[Path]) -> list[str]:
    """Return configured pytest testpaths that do not exist as directories."""
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be a pathlib.Path")
    if not repo_root.is_dir():
        raise ValueError(f"repo_root must be an existing directory: {repo_root}")
    return [path.as_posix() for path in testpaths if not (repo_root / path).is_dir()]


def main() -> int:
    testpaths = _configured_testpaths()
    missing_dirs = find_missing_testpaths(ROOT, testpaths)
    missing = [
        str(path)
        for path in _tracked_in_tree_tests()
        if not _is_covered(path, testpaths)
    ]
    if missing_dirs:
        sys.stderr.write("Configured pytest testpaths do not exist on disk:\n")
        sys.stderr.write("\n".join(missing_dirs) + "\n")
    if missing:
        sys.stderr.write("Tracked in-tree tests are not covered by pytest testpaths:\n")
        sys.stderr.write("\n".join(missing) + "\n")
    if missing_dirs or missing:
        return 1
    sys.stdout.write(
        "All configured testpaths exist and cover tracked in-tree tests.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
