#!/usr/bin/env python3
"""Filter explicit mypy file arguments through ``[tool.mypy].exclude``."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]


def _load_exclude_patterns(pyproject: Path) -> list[re.Pattern[str]]:
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    raw_excludes = data.get("tool", {}).get("mypy", {}).get("exclude", [])
    if not isinstance(raw_excludes, list):
        raise ValueError("[tool.mypy].exclude must be a list")
    return [re.compile(str(pattern)) for pattern in raw_excludes]


def _is_excluded(path: str, patterns: list[re.Pattern[str]]) -> bool:
    normalized = path.replace("\\", "/")
    return any(pattern.search(normalized) for pattern in patterns)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", default="pyproject.toml")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args(argv)

    patterns = _load_exclude_patterns(Path(args.pyproject))
    for path in args.paths:
        if not _is_excluded(path, patterns):
            sys.stdout.write(path)
            sys.stdout.write("\0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
