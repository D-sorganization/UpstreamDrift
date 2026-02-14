#!/usr/bin/env python3
"""Ratcheting module-size budget for Python source files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_MAX_LINES = 1500
DEFAULT_INCLUDE = ("src",)
DEFAULT_BASELINE = Path("scripts/config/module_size_budget_baseline.json")
DEFAULT_EXCLUDE_PARTS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "archive",
    "legacy",
    "experimental",
    "__pycache__",
    "build",
    "dist",
    ".mypy_cache",
    ".pytest_cache",
}


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for _ in handle)


def should_skip(path: Path, exclude_parts: set[str]) -> bool:
    return any(part in exclude_parts for part in path.parts)


def iter_python_files(include_roots: tuple[str, ...], exclude_parts: set[str]):
    for root in include_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for candidate in root_path.rglob("*.py"):
            if should_skip(candidate, exclude_parts):
                continue
            yield candidate


def load_baseline(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in data.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-lines", type=int, default=DEFAULT_MAX_LINES)
    parser.add_argument("--include", nargs="+", default=list(DEFAULT_INCLUDE))
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    args = parser.parse_args()

    baseline = load_baseline(Path(args.baseline))
    offenders: list[tuple[Path, int, str]] = []

    for py_file in iter_python_files(tuple(args.include), DEFAULT_EXCLUDE_PARTS):
        rel = str(py_file).replace("\\", "/")
        lines = count_lines(py_file)
        if lines <= args.max_lines:
            continue

        allowed = baseline.get(rel)
        if allowed is None:
            offenders.append((py_file, lines, "new oversized file (not in baseline)"))
            continue
        if lines > allowed:
            offenders.append((py_file, lines, f"grew beyond baseline ({allowed})"))

    if not offenders:
        sys.stdout.write(
            f"module-size budget passed (max {args.max_lines}, baseline {args.baseline})\n"
        )
        return 0

    sys.stderr.write(
        f"module-size budget failed (max {args.max_lines}, baseline {args.baseline})\n"
    )
    for file_path, line_count, reason in sorted(offenders):
        sys.stderr.write(f"  {file_path}: {line_count} lines [{reason}]\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
