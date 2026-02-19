#!/usr/bin/env python3
"""Block net-new print() usage in production Python paths."""

from __future__ import annotations

import argparse
import ast
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PRODUCTION_ROOTS = (
    "src/api",
    "src/shared/python",
    "src/robotics",
    "src/launchers",
    "src/tools",
)
EXCLUDED_SEGMENTS = {"examples", "tutorials"}


def _run_git(args: list[str], repo_root: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git command failed")
    return result.stdout


def _in_production_path(path: str, roots: tuple[str, ...]) -> bool:
    return any(path.startswith(f"{root}/") or path == root for root in roots)


def _is_excluded(path: str) -> bool:
    parts = set(Path(path).parts)
    return bool(parts & EXCLUDED_SEGMENTS)


def changed_python_files(
    repo_root: Path, base_ref: str, production_roots: tuple[str, ...]
) -> list[Path]:
    output = _run_git(["diff", "--name-only", f"{base_ref}...HEAD", "--"], repo_root)
    paths: list[Path] = []
    for raw in output.splitlines():
        raw = raw.strip()
        if not raw.endswith(".py"):
            continue
        if not _in_production_path(raw, production_roots):
            continue
        if _is_excluded(raw):
            continue
        paths.append(repo_root / raw)
    return paths


def find_print_calls(file_path: Path) -> list[int]:
    """Return line numbers containing print() calls."""
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    except (UnicodeDecodeError, SyntaxError):
        return []

    lines: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ):
            lines.append(node.lineno)
    return sorted(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Block net-new print() in changed production files."
    )
    parser.add_argument(
        "--base-ref",
        default="origin/main",
        help="Git base ref used for diff (default: origin/main).",
    )
    parser.add_argument(
        "--production-roots",
        nargs="+",
        default=list(DEFAULT_PRODUCTION_ROOTS),
        help="Roots treated as production code.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    production_roots = tuple(args.production_roots)

    try:
        files = changed_python_files(repo_root, args.base_ref, production_roots)
    except RuntimeError as exc:
        # Fallback keeps local/dev usage working when origin/main is unavailable.
        if args.base_ref != "HEAD~1":
            files = changed_python_files(repo_root, "HEAD~1", production_roots)
        else:
            logger.error("FAIL: Unable to compute changed files: %s", exc)
            return 1

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    violations: list[str] = []
    for path in files:
        for line in find_print_calls(path):
            violations.append(f"{path.relative_to(repo_root)}:{line}")

    if violations:
        logger.error("FAIL: print() usage detected in changed production files:")
        for violation in violations:
            logger.error("  %s", violation)
        logger.error("\nUse structured logging via get_logger() instead of print().")
        return 1

    logger.info("OK: No print() calls found in changed production files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
