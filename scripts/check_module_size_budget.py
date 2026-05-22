#!/usr/bin/env python3
"""Ratcheting module-size budget for Python source files."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterator
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for _ in handle)


def should_skip(path: Path, exclude_parts: set[str]) -> bool:
    return any(part in exclude_parts for part in path.parts)


def iter_python_files(
    include_roots: tuple[str, ...], exclude_parts: set[str], repo_root: Path
) -> Iterator[Path]:
    for root in include_roots:
        root_path = repo_root / root
        if not root_path.exists():
            continue
        for candidate in root_path.rglob("*.py"):
            if should_skip(candidate, exclude_parts):
                continue
            yield candidate


def _get_owner(path: str, repo_root: Path) -> str:
    codeowners_path = repo_root / ".github" / "CODEOWNERS"
    if not codeowners_path.exists():
        return "Unknown"

    owner = "Unknown"
    for line in codeowners_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            pattern = parts[0]
            if path.startswith(pattern.lstrip("/")) or f"/{path}".startswith(pattern):
                owner = " ".join(parts[1:])
    return owner


def _exception_is_active(exc: dict) -> bool:
    expires_on = exc.get("expires_on")
    if not expires_on:
        return True
    return date.today() <= date.fromisoformat(expires_on)


def _collect_active_exceptions(config: dict) -> tuple[dict[str, dict], list[str]]:
    active_exceptions: dict[str, dict] = {}
    invalid_exceptions: list[str] = []

    for exc in config.get("exceptions", []):
        path = str(exc.get("path", "")).strip()
        owner = str(exc.get("owner", "")).strip()
        reason = str(exc.get("reason", "")).strip()
        if not path or not owner or not reason:
            invalid_exceptions.append(f"Invalid exception entry: {exc}")
            continue
        if (
            "issue" not in reason.lower()
            and "#" not in reason
            and "decomposition" not in reason.lower()
            and "legacy" not in reason.lower()
        ):
            invalid_exceptions.append(
                f"Exception missing linked issue in reason: {path}"
            )
            continue
        try:
            if _exception_is_active(exc):
                active_exceptions[path] = exc
            else:
                invalid_exceptions.append(
                    f"Expired exception: {path} (owner={owner}, expires_on={exc.get('expires_on')})"
                )
        except ValueError:
            invalid_exceptions.append(
                f"Invalid expires_on date in exception: {path} ({exc.get('expires_on')})"
            )

    return active_exceptions, invalid_exceptions


def load_baseline(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-lines", type=int, default=DEFAULT_MAX_LINES)
    parser.add_argument("--include", nargs="+", default=list(DEFAULT_INCLUDE))
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    args = parser.parse_args()

    repo_root = _repo_root()
    baseline_path = repo_root / args.baseline
    config = load_baseline(baseline_path)
    budget = int(config.get("max_lines", args.max_lines))

    active_exceptions, invalid_exceptions = _collect_active_exceptions(config)

    # 10 is the current maximum exceptions for module sizes.
    if len(config.get("exceptions", [])) > 10:
        invalid_exceptions.append(
            f"Too many exceptions: {len(config.get('exceptions', []))} (max 10)"
        )

    violations = list(invalid_exceptions)
    watchlist: list[str] = []

    for py_file in iter_python_files(
        tuple(args.include), DEFAULT_EXCLUDE_PARTS, repo_root
    ):
        rel = str(py_file.relative_to(repo_root)).replace("\\", "/")
        if rel in active_exceptions:
            continue

        count = count_lines(py_file)
        if count > budget:
            violations.append(f"{rel}: {count} lines (budget={budget})")
        elif budget * 0.9 <= count <= budget:
            owner = _get_owner(rel, repo_root)
            watchlist.append(f"{rel}: {count} lines (owner: {owner})")

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if watchlist:
        logger.info("WATCHLIST (%d-%d lines):", int(budget * 0.9), budget)
        for item in watchlist:
            logger.info("  %s", item)
        logger.info("")

    if violations:
        logger.error("FAIL: module size budget violations detected:\n")
        for violation in violations:
            logger.error("  %s", violation)
        return 1

    logger.info("OK: All modules are within line-count budget.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
