#!/usr/bin/env python3
"""Enforce line-count budgets for tracked Python files with owned exceptions."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("scripts/config/file_size_budget.json")


def _repo_root() -> Path:
    """Return the repository root for this script."""
    return Path(__file__).resolve().parents[2]


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


def _load_config(repo_root: Path, config_path: Path) -> dict:
    with (repo_root / config_path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _changed_python_files(repo_root: Path, base_ref: str) -> list[Path]:
    output = _run_git(["diff", "--name-only", f"{base_ref}...HEAD", "--"], repo_root)
    return [
        repo_root / path
        for path in output.splitlines()
        if path.endswith(".py") and (repo_root / path).exists()
    ]


def _tracked_python_files(repo_root: Path) -> list[Path]:
    output = _run_git(["ls-files", "--", "*.py"], repo_root)
    return [
        repo_root / path
        for path in output.splitlines()
        if path and (repo_root / path).exists()
    ]


def _fallback_python_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in repo_root.rglob("*.py"):
        if any(part.startswith(".") for part in path.relative_to(repo_root).parts):
            continue
        files.append(path)
    return files


def _exception_is_active(exc: dict) -> bool:
    expires_on = exc.get("expires_on")
    if not expires_on:
        return True
    return date.today() <= date.fromisoformat(expires_on)


def _line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


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


def main() -> int:  # noqa: C901
    parser = argparse.ArgumentParser(
        description="Enforce changed-file line-count budget."
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to JSON config relative to repository root.",
    )
    parser.add_argument(
        "--base-ref",
        default="origin/main",
        help="Legacy option retained for compatibility; tracked-file scanning ignores it.",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    config = _load_config(repo_root, args.config_path)
    budget = int(config.get("max_lines", 1200))
    max_exceptions = int(config.get("max_exceptions", 8))

    active_exceptions, invalid_exceptions = _collect_active_exceptions(config)

    try:
        python_files = _tracked_python_files(repo_root)
    except RuntimeError:
        python_files = _fallback_python_files(repo_root)

    if len(config.get("exceptions", [])) > max_exceptions:
        invalid_exceptions.append(
            f"Too many exceptions: {len(config.get('exceptions', []))} (max {max_exceptions})"
        )

    violations = list(invalid_exceptions)
    watchlist: list[str] = []
    for file_path in python_files:
        rel = str(file_path.relative_to(repo_root)).replace("\\", "/")
        if rel.startswith("tests/"):
            continue
        if rel in active_exceptions:
            continue
        count = _line_count(file_path)
        if count > budget:
            violations.append(f"{rel}: {count} lines (budget={budget})")
        elif budget * 0.9 <= count <= budget:
            owner = _get_owner(rel, repo_root)
            watchlist.append(f"{rel}: {count} lines (owner: {owner})")

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if watchlist:
        logger.info("WATCHLIST (1080-1200 lines):")
        for item in watchlist:
            logger.info("  %s", item)
        logger.info("")

    if violations:
        logger.error("FAIL: file size budget violations detected:\n")
        for violation in violations:
            logger.error("  %s", violation)
        logger.error(
            "\nSplit orchestration/domain/IO concerns or add owned, expiring exception."
        )
        return 1

    logger.info("OK: Tracked files are within line-count budget.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
