#!/usr/bin/env python3
"""Enforce line-count budgets for changed Python files with owned exceptions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("scripts/config/file_size_budget.json")


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


def _exception_is_active(exc: dict) -> bool:
    expires_on = exc.get("expires_on")
    if not expires_on:
        return True
    return date.today() <= date.fromisoformat(expires_on)


def _line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def main() -> int:
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
        help="Git base ref used for changed-file detection.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    config = _load_config(repo_root, args.config_path)
    budget = int(config.get("max_lines", 1200))

    active_exceptions: dict[str, dict] = {}
    invalid_exceptions: list[str] = []
    for exc in config.get("exceptions", []):
        path = str(exc.get("path", "")).strip()
        owner = str(exc.get("owner", "")).strip()
        reason = str(exc.get("reason", "")).strip()
        if not path or not owner or not reason:
            invalid_exceptions.append(f"Invalid exception entry: {exc}")
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

    try:
        changed_files = _changed_python_files(repo_root, args.base_ref)
    except RuntimeError:
        changed_files = _changed_python_files(repo_root, "HEAD~1")

    violations = list(invalid_exceptions)
    for file_path in changed_files:
        rel = str(file_path.relative_to(repo_root)).replace("\\", "/")
        if rel.startswith("tests/"):
            continue
        if rel in active_exceptions:
            continue
        count = _line_count(file_path)
        if count > budget:
            violations.append(f"{rel}: {count} lines (budget={budget})")

    if violations:
        print("FAIL: file size budget violations detected:\n")
        for violation in violations:
            print(f"  {violation}")
        print(
            "\nSplit orchestration/domain/IO concerns or add owned, expiring exception."
        )
        return 1

    print("OK: Changed files are within line-count budget.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
