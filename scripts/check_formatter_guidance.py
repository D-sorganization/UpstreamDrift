#!/usr/bin/env python3
"""Guard to keep formatter guidance aligned to Ruff."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MAKEFILE = ROOT / "Makefile"
PRE_COMMIT = ROOT / ".pre-commit-config.yaml"
MANUAL = ROOT / "docs" / "user_guide" / "upstream_drift_user_manual.md"
CONTRIBUTING = ROOT / "docs" / "development" / "contributing.md"


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _fail(message: str, details: list[str], errors: list[str]) -> None:
    text = "\n".join(f"    - {item}" for item in details)
    errors.append(f"{message}:\n{text}")


def check_makefile(errors: list[str]) -> None:
    lines = _read_lines(MAKEFILE)
    content = "\n".join(lines)

    forbidden = [
        (
            r'@echo "  make format    - Format code \(black, ruff\)"',
            "Use Ruff-only format guidance",
        ),
        (
            r"@echo \"Running black\\.\\.\\.\"",
            "Remove black formatter step from format target",
        ),
        (r"^black \\.$", "Remove black formatter command from Makefile format target"),
        (
            r"including dev tools: ruff, black, mypy, pytest",
            "Remove black from install dev-tool note",
        ),
    ]

    hits = [pattern for pattern, _ in forbidden if re.search(pattern, content)]
    if hits:
        descriptions = [
            desc for pattern, desc in forbidden if re.search(pattern, content)
        ]
        _fail(
            "Makefile formatter guidance is no longer Ruff-only", descriptions, errors
        )

    if "black" in content:
        _fail(
            "Makefile contains unexpected formatter references to black",
            ["black"],
            errors,
        )


def check_precommit(errors: list[str]) -> None:
    lines = _read_lines(PRE_COMMIT)
    black_hooks = [
        i + 1 for i, line in enumerate(lines) if re.match(r"^\s*- id:\s*black$", line)
    ]
    if black_hooks:
        _fail(
            ".pre-commit-config.yaml still defines a black hook",
            [f"line {line}: remove black hook ID" for line in black_hooks],
            errors,
        )

    if not any("ruff-format" in line for line in lines):
        _fail(
            ".pre-commit-config.yaml has no ruff-format hook",
            ["Add or restore a ruff-format hook under .pre-commit-config.yaml"],
            errors,
        )


def check_docs(errors: list[str]) -> None:
    manual_lines = "\n".join(_read_lines(MANUAL))
    contributing_lines = "\n".join(_read_lines(CONTRIBUTING))

    if "Black, Ruff" in manual_lines:
        _fail(
            "Manual chapter heading still calls out Black alongside Ruff",
            ["Update 24.2 heading to Ruff-only language"],
            errors,
        )

    if re.search(r"\*\*black\*\*:\\s*Code formatter", manual_lines):
        _fail(
            "Manual pre-commit section still references black as formatter",
            ["Replace black entry with ruff formatter text"],
            errors,
        )

    if re.search(r"Run black, ruff format", manual_lines):
        _fail(
            "Manual Makefile target table still documents black in format command",
            ["Update format command to Ruff-only guidance"],
            errors,
        )

    if "ruff and black" in contributing_lines:
        _fail(
            "Contributing guide still claims Python uses ruff and black",
            ["Update style guidance to Ruff-only"],
            errors,
        )


def main() -> int:
    errors: list[str] = []
    check_makefile(errors)
    check_precommit(errors)
    check_docs(errors)

    if errors:
        print("FAIL: formatter-guidance check failed")
        for error in errors:
            print(f"- {error}")
        return 1
    print(
        "OK: formatter guidance is Ruff-aligned in Makefile, docs, and pre-commit config"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
