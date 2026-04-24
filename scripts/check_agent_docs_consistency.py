#!/usr/bin/env python3
"""Check that public contributor guidance matches the enforced CI surface."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
CLAUDE = ROOT / "CLAUDE.md"
CONTRIBUTING = ROOT / "CONTRIBUTING.md"
PYPROJECT = ROOT / "pyproject.toml"
CHANGELOG = ROOT / "CHANGELOG.md"
CI_STANDARD = ROOT / ".github" / "workflows" / "ci-standard.yml"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _assert_contains(text: str, needle: str, message: str, errors: list[str]) -> None:
    if needle not in text:
        errors.append(message)


def _assert_not_contains(
    text: str, needle: str, message: str, errors: list[str]
) -> None:
    if needle in text:
        errors.append(message)


def main() -> int:
    errors: list[str] = []

    readme = _read(README)
    claude = _read(CLAUDE)
    contributing = _read(CONTRIBUTING)
    pyproject = _read(PYPROJECT)
    changelog = _read(CHANGELOG)
    ci_standard = _read(CI_STANDARD)

    _assert_not_contains(
        readme,
        "code%20style-black",
        "README.md still advertises Black instead of Ruff.",
        errors,
    )
    _assert_not_contains(
        readme,
        "Format code with black and ruff",
        "README.md still tells contributors to format with Black and Ruff.",
        errors,
    )

    _assert_contains(
        claude,
        "`CLAUDE.md` is the authoritative contributor and agent policy file.",
        "CLAUDE.md must declare itself the authoritative policy file.",
        errors,
    )
    _assert_contains(
        claude,
        "**30% coverage minimum**",
        "CLAUDE.md must match the CI coverage floor of 30%.",
        errors,
    )
    _assert_not_contains(
        claude,
        "NOT Black",
        "CLAUDE.md should state Ruff directly instead of referencing Black.",
        errors,
    )

    _assert_contains(
        contributing,
        "`CLAUDE.md` is the authoritative source for repository rules and quality gates.",
        "CONTRIBUTING.md must point to CLAUDE.md as the source of truth.",
        errors,
    )
    for forbidden in (
        "Golf Modeling Suite",
        "Black (default settings)",
        "Format with black and ruff",
        "python3 -m black .",
        "black tests/",
        "ruff, black, mypy, pytest",
    ):
        _assert_not_contains(
            contributing,
            forbidden,
            f"CONTRIBUTING.md still contains stale guidance: {forbidden}",
            errors,
        )

    _assert_not_contains(
        pyproject,
        '"black>=26.3.1"',
        "pyproject.toml still lists Black in the dev extra.",
        errors,
    )
    _assert_not_contains(
        pyproject,
        "[tool.black]",
        "pyproject.toml still defines a [tool.black] section.",
        errors,
    )

    _assert_contains(
        changelog,
        "All notable changes to UpstreamDrift will be documented in this file.",
        "CHANGELOG.md still uses the old project name.",
        errors,
    )
    _assert_contains(
        ci_standard,
        "--cov-fail-under=30",
        "CI workflow no longer exposes the expected 30% coverage floor.",
        errors,
    )

    if errors:
        print("FAIL: agent docs consistency check failed")
        for error in errors:
            print(f"- {error}")
        return 1

    print("OK: README, CLAUDE, CONTRIBUTING, CHANGELOG, pyproject, and CI are aligned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
