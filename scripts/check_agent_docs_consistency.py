#!/usr/bin/env python3
"""Check that public contributor guidance matches the enforced CI surface."""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
CLAUDE = ROOT / "CLAUDE.md"
CONTRIBUTING = ROOT / "CONTRIBUTING.md"
PYPROJECT = ROOT / "pyproject.toml"
CHANGELOG = ROOT / "CHANGELOG.md"
CI_STANDARD = ROOT / ".github" / "workflows" / "ci-standard.yml"
_COVERAGE_LITERAL = re.compile(r"(\d+)%")
_COVERAGE_GATE = re.compile(r"--cov-fail-under=(\d+)")


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


def _load_coverage_threshold(pyproject_text: str) -> int:
    return int(
        tomllib.loads(pyproject_text)["tool"]["coverage"]["report"]["fail_under"]
    )


def _iter_duplicate_paragraphs(text: str) -> list[str]:
    paragraphs = [
        " ".join(paragraph.split())
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]
    return [paragraph for paragraph, count in Counter(paragraphs).items() if count > 1]


def _iter_repo_relative_paths(text: str) -> list[str]:
    candidates = re.findall(r"`([^`\n]+)`", text)
    repo_roots = {
        ".gaai",
        ".github",
        "docs",
        "installer",
        "rust",
        "rust_core",
        "scripts",
        "shared",
        "src",
        "tests",
        "ui",
        "vendor",
    }
    exact_files = {
        "AGENTS.md",
        "CHANGELOG.md",
        "CLAUDE.md",
        "CONTRIBUTING.md",
        "Cargo.toml",
        "README.md",
        "SPEC.md",
        "pyproject.toml",
    }
    paths: list[str] = []
    for candidate in candidates:
        normalized = candidate.strip().lstrip("@")
        if (
            not normalized
            or " " in normalized
            or normalized.startswith(("http://", "https://", "/", "D-sorganization/"))
        ):
            continue
        first_segment = normalized.split("/", 1)[0]
        if (
            first_segment in repo_roots or normalized in exact_files
        ) and normalized not in paths:
            paths.append(normalized)
    return paths


def _assert_coverage_alignment(
    claude: str, ci_standard: str, coverage_threshold: int, errors: list[str]
) -> None:
    for paragraph in re.split(r"\n\s*\n", claude):
        if "coverage" not in paragraph.lower():
            continue
        for match in _COVERAGE_LITERAL.findall(paragraph):
            value = int(match)
            if value != coverage_threshold:
                errors.append(
                    "CLAUDE.md cites "
                    f"{value}% coverage but pyproject.toml [tool.coverage.report] "
                    f"fail_under is {coverage_threshold}%."
                )
    gate_match = _COVERAGE_GATE.search(ci_standard)
    if gate_match is None:
        errors.append("CI workflow must define a --cov-fail-under gate.")
        return
    gate_value = int(gate_match.group(1))
    if gate_value != coverage_threshold:
        errors.append(
            "CI workflow coverage floor "
            f"{gate_value}% does not match pyproject.toml fail_under {coverage_threshold}%."
        )


_GLOB_METACHARACTERS = frozenset("*{}?[]")


def _is_glob_pattern(path_text: str) -> bool:
    """Return True if the reference is a glob/brace pattern, not a literal path.

    Documentation often cites patterns such as ``scripts/**`` or
    ``src/shared/python/codemap/{cli,watcher,mcp_server}.py``. These describe a
    family of files rather than a single literal path, so existence checks do
    not apply.
    """
    return any(char in _GLOB_METACHARACTERS for char in path_text)


def _assert_path_references_exist(claude: str, errors: list[str]) -> None:
    for path_text in _iter_repo_relative_paths(claude):
        if _is_glob_pattern(path_text):
            continue
        if not (ROOT / path_text.rstrip("/")).exists():
            errors.append(f"CLAUDE.md references a missing path: {path_text}")


def _assert_no_duplicate_paragraphs(claude: str, errors: list[str]) -> None:
    duplicates = _iter_duplicate_paragraphs(claude)
    if duplicates:
        errors.append("CLAUDE.md contains duplicate paragraphs.")


def main() -> int:
    errors: list[str] = []

    readme = _read(README)
    claude = _read(CLAUDE)
    contributing = _read(CONTRIBUTING)
    pyproject = _read(PYPROJECT)
    changelog = _read(CHANGELOG)
    ci_standard = _read(CI_STANDARD)
    coverage_threshold = _load_coverage_threshold(pyproject)

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
    _assert_not_contains(
        claude,
        "NOT Black",
        "CLAUDE.md should state Ruff directly instead of referencing Black.",
        errors,
    )
    _assert_coverage_alignment(claude, ci_standard, coverage_threshold, errors)
    _assert_path_references_exist(claude, errors)
    _assert_no_duplicate_paragraphs(claude, errors)

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

    if errors:
        print("FAIL: agent docs consistency check failed")
        for error in errors:
            print(f"- {error}")
        return 1

    print("OK: README, CLAUDE, CONTRIBUTING, CHANGELOG, pyproject, and CI are aligned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
