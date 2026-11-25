"""pyproject.toml must not declare the same runtime dependency twice (UD #15768)."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def _canonical(requirement: str) -> str:
    """Return the PEP 503 normalised project name of a requirement string."""
    match = _NAME_RE.match(requirement.strip())
    assert match, f"unparseable requirement: {requirement!r}"
    return re.sub(r"[-_.]+", "-", match.group(0)).lower()


def _duplicates(requirements: list[str]) -> list[str]:
    counts = Counter(_canonical(req) for req in requirements)
    return sorted(name for name, n in counts.items() if n > 1)


def test_duplicate_detector_flags_repeated_names() -> None:
    assert _duplicates(
        ["certifi>=1", "Flask>=3", "certifi>=1", "my_pkg", "my-pkg"]
    ) == [
        "certifi",
        "my-pkg",
    ]


def test_project_dependencies_have_no_duplicates() -> None:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]
    assert _duplicates(deps) == []


def test_optional_dependency_groups_have_no_duplicates() -> None:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"].get("optional-dependencies", {})
    offenders = {g: d for g, reqs in extras.items() if (d := _duplicates(reqs))}
    assert offenders == {}
