"""Guard tests for the quality budgets index (issue #7133).

docs/development/quality_budgets.md is the single discoverable index of the
repo's enforced quality budgets. These tests keep it honest: every config file
and enforcer script it references must actually exist, so the index cannot
silently drift from the gates it documents.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INDEX = _REPO_ROOT / "docs" / "development" / "quality_budgets.md"

# Referenced paths that live in other in-flight PRs; tolerate their absence so
# this index PR is not coupled to merge order.
_TOLERATED_ABSENT = {
    "scripts/gen_monolith_register.py",
    "docs/development/monolith_refactor_register.md",
}

_PATH_RE = re.compile(r"`(scripts/[^`]+\.(?:json|py)|docs/[^`]+\.md)`")


def _referenced_paths() -> set[str]:
    text = _INDEX.read_text(encoding="utf-8")
    return set(_PATH_RE.findall(text))


def test_index_exists() -> None:
    assert _INDEX.exists(), (
        "docs/development/quality_budgets.md must exist (issue #7133)"
    )


def test_index_references_core_budgets() -> None:
    text = _INDEX.read_text(encoding="utf-8")
    for expected in (
        "file_size_budget.json",
        "module_size_budget_baseline.json",
        "error_handling_baseline.json",
        "mypy_exclusion_budget.json",
    ):
        assert expected in text, expected


def test_all_referenced_config_files_exist() -> None:
    missing = []
    for rel in sorted(_referenced_paths()):
        if rel in _TOLERATED_ABSENT:
            continue
        if not (_REPO_ROOT / rel).exists():
            missing.append(rel)
    assert not missing, f"quality_budgets.md references missing paths: {missing}"


def test_referenced_config_jsons_are_under_scripts_config() -> None:
    """Budget configs must live in the documented scripts/config/ location."""
    json_refs = {p for p in _referenced_paths() if p.endswith(".json")}
    assert json_refs, "index should reference at least one budget config"
    for rel in json_refs:
        assert rel.startswith("scripts/config/"), rel
