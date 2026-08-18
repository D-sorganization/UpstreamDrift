"""Repo-hygiene guard for the deleted legacy Motion Capture Plotter monoliths.

Issue #7061: the dead legacy monoliths
``.../golf_gui/Motion Capture Plotter/starting_pose_matcher.py`` (2671 LOC) and
``.../Motion Capture Plotter/Motion_Capture_Plotter.py`` (1402 LOC) were never
imported anywhere in ``src/`` and were superseded by the tested
``src/tools/starting_pose_matcher/`` package. They existed only to be
grandfathered in the size budgets.

These invariants prevent the dead files (or their budget exceptions) from
silently coming back:

1. Neither monolith file exists on disk.
2. No ``src/`` file references the legacy monolith path (no import resolves it).
3. The size-budget configs no longer carry an exception for either file.
"""

from __future__ import annotations

import json
import pathlib

import pytest

# Without a suite marker, unit-test-gate's `-m "unit and ..."` selector
# deselects this file entirely and the guard never runs in CI (#7158).
pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

REPO_ROOT = pathlib.Path(__file__).parents[3]

_LEGACY_DIR = (
    "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/"
    "golf_gui/Motion Capture Plotter"
)
_DELETED_FILES = (
    f"{_LEGACY_DIR}/starting_pose_matcher.py",
    f"{_LEGACY_DIR}/Motion_Capture_Plotter.py",
)
_DELETED_BASENAMES = (
    "Motion Capture Plotter/starting_pose_matcher.py",
    "Motion Capture Plotter/Motion_Capture_Plotter.py",
)

_BUDGET_CONFIGS = (
    "scripts/config/file_size_budget.json",
    "scripts/config/module_size_budget_baseline.json",
)


def test_legacy_monolith_files_deleted() -> None:
    """Pre: neither legacy monolith remains on disk."""
    present = [rel for rel in _DELETED_FILES if (REPO_ROOT / rel).is_file()]
    assert not present, f"Legacy monoliths still present: {present}"


def test_no_src_reference_resolves_legacy_path() -> None:
    """No ``src/`` file imports or references the deleted monolith path.

    Scans every ``src/`` Python file for a textual reference to the legacy
    ``Motion Capture Plotter`` monolith paths; an import that resolved the old
    path would necessarily contain such a reference.
    """
    offenders: list[str] = []
    for py_file in (REPO_ROOT / "src").rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(needle in content for needle in _DELETED_BASENAMES):
            offenders.append(str(py_file.relative_to(REPO_ROOT)).replace("\\", "/"))
    assert not offenders, (
        f"src/ files still reference the deleted legacy monolith path: {offenders}"
    )


def test_budgets_no_longer_list_legacy_monoliths() -> None:
    """Post: no size-budget config carries an exception for either monolith."""
    listed: list[str] = []
    for config_rel in _BUDGET_CONFIGS:
        config = json.loads((REPO_ROOT / config_rel).read_text(encoding="utf-8"))
        exception_paths = {
            str(exc.get("path", "")) for exc in config.get("exceptions", [])
        }
        for deleted in _DELETED_FILES:
            if deleted in exception_paths:
                listed.append(f"{config_rel}: {deleted}")
    assert not listed, f"Budget configs still list deleted monoliths: {listed}"
