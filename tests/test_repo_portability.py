"""Regression tests for portable maintenance scripts."""

from __future__ import annotations

from pathlib import Path

import patch_analyzers

WORKSTATION_ROOT_LITERAL = "C:/Users/diete/Repositories/UpstreamDrift"


def _maintained_script_paths(repo_root: Path) -> list[Path]:
    """Return maintained script paths that should stay portable."""
    candidates = list(repo_root.glob("*.py"))
    candidates.extend((repo_root / "scripts").rglob("*.py"))
    return [path for path in candidates if path.is_file()]


def test_patch_analyzers_resolves_repo_root_from_script_location() -> None:
    """The maintenance script should discover the repo root relative to itself."""
    expected_root = Path(__file__).resolve().parents[1]
    assert patch_analyzers.resolve_repo_root() == expected_root


def test_maintained_scripts_do_not_contain_workstation_specific_repo_paths() -> None:
    """Maintained scripts should not pin one developer checkout path."""
    repo_root = Path(__file__).resolve().parents[1]
    offending_files: list[str] = []
    for path in _maintained_script_paths(repo_root):
        normalized = path.read_text(encoding="utf-8").replace("\\\\", "/")
        if WORKSTATION_ROOT_LITERAL in normalized:
            offending_files.append(str(path.relative_to(repo_root)))

    assert offending_files == []
