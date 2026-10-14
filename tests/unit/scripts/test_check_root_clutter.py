"""Unit tests for root clutter gate script (scripts/check_root_clutter.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_root_clutter import (
    ALLOWLIST,
    DIRECTORY_ALLOWLIST,
    check_root_clutter,
    main,
)

pytestmark = pytest.mark.unit


def test_allowed_files_and_directories_pass(tmp_path: Path) -> None:
    """Ensure allowed files, allowed directories, and hidden dot-entries pass."""
    # Create a subset of allowed files
    (tmp_path / "README.md").write_text("# Test", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("# Agents", encoding="utf-8")

    # Create allowed directories
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "docs").mkdir()

    # Create hidden files and directories (agent governance dotfiles)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".ai").mkdir()
    (tmp_path / ".agent").mkdir()
    (tmp_path / ".gitignore").write_text("*.pyc", encoding="utf-8")

    assert check_root_clutter(tmp_path) == []
    assert main(tmp_path) == 0


def test_unauthorized_file_fails(tmp_path: Path) -> None:
    """Ensure unexpected top-level files are flagged."""
    (tmp_path / "README.md").write_text("# Test", encoding="utf-8")
    (tmp_path / "rogue_dump.csv").write_text("a,b,c", encoding="utf-8")

    bad = check_root_clutter(tmp_path)
    assert bad == ["rogue_dump.csv"]
    assert main(tmp_path) == 1


def test_unauthorized_directory_fails(tmp_path: Path) -> None:
    """Ensure unexpected directories (e.g. worktree leaks or stray dirs) are flagged."""
    (tmp_path / "src").mkdir()
    (tmp_path / "UsersdieteRepositorieswt-11c").mkdir()
    (tmp_path / "rogue_dir").mkdir()

    bad = check_root_clutter(tmp_path)
    assert bad == ["UsersdieteRepositorieswt-11c/", "rogue_dir/"]
    assert main(tmp_path) == 1


def test_mixed_unauthorized_entries(tmp_path: Path) -> None:
    """Ensure both unauthorized files and directories are detected simultaneously."""
    (tmp_path / "README.md").write_text("# Test", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "stray_file.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "accidental_dir").mkdir()

    bad = check_root_clutter(tmp_path)
    assert bad == ["accidental_dir/", "stray_file.txt"]
    assert main(tmp_path) == 1


def test_allowlists_are_non_empty_and_disjoint() -> None:
    """Verify integrity of constant allowlists."""
    assert len(ALLOWLIST) > 0
    assert len(DIRECTORY_ALLOWLIST) > 0
    assert ALLOWLIST.isdisjoint(DIRECTORY_ALLOWLIST)


def test_actual_repo_root_passes() -> None:
    """Verify that the actual repository root has no root clutter."""
    assert check_root_clutter() == []
    assert main() == 0
