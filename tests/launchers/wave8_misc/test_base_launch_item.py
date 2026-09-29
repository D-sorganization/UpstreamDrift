"""Coverage for ``LaunchItem`` and small pure helpers in src/launchers/base.py.

The bulk of ``BaseLauncher`` is GUI; this file covers the pure-Python
``LaunchItem`` dataclass-like wrapper and its ``get_full_path`` helper
that resolves paths relative to the repo root.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.launchers import base as launchers_base
from src.launchers.base import REPO_ROOT, LaunchItem


def test_minimum_construction() -> None:
    item = LaunchItem(name="Tool", description="A test tool")
    assert item.name == "Tool"
    assert item.description == "A test tool"
    assert item.path is None
    assert item.item_type == "tool"
    assert item.icon is None
    assert item.action is None


def test_full_construction_keeps_all_fields() -> None:
    action = lambda: None  # noqa: E731
    item = LaunchItem(
        name="ModelX",
        description="desc",
        path="src/foo/bar.py",
        item_type="model",
        icon="icons/foo.png",
        action=action,
    )
    assert item.item_type == "model"
    assert item.icon == "icons/foo.png"
    assert item.action is action


def test_name_must_not_be_none() -> None:
    with pytest.raises(ValueError, match="name must be provided"):
        LaunchItem(name=None, description="x")  # type: ignore[arg-type]


def test_get_full_path_with_path_set() -> None:
    item = LaunchItem(name="x", description="x", path="sub/file.txt")
    result = item.get_full_path()
    assert result == REPO_ROOT / "sub" / "file.txt"
    assert isinstance(result, Path)


def test_get_full_path_returns_none_when_no_path() -> None:
    item = LaunchItem(name="x", description="x")
    assert item.get_full_path() is None


def test_repo_root_is_resolvable() -> None:
    """REPO_ROOT must be an absolute Path pointing to the repo."""
    assert REPO_ROOT.is_absolute()
    # The constant lives 3 levels up from src/launchers/base.py
    assert (REPO_ROOT / "src" / "launchers" / "base.py").is_file()


def test_module_exports_run_launcher() -> None:
    """The convenience runner function is exported for import-side use."""
    assert callable(launchers_base.run_launcher)
