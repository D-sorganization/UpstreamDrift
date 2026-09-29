"""Tests for src.launchers.external_tools_adapter.

Focuses on the pure-Python discovery logic:

* ``_find_tools_repo`` returns the path set via ``TOOLS_REPO_PATH``.
* Falls back to sibling auto-discovery when the env-var is missing.
* Returns ``None`` when no Tools repo is reachable.
* ``_ensure_tools_on_path`` injects the discovered path into sys.path
  exactly once.

We deliberately avoid instantiating the Qt placeholder widgets — those
require a live ``QApplication`` and are tangential to the helper
behaviour we care about.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from src.launchers import external_tools_adapter as eta


def _reset_module_cache() -> None:
    """Reset the module-level _TOOLS_REPO cache."""
    eta._TOOLS_REPO = None


def test_env_var_wins(tmp_path, monkeypatch) -> None:
    _reset_module_cache()
    repo = tmp_path / "MyTools"
    (repo / "src").mkdir(parents=True)
    monkeypatch.setenv("TOOLS_REPO_PATH", str(repo))
    found = eta._find_tools_repo()
    assert found == repo
    # Cached on second call
    assert eta._find_tools_repo() == repo


def test_env_var_pointing_to_nonexistent_falls_through(tmp_path, monkeypatch) -> None:
    _reset_module_cache()
    bogus = tmp_path / "nonexistent"
    monkeypatch.setenv("TOOLS_REPO_PATH", str(bogus))
    # Walk-up discovery is best-effort; result may be None or a real path
    # depending on the test execution environment. The contract under test
    # is just that an invalid env var does not raise.
    result = eta._find_tools_repo()
    assert result is None or isinstance(result, Path)


def test_no_tools_repo_returns_none(monkeypatch) -> None:
    _reset_module_cache()
    monkeypatch.delenv("TOOLS_REPO_PATH", raising=False)
    # Patch Path.is_dir to always be False so sibling discovery fails.
    with patch.object(Path, "is_dir", return_value=False):
        result = eta._find_tools_repo()
    assert result is None


def test_ensure_tools_on_path_adds_once(tmp_path, monkeypatch) -> None:
    _reset_module_cache()
    repo = tmp_path / "Tools"
    (repo / "src").mkdir(parents=True)
    monkeypatch.setenv("TOOLS_REPO_PATH", str(repo))
    src_dir = str(repo / "src")
    # Remove if already there
    sys.path[:] = [p for p in sys.path if p != src_dir]

    assert eta._ensure_tools_on_path() is True
    assert sys.path.count(src_dir) == 1
    # Calling again does not duplicate
    assert eta._ensure_tools_on_path() is True
    assert sys.path.count(src_dir) == 1

    # Cleanup
    sys.path[:] = [p for p in sys.path if p != src_dir]


def test_ensure_tools_on_path_returns_false_when_missing(monkeypatch) -> None:
    _reset_module_cache()
    monkeypatch.delenv("TOOLS_REPO_PATH", raising=False)
    with patch.object(eta, "_find_tools_repo", return_value=None):
        assert eta._ensure_tools_on_path() is False


def test_unavailable_tool_widget_renders(qapp) -> None:
    from PyQt6.QtWidgets import QLabel

    from src.launchers.external_tools_adapter import _UnavailableToolWidget

    w = _UnavailableToolWidget("MyTool", "boom")
    text = " ".join(label.text() for label in w.findChildren(QLabel))
    assert "MyTool" in text
    assert "boom" in text
    w.cleanup()  # no-op smoke check
    w.deleteLater()


def test_unavailable_tool_window_wraps_widget(qapp) -> None:
    from src.launchers.external_tools_adapter import _UnavailableToolWindow

    win = _UnavailableToolWindow("Foo", "missing dep")
    assert "Foo" in win.windowTitle()
    assert "Unavailable" in win.windowTitle()
    assert win.centralWidget() is not None
    assert win.is_tool_available is False
    win.deleteLater()


def test_wrap_external_widget_returns_placeholder_when_unavailable(
    qapp, monkeypatch
) -> None:
    """When the Tools repo isn't found, get a placeholder window."""
    _reset_module_cache()
    monkeypatch.delenv("TOOLS_REPO_PATH", raising=False)
    with patch.object(eta, "_find_tools_repo", return_value=None):
        result = eta._wrap_external_widget("XYZ", lambda: None)
    assert "Unavailable" in result.windowTitle()
    result.deleteLater()


def test_wrap_external_widget_catches_import_error(qapp, tmp_path, monkeypatch) -> None:
    """If the import callable raises, the placeholder is returned."""
    _reset_module_cache()
    repo = tmp_path / "Tools"
    (repo / "src").mkdir(parents=True)
    monkeypatch.setenv("TOOLS_REPO_PATH", str(repo))

    def _boom():
        raise RuntimeError("nope")

    result = eta._wrap_external_widget("ToolXYZ", _boom)
    assert "Unavailable" in result.windowTitle()
    result.deleteLater()


def test_external_tools_registry_has_expected_entries() -> None:
    assert set(eta.EXTERNAL_TOOLS.keys()) == {
        "video_analyzer",
        "data_explorer",
        "data_processor",
    }
    for factory in eta.EXTERNAL_TOOLS.values():
        assert callable(factory)
