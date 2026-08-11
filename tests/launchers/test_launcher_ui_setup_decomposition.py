"""Architecture contracts for the launcher UI-setup decomposition (#8490)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PyQt6.QtCore import QEvent, QObject

from src.launchers import (
    _launcher_navigation_ui,
    _launcher_top_bar_ui,
    launcher_ui_setup,
)
from src.launchers._launcher_navigation_ui import LauncherNavigationUIMixin
from src.launchers._launcher_top_bar_ui import (
    ClickableLabel,
    HelpButtonHoverFilter,
    LauncherTopBarUIMixin,
    RuntimeButton,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PRODUCTION_PATHS = (
    _REPO_ROOT / "src" / "launchers" / "launcher_ui_setup.py",
    _REPO_ROOT / "src" / "launchers" / "_launcher_navigation_ui.py",
    _REPO_ROOT / "src" / "launchers" / "_launcher_top_bar_ui.py",
)

pytestmark = pytest.mark.unit


class _LauncherProxy:
    """Exercise the production launcher's manager-method rebinding contract."""

    def __init__(self) -> None:
        self.ui_setup_manager = launcher_ui_setup.UISetupManager(self)

    def __getattr__(self, name: str):
        manager = self.ui_setup_manager
        if name in manager.__dict__ or hasattr(type(manager), name):
            attribute = getattr(manager, name)
            if hasattr(attribute, "__func__"):
                return attribute.__func__.__get__(self, type(self))
            return attribute
        raise AttributeError(name)


def test_launcher_ui_setup_preserves_widget_compatibility_exports() -> None:
    """Historical imports keep resolving through the public facade."""
    assert launcher_ui_setup.ClickableLabel is ClickableLabel
    assert launcher_ui_setup.HelpButtonHoverFilter is HelpButtonHoverFilter
    assert launcher_ui_setup.RuntimeButton is RuntimeButton


def test_ui_setup_manager_delegates_to_focused_mixins() -> None:
    """Navigation and top-bar behavior each have one implementation owner."""
    manager = launcher_ui_setup.UISetupManager
    assert issubclass(manager, LauncherNavigationUIMixin)
    assert issubclass(manager, LauncherTopBarUIMixin)
    assert (
        manager._setup_global_sidebar is LauncherNavigationUIMixin._setup_global_sidebar
    )
    assert manager._setup_help_menu is LauncherNavigationUIMixin._setup_help_menu
    assert manager._setup_top_bar is LauncherTopBarUIMixin._setup_top_bar
    assert (
        manager._setup_view_mode_and_zoom
        is LauncherTopBarUIMixin._setup_view_mode_and_zoom
    )


def test_extracted_methods_preserve_launcher_dynamic_rebinding() -> None:
    """Inherited manager methods still execute with the launcher as ``self``."""
    launcher = _LauncherProxy()
    sidebar_method = launcher._setup_global_sidebar
    top_bar_method = launcher._setup_top_bar

    assert sidebar_method.__self__ is launcher
    assert sidebar_method.__func__ is LauncherNavigationUIMixin._setup_global_sidebar
    assert top_bar_method.__self__ is launcher
    assert top_bar_method.__func__ is LauncherTopBarUIMixin._setup_top_bar


def test_help_hover_filter_uses_runtime_button_timer_boundary() -> None:
    """The hover filter controls hiding without reaching into timer internals."""
    runtime_button = MagicMock(spec=RuntimeButton)
    hover_filter = HelpButtonHoverFilter(runtime_button)
    watched = QObject()

    hover_filter.eventFilter(watched, QEvent(QEvent.Type.Enter))
    runtime_button.cancel_pending_hide.assert_called_once_with()

    hover_filter.eventFilter(watched, QEvent(QEvent.Type.Leave))
    runtime_button.schedule_hide.assert_called_once_with()


def test_private_mixins_keep_independent_default_builder_seams(monkeypatch) -> None:
    """Private mixins remain usable without importing the historical facade."""
    close_widget = object()
    build_close_widget = MagicMock(return_value=close_widget)
    monkeypatch.setattr(
        _launcher_navigation_ui,
        "_build_menu_bar_close_widget",
        build_close_widget,
    )
    navigation = LauncherNavigationUIMixin()
    parent = MagicMock()
    close_callback = MagicMock()
    assert (
        navigation._create_menu_bar_close_widget(parent, close_callback) is close_widget
    )
    build_close_widget.assert_called_once_with(parent, close_callback)

    build_description = MagicMock(return_value="Mixin zoom description")
    monkeypatch.setattr(
        _launcher_top_bar_ui,
        "_build_zoom_accessible_description",
        build_description,
    )
    top_bar = LauncherTopBarUIMixin()
    assert top_bar._get_zoom_accessible_description() == "Mixin zoom description"
    build_description.assert_called_once_with()


def test_launcher_ui_modules_are_under_budget_without_size_exceptions() -> None:
    """The decomposition retires both launcher-UI size exceptions."""
    for path in _PRODUCTION_PATHS:
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 1200, path

    for config_name in (
        "file_size_budget.json",
        "module_size_budget_baseline.json",
    ):
        config_path = _REPO_ROOT / "scripts" / "config" / config_name
        config = json.loads(config_path.read_text(encoding="utf-8"))
        paths = {entry["path"] for entry in config["exceptions"]}
        assert "src/launchers/launcher_ui_setup.py" not in paths


def test_moved_long_function_exceptions_are_retired() -> None:
    """Moved builders are decomposed, not re-waived under another path."""
    config_path = _REPO_ROOT / "scripts" / "config" / "architecture_budget.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    moved_symbols = {
        "UISetupManager._setup_global_sidebar",
        "UISetupManager._setup_help_menu",
        "UISetupManager._setup_top_bar_status_and_search",
        "UISetupManager._setup_view_mode_and_zoom",
    }
    exceptions = {
        (entry["path"], entry["symbol"])
        for entry in config["exceptions"]
        if entry["rule"] == "function-lines"
    }
    assert not (
        {("src/launchers/launcher_ui_setup.py", symbol) for symbol in moved_symbols}
        & exceptions
    )
