"""Tests for tab_context_menu.py."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("PyQt6", reason="PyQt6 not installed")

from PyQt6.QtCore import QPoint
from PyQt6.QtWidgets import QMenu, QTabBar, QWidget
from sidekick.tab_context_menu import build_tab_context_menu, show_tab_context_menu


class MockTabBar(QTabBar):
    """Mock tab bar for testing."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tab_at_val = 0
        self._map_to_global_val = QPoint(0, 0)

    def tabAt(self, pos: QPoint) -> int:
        return self._tab_at_val

    def mapToGlobal(self, pos: QPoint) -> QPoint:  # type: ignore[override]
        return self._map_to_global_val


class MockTabs:
    """Mock tabs container."""

    def __init__(self, tab_bar: QTabBar) -> None:
        self._tab_bar = tab_bar

    def tabBar(self) -> QTabBar:
        return self._tab_bar


class MockSidebar(QWidget):
    """Mock sidebar inheriting QWidget."""

    def __init__(self) -> None:
        super().__init__()
        self.tab_def = MagicMock()
        self.tab_def.popout_enabled = True
        self.tab_def.duplicate_enabled = True
        self.tab_def.help_metadata = MagicMock()
        self.tab_display_name_val: str | None = "My Tab"
        self.tab_bar = MockTabBar(self)
        self.tabs = MockTabs(self.tab_bar)

    def get_tab_definition(self, tab_id: str) -> Any:
        return self.tab_def

    def get_tab_display_name(self, tab_id: str) -> str | None:
        return self.tab_display_name_val

    def set_dock_area(self, area: str) -> None:
        pass

    def pop_out_tab(self, tab_id: str) -> None:
        pass

    def duplicate_tab(self, tab_id: str) -> None:
        pass

    def prompt_rename_tab(self, tab_id: str) -> None:
        pass

    def reset_tab_display_name(self, tab_id: str) -> None:
        pass

    def show_tab_help(self, tab_id: str) -> None:
        pass

    def set_tab_visible(self, tab_id: str, visible: bool) -> None:
        pass

    def open_configure_tabs(self) -> None:
        pass

    def set_minimized(self, minimized: bool) -> None:
        pass

    def get_tab_id_at(self, index: int) -> str | None:
        return "my_tab_id" if index >= 0 else None


def test_build_tab_context_menu(qapp: Any) -> None:
    """Test build_tab_context_menu with a mocked sidebar."""
    sidebar = MockSidebar()
    menu = build_tab_context_menu(sidebar, "test_tab_id")

    assert isinstance(menu, QMenu)


def test_build_tab_context_menu_disabled_features(qapp: Any) -> None:
    """Test build_tab_context_menu with disabled options."""
    sidebar = MockSidebar()
    sidebar.tab_def.popout_enabled = False
    sidebar.tab_def.duplicate_enabled = False
    sidebar.tab_def.help_metadata = None
    sidebar.tab_display_name_val = None

    menu = build_tab_context_menu(sidebar, "test_tab_id")

    assert isinstance(menu, QMenu)


def test_show_tab_context_menu(qapp: Any) -> None:
    """Test show_tab_context_menu calls exec on menu."""
    sidebar = MockSidebar()
    sidebar.tab_bar._tab_at_val = 0

    # Mock menu.exec to avoid opening actual GUI event loop
    with patch(
        "sidekick.ui.tools_sidebar.tab_context_menu.QtWidgets.QMenu.exec"
    ) as mock_exec:
        show_tab_context_menu(sidebar, QPoint(10, 10))
        mock_exec.assert_called_once()


def test_show_tab_context_menu_no_tab(qapp: Any) -> None:
    """Test show_tab_context_menu when no tab is selected."""
    sidebar = MockSidebar()
    sidebar.tab_bar._tab_at_val = -1

    with patch("sidekick.tab_context_menu.build_tab_context_menu") as mock_build:
        show_tab_context_menu(sidebar, QPoint(10, 10))
        mock_build.assert_not_called()
