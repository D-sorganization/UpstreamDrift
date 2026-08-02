"""Tests for selected_tab_panel.py."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("PyQt6", reason="PyQt6 not installed")

from PyQt6.QtWidgets import QDialog, QToolBar, QToolButton, QWidget
from sidekick.selected_tab_panel import (
    SIDEKICK_TAB_SETTINGS_BUTTON_OBJECT_NAME,
    TabSettingsMixin,
    build_tab_settings_dialog,
    build_tab_settings_toolbar,
)


class DummyUnifiedSidebar(QWidget, TabSettingsMixin):
    """Dummy class to test TabSettingsMixin."""

    def __init__(self) -> None:
        super().__init__()
        self._tab_definitions: dict[str, Any] = {}
        self._tab_ids: list[str] = []
        self._state = MagicMock()
        self._settings_button = QToolButton(self)
        self._emit_context_called = False
        self._set_tab_visible_called = False

    def _emit_context(self) -> None:
        self._emit_context_called = True

    def active_tab_id(self) -> str:
        return "active_tab_1"

    def tab_display_name(self, tab_id: str) -> str:
        return f"Display {tab_id}"

    def register_settings_button(self, button: Any) -> None:
        pass

    def visible_tab_ids(self) -> list[str]:
        return ["active_tab_1"]

    def set_tab_visible(self, tab_id: str, checked: bool) -> bool:
        self._set_tab_visible_called = True
        return True


def test_constants() -> None:
    """Test constants are exported and correct."""
    assert SIDEKICK_TAB_SETTINGS_BUTTON_OBJECT_NAME == "SidekickActiveTabSettings"


def test_tab_settings_mixin_methods(qapp: Any) -> None:
    """Test methods on TabSettingsMixin via DummyUnifiedSidebar."""
    sidebar = DummyUnifiedSidebar()

    # Mock settings store behavior
    mock_store = MagicMock()
    mock_store.settings_for.return_value = {"values": {"foo": "bar"}}
    mock_store.update_settings.return_value = {"values": {"foo": "baz"}}
    mock_store.raw_settings.return_value = {"tab_settings": "data"}
    mock_store.materialized_settings.return_value = {"active_tab_1": {"values": {}}}

    sidebar._settings_store = mock_store

    # Test tab_settings
    settings = sidebar.tab_settings("active_tab_1")
    assert settings == {"values": {"foo": "bar"}}
    mock_store.settings_for.assert_called_once_with("active_tab_1")

    # Test update_tab_settings
    updated = sidebar.update_tab_settings("active_tab_1", {"foo": "baz"})
    assert updated == {"values": {"foo": "baz"}}
    assert sidebar._emit_context_called is True
    mock_store.update_settings.assert_called_once_with("active_tab_1", {"foo": "baz"})

    # Test _tab_settings_payload
    payload = sidebar._tab_settings_payload()
    assert payload == {"active_tab_1": {"values": {}}}


def test_tab_settings_mixin_refresh_button(qapp: Any) -> None:
    """Test refresh settings button logic."""
    sidebar = DummyUnifiedSidebar()

    # If no tab ids, button disabled
    sidebar._tab_ids = []
    sidebar._refresh_settings_button()
    assert not sidebar._settings_button.isEnabled()

    # If active tab definition is not found or has no settings
    sidebar._tab_ids = ["active_tab_1"]
    sidebar._tab_definitions = {"active_tab_1": MagicMock(settings=None)}
    sidebar._refresh_settings_button()
    assert not sidebar._settings_button.isEnabled()

    # If active tab definition has settings
    sidebar._tab_definitions = {"active_tab_1": MagicMock(settings=MagicMock())}
    sidebar._refresh_settings_button()
    assert sidebar._settings_button.isEnabled()


def test_tab_settings_mixin_configure_tabs(qapp: Any) -> None:
    """Test configuring tab settings."""
    sidebar = DummyUnifiedSidebar()
    sidebar._tab_definitions = {"t1": MagicMock()}

    with patch(
        "sidekick.ui.tools_sidebar.tab_settings_panel.SidebarTabSettingsStore"
    ) as mock_store_cls:
        sidebar._configure_tab_settings()
        mock_store_cls.assert_called_once()


def test_tab_settings_mixin_open_active_tab_settings(qapp: Any) -> None:
    """Test open_active_tab_settings."""
    sidebar = DummyUnifiedSidebar()

    # No definition
    sidebar._tab_definitions = {}
    assert sidebar.open_active_tab_settings() is False

    # With definition but no settings
    def_mock = MagicMock(settings=None)
    sidebar._tab_definitions = {"active_tab_1": def_mock}
    assert sidebar.open_active_tab_settings() is False

    # With settings
    settings_mock = MagicMock(widget_factory=MagicMock(return_value=MagicMock()))
    def_mock = MagicMock(settings=settings_mock)
    sidebar._tab_definitions = {"active_tab_1": def_mock}

    # Mock dialog execution
    with patch(
        "sidekick.ui.tools_sidebar.tab_settings_panel.build_tab_settings_dialog"
    ) as mock_build:
        dialog_mock = MagicMock()
        mock_build.return_value = dialog_mock
        assert sidebar.open_active_tab_settings() is True
        dialog_mock.exec.assert_called_once()


def test_build_tab_settings_toolbar(qapp: Any) -> None:
    """Test build_tab_settings_toolbar."""
    sidebar = DummyUnifiedSidebar()
    toolbar = build_tab_settings_toolbar(sidebar)
    assert isinstance(toolbar, QToolBar)


def test_build_tab_settings_dialog(qapp: Any) -> None:
    """Test build_tab_settings_dialog."""
    sidebar = DummyUnifiedSidebar()

    # Mock display name and settings on sidebar
    sidebar.tab_display_name = lambda tid: "Test Tab"  # type: ignore[assignment]
    sidebar.tab_settings = lambda tid: {"values": {"key": "val"}}  # type: ignore[assignment]

    # With provided content
    content = QWidget()
    dialog = build_tab_settings_dialog(sidebar, "test_tab", content)
    assert isinstance(dialog, QDialog)

    # Without provided content (uses fallback labels)
    dialog2 = build_tab_settings_dialog(sidebar, "test_tab", None)
    assert isinstance(dialog2, QDialog)


def test_configure_tabs_dialog(qapp: Any) -> None:
    """Test ConfigureTabsDialog UI creation and toggling."""
    sidebar = DummyUnifiedSidebar()
    sidebar._tab_definitions = {
        "t1": MagicMock(title="Tab 1"),
        "t2": MagicMock(title="Tab 2"),
    }

    with patch(
        "sidekick.ui.tools_sidebar.tab_settings_panel.QtWidgets.QDialog.exec"
    ) as mock_exec:
        sidebar.open_configure_tabs()
        mock_exec.assert_called_once()
