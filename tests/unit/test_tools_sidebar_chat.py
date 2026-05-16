"""Tests for UnifiedToolsSidebar chat tab (Issue #5490).

Verifies that the 'chat' tab factory in the sidebar returns a ChatDockWidget
instance rather than a placeholder QLabel.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sidebar():
    """Instantiate UnifiedToolsSidebar with all Qt widgets mocked."""
    from src.shared.python.upstream_drift_tools.ui.tools_sidebar.sidebar import (
        UnifiedToolsSidebar,
    )

    # Patch QWidget so no real Qt application is needed.
    with patch(
        "src.shared.python.upstream_drift_tools.ui.tools_sidebar.sidebar.QWidget.__init__",
        return_value=None,
    ):
        sidebar = object.__new__(UnifiedToolsSidebar)
    return sidebar


# ---------------------------------------------------------------------------
# SidebarTabDefinition data class
# ---------------------------------------------------------------------------


class TestSidebarTabDefinition:
    def test_has_required_fields(self) -> None:
        from src.shared.python.upstream_drift_tools.ui.tools_sidebar.sidebar import (
            SidebarTabDefinition,
        )

        tab = SidebarTabDefinition(
            tab_id="chat",
            label="Chat",
            factory=lambda _sidebar: None,
        )
        assert tab.tab_id == "chat"
        assert tab.label == "Chat"
        assert callable(tab.factory)


# ---------------------------------------------------------------------------
# Chat tab factory returns ChatDockWidget
# ---------------------------------------------------------------------------


class TestChatTabFactory:
    """The 'chat' tab factory must create a ChatDockWidget, not a QLabel."""

    @pytest.mark.requires_gl
    def test_chat_tab_produces_chat_dock_widget(self) -> None:
        """With real Qt available, factory returns a ChatDockWidget instance."""
        # Skip if PyQt6 is mocked (conftest replaces it with MagicMock in headless CI)
        import PyQt6.QtWidgets as _qtw

        if isinstance(_qtw, MagicMock):
            pytest.skip("PyQt6 is mocked in this environment — skipping live Qt test")

        try:
            from PyQt6.QtWebSockets import QWebSocket  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            pytest.skip("PyQt6.QtWebSockets not available")

        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])  # noqa: F841

        from src.shared.python.chat._chat_dock_widget_qt import ChatDockWidget
        from src.shared.python.upstream_drift_tools.ui.tools_sidebar.sidebar import (
            UnifiedToolsSidebar,
        )

        sidebar = UnifiedToolsSidebar.__new__(UnifiedToolsSidebar)
        # Minimal attribute setup so _default_tab_definitions() can run.
        sidebar._parent = None

        tabs = sidebar._default_tab_definitions()
        chat_tab = next((t for t in tabs if t.tab_id == "chat"), None)
        assert chat_tab is not None, "No 'chat' tab found in default tab definitions"

        widget = chat_tab.factory(sidebar)
        assert isinstance(widget, ChatDockWidget), (
            f"Expected ChatDockWidget, got {type(widget).__name__}"
        )

    def test_chat_tab_not_qlabel_no_qt(self) -> None:
        """Without constructing Qt widgets, verify the factory imports ChatDockWidget."""
        from src.shared.python.upstream_drift_tools.ui.tools_sidebar.sidebar import (
            UnifiedToolsSidebar,
        )

        # Just check the source of the tab definitions uses ChatDockWidget.
        import inspect

        source = inspect.getsource(UnifiedToolsSidebar._default_tab_definitions)
        # Must reference ChatDockWidget, not just QLabel
        assert "ChatDockWidget" in source, (
            "Chat tab factory does not reference ChatDockWidget"
        )
        # Must NOT use _placeholder for the chat tab
        # (find 'chat' tab definition and verify it is not a placeholder)
        assert (
            "_placeholder" not in source.split("ChatDockWidget")[0].split('"chat"')[-1]
        ), "Chat tab still uses _placeholder factory"

    def test_default_tabs_include_chat(self) -> None:
        """_default_tab_definitions returns a list containing a 'chat' entry."""
        from src.shared.python.upstream_drift_tools.ui.tools_sidebar.sidebar import (
            SidebarTabDefinition,
            UnifiedToolsSidebar,
        )

        # Call as unbound method; pass a minimal fake sidebar.
        fake_sidebar = MagicMock(spec=UnifiedToolsSidebar)
        tabs = UnifiedToolsSidebar._default_tab_definitions(fake_sidebar)
        assert isinstance(tabs, list)
        chat_tab = next((t for t in tabs if t.tab_id == "chat"), None)
        assert chat_tab is not None
        assert isinstance(chat_tab, SidebarTabDefinition)
