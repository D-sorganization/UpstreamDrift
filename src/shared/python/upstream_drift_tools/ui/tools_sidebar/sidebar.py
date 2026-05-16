"""UnifiedToolsSidebar — tab-based sidebar panel for UpstreamDrift launchers.

Provides a collapsible tab container with configurable panels. The 'chat'
tab uses ChatDockWidget from src.shared.python.chat rather than a placeholder.

Issue #5490: replace the placeholder QLabel in the Chat tab with the real
ChatDockWidget implementation.
Issue #5616: add LayoutMode enum + Workspace tab with MATLAB-style layout.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable

from PyQt6.QtWidgets import QWidget

from .registry import WorkspaceRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LayoutMode enum (Issue #5616)
# ---------------------------------------------------------------------------


class LayoutMode(enum.Enum):
    """Sidebar layout mode selector.

    SIDEBAR (default): conventional right-edge dock with tab icons.
    MATLAB_HOME: replaces the dock with a full 70/30 horizontal split
        (command window left, workspace inspector + history right).
    """

    SIDEBAR = "sidebar"
    MATLAB_HOME = "matlab_home"


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class SidebarTabDefinition:
    """Descriptor for a single tab in UnifiedToolsSidebar.

    Attributes:
        tab_id: Unique string identifier for the tab (e.g. "chat").
        label: Human-readable label shown in the sidebar button.
        factory: Callable[sidebar] -> QWidget that creates the tab content.
        tooltip: Optional tooltip shown on the sidebar button.
    """

    tab_id: str
    label: str
    factory: Callable[[Any], QWidget]
    tooltip: str = ""


# ---------------------------------------------------------------------------
# Sidebar widget
# ---------------------------------------------------------------------------


class UnifiedToolsSidebar(QWidget):
    """Tab-based sidebar with configurable panel definitions.

    Each panel is described by a :class:`SidebarTabDefinition`. The sidebar
    lazily instantiates panel widgets via their ``factory`` callable on first
    activation so startup cost is minimised.

    Attributes:
        registry: WorkspaceRegistry shared between the workspace tab and
            the Python REPL. Seed variables here after construction so they
            appear in the variable inspector.
        layout_mode: Active LayoutMode (SIDEBAR or MATLAB_HOME).

    Usage::

        sidebar = UnifiedToolsSidebar(parent=main_window)
        sidebar.registry.set_variable('engine_manager', engine_manager)
        main_layout.addWidget(sidebar)
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._parent = parent
        self.registry: WorkspaceRegistry = WorkspaceRegistry()
        self.layout_mode: LayoutMode = LayoutMode.SIDEBAR
        self._tab_definitions: list[SidebarTabDefinition] = (
            self._default_tab_definitions()
        )
        self._active_widgets: dict[str, QWidget] = {}
        self._setup_ui()

    # ------------------------------------------------------------------
    # Tab definitions
    # ------------------------------------------------------------------

    def _default_tab_definitions(self) -> list[SidebarTabDefinition]:
        """Return the ordered list of default sidebar tab definitions.

        Tab order (left → right):
        1. chat      — AI chat assistant (Issue #5490)
        2. terminal  — Real OS shell with PTY backend (Issue #5617)
        3. python-repl — Python REPL sharing workspace variables (Issue #5617)
        4. workspace — MATLAB-style variable inspector (Issue #5616)

        Postcondition: returned list contains 'terminal', 'python-repl', and
        'workspace' tabs.
        """
        return [
            SidebarTabDefinition(
                tab_id="chat",
                label="Chat",
                tooltip="AI Chat assistant",
                factory=self._make_chat_widget,
            ),
            SidebarTabDefinition(
                tab_id="terminal",
                label="Terminal",
                tooltip=(
                    "Real OS shell (bash / pwsh / cmd) with PTY backend, "
                    "shell selector, and live cwd display."
                ),
                factory=self._make_os_terminal_widget,
            ),
            SidebarTabDefinition(
                tab_id="python-repl",
                label="Python REPL",
                tooltip=(
                    "Python execution surface sharing variables with the "
                    "workspace registry."
                ),
                factory=self._make_python_repl_widget,
            ),
            SidebarTabDefinition(
                tab_id="workspace",
                label="Workspace",
                tooltip="Variable inspector + Python REPL (MATLAB-style)",
                factory=self._make_workspace_widget,
            ),
        ]

    # ------------------------------------------------------------------
    # Tab factories
    # ------------------------------------------------------------------

    def _make_chat_widget(self, _sidebar: Any) -> QWidget:
        """Create and return the ChatDockWidget for the Chat tab.

        Uses the canonical ``_chat_dock_widget_qt.ChatDockWidget`` so the
        Chat tab provides real AI-chat functionality rather than a placeholder.
        Issue #5490 — replace placeholder QLabel with ChatDockWidget.

        Args:
            _sidebar: The sidebar instance (unused; matches factory signature).

        Returns:
            A ChatDockWidget instance.
        """
        from src.shared.python.chat._chat_dock_widget_qt import ChatDockWidget

        widget = ChatDockWidget(
            app_context="upstream_drift",
            app_name="upstream_drift",
            parent=None,
        )
        logger.debug("Chat tab: created ChatDockWidget")
        return widget

    def _make_os_terminal_widget(self, _sidebar: Any) -> QWidget:
        """Create the PTY-backed OS terminal widget for the Terminal tab.

        Uses :class:`~os_terminal.SidekickOsTerminalWidget` which runs a real
        OS shell with a shell-selector combo and live cwd label.  Degrades to a
        placeholder when Qt widgets cannot be constructed.

        Args:
            _sidebar: The sidebar instance (matches factory signature).

        Returns:
            The OS terminal Qt widget, or a placeholder on failure.

        Issue #5617.
        """
        try:
            from src.shared.python.upstream_drift_tools.ui.tools_sidebar.os_terminal import (
                SidekickOsTerminalWidget,
            )

            terminal = SidekickOsTerminalWidget(parent=self)
            if terminal.widget is not None:
                logger.debug("OS terminal tab: created SidekickOsTerminalWidget")
                return terminal.widget
        except Exception:  # noqa: BLE001 — degrade gracefully
            logger.debug("OS terminal unavailable, using placeholder")

        return self._placeholder("Terminal (OS shell unavailable)")

    def _make_python_repl_widget(self, _sidebar: Any) -> QWidget:
        """Create the Python REPL widget for the Python REPL tab.

        Args:
            _sidebar: The sidebar instance (matches factory signature).

        Returns:
            A placeholder until a Qt-capable REPL widget is wired in.

        Issue #5617.
        """
        logger.debug("Python REPL tab: created placeholder")
        return self._placeholder("Python REPL")

    def _make_workspace_widget(self, _sidebar: Any) -> QWidget:
        """Build the MATLAB-style workspace tab (Issue #5616).

        Delegates to :func:`default_tabs.build_workspace_tab`, injecting
        this sidebar as the host so the widget shares ``self.registry``.

        Args:
            _sidebar: Unused; matches factory signature.

        Returns:
            QWidget with command-window / variable-inspector layout.
        """
        from .default_tabs import build_workspace_tab

        widget = build_workspace_tab(self)
        logger.debug("Workspace tab: created MATLAB-style layout")
        return widget

    def set_context_variable(self, name: str, value: Any) -> None:
        """Set a variable in the shared workspace registry.

        Convenience method that lets REPL widgets call
        ``sidebar.set_context_variable(name, value)`` without reaching
        directly into the registry internals.

        Args:
            name: Variable name.
            value: Variable value.
        """
        self.registry.set_variable(name, value)

    def _placeholder(self, label: str) -> QWidget:
        """Return a simple placeholder label widget.

        This helper is kept for panels that have not yet been implemented.
        The Chat tab must NOT call this method (Issue #5490).

        Args:
            label: Text to display in the placeholder.

        Returns:
            A QWidget displaying the placeholder text.
        """
        from PyQt6.QtWidgets import QLabel

        placeholder = QLabel(label, self)
        logger.debug("Placeholder tab: %s", label)
        return placeholder

    # ------------------------------------------------------------------
    # UI setup (minimal skeleton)
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """Build the sidebar layout."""
        from PyQt6.QtWidgets import QVBoxLayout

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tab_definitions(self) -> list[SidebarTabDefinition]:
        """Return the active list of tab definitions.

        Returns:
            List of SidebarTabDefinition instances.
        """
        return list(self._tab_definitions)

    def get_or_create_tab(self, tab_id: str) -> QWidget | None:
        """Return the widget for ``tab_id``, creating it on first access.

        Args:
            tab_id: The tab identifier to activate.

        Returns:
            The tab widget, or None if ``tab_id`` is not defined.
        """
        if tab_id in self._active_widgets:
            return self._active_widgets[tab_id]

        definition = next(
            (t for t in self._tab_definitions if t.tab_id == tab_id), None
        )
        if definition is None:
            logger.warning("Unknown sidebar tab: %r", tab_id)
            return None

        try:
            widget = definition.factory(self)
            self._active_widgets[tab_id] = widget
            return widget
        except Exception:  # noqa: BLE001 — sidebar tab creation must not crash launcher
            logger.exception("Failed to create tab %r", tab_id)
            return None
