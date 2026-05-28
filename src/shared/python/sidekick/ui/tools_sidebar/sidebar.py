"""Unified dockable tools sidebar for Qt host applications."""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """Fallback StrEnum for Python < 3.11."""


from pathlib import Path
from typing import Any, cast

from . import (
    SIDEKICK_DOCK_OBJECT_NAME,
    SIDEKICK_SIDEBAR_OBJECT_NAME,
    SIDEKICK_TAB_BAR_OBJECT_NAME,
    SIDEKICK_TABS_OBJECT_NAME,
    SidekickDesignTokens,
    sidekick_qss,
)
from .calculator_assist import calculator_context_preferences, calculator_state_fields
from .default_tabs import (
    WorkspaceTableWidget,
    build_default_tab_definitions,
    refresh_workspace_list,
    set_project_explorer_root,
)
from .dock_title_bar import SidekickDockTitleBar
from .help_content import render_help_markdown
from .qt_compat import QtCore, QtWidgets, Signal, all_sidebar_dock_features, dock_area
from .registry import WorkspaceRegistry
from .runtime_tabs import PythonReplWidget
from .state import SidebarState
from .state_profile_actions import StateProfileMixin
from .tab_context_menu import show_tab_context_menu
from .tab_definition import SidebarTabDefinition
from .tab_display_names import TabDisplayNameMixin
from .tab_popout import TabPopoutMixin
from .tab_settings_panel import TabSettingsMixin, build_tab_settings_toolbar
from .tab_visibility import (
    initially_visible_tab_ids,
    sanitize_tab_state,
    with_default_tab_visibility,
    without_default_tab_visibility,
)
from .theme_settings import resolve_sidekick_theme


class LayoutMode(StrEnum):
    """Sidebar layout strategies.

    ``SIDEBAR`` is the classic dockable tab strip; ``MATLAB_HOME`` is a
    two-pane layout (command window + workspace inspector) modelled on the
    MATLAB Home tab.
    """

    SIDEBAR = "sidebar"
    MATLAB_HOME = "matlab_home"


class MatlabHomeWidget(QtWidgets.QWidget):
    """Two-pane MATLAB-home layout: command window + workspace inspector.

    Both panes are bound to the host sidebar's :class:`WorkspaceRegistry`
    so assignments made in the command window appear immediately in the
    workspace table (LOD: only the registry reference is shared; no reach
    into theme tokens or sidebar internals).

    Args:
        sidebar: Host sidebar exposing ``registry`` and ``set_context_variable``.
        parent: Optional Qt parent.

    Raises:
        TypeError: If ``sidebar`` is ``None``.
    """

    def __init__(
        self,
        *,
        sidebar: UnifiedToolsSidebar,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        if sidebar is None:
            raise TypeError("sidebar must be provided")
        super().__init__(parent)
        self.setObjectName("SidekickMatlabHomeWidget")
        self._sidebar = sidebar
        self._workspace = WorkspaceTableWidget(registry=sidebar.registry, parent=self)
        self._command_window = PythonReplWidget(
            registry=sidebar.registry,
            set_variable=sidebar.set_context_variable,
            parent=self,
        )

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
        splitter.setObjectName("SidekickMatlabHomeSplitter")
        splitter.addWidget(self._command_window)
        splitter.addWidget(self._workspace)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    def command_window_widget(self) -> PythonReplWidget:
        """Return the embedded Python REPL command window."""
        return self._command_window

    def workspace_widget(self) -> WorkspaceTableWidget:
        """Return the embedded MATLAB-style workspace table."""
        return self._workspace


class UnifiedToolsSidebar(
    QtWidgets.QWidget,
    TabDisplayNameMixin,
    TabSettingsMixin,
    StateProfileMixin,
    TabPopoutMixin,
):
    """Tabbed sidebar that can be installed as a tear-off dock widget."""

    file_open_requested = Signal(str)
    context_updated = Signal(dict)
    tool_launch_requested = Signal(str, object)

    def __init__(
        self,
        project_root: str | Path | None = None,
        registry: WorkspaceRegistry | None = None,
        state: SidebarState | None = None,
        tab_definitions: list[SidebarTabDefinition] | None = None,
        design_tokens: SidekickDesignTokens | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(SIDEKICK_SIDEBAR_OBJECT_NAME)
        self.registry = registry or WorkspaceRegistry()
        self._state = state or SidebarState()
        self._design_tokens = resolve_sidekick_theme(
            parent_tokens=design_tokens,
            settings=self._state.theme_settings,
        )
        self._dock_widget: QtWidgets.QDockWidget | None = None
        self._title_bar: SidekickDockTitleBar | None = None
        self._is_collapsed: bool = False
        self._expanded_width = self._state.width
        # Per-tab last pop-out positions (issue #2881)
        self._popout_positions: dict[str, tuple[int, int]] = {}
        self._tab_ids: list[str] = []
        self._tab_definitions: dict[str, SidebarTabDefinition] = {}
        self._tab_widgets: dict[str, QtWidgets.QWidget] = {}
        self._popout_windows: dict[str, QtWidgets.QMainWindow] = {}
        self._duplicate_counts: dict[str, int] = {}
        self._help_dialog: QtWidgets.QDialog | None = None
        self._settings_dialog: QtWidgets.QDialog | None = None
        self._workspace_list: QtWidgets.QListWidget | None = None
        self._workspace_table: QtWidgets.QWidget | None = None
        self._settings_button: QtWidgets.QToolButton | None = None
        self._project_root = Path(project_root or Path.cwd()).expanduser().resolve()
        self._layout_mode = _coerce_layout_mode(self._state.layout_mode)

        self.tabs = QtWidgets.QTabWidget(self)
        self.tabs.setObjectName(SIDEKICK_TABS_OBJECT_NAME)
        self.tabs.tabBar().setObjectName(SIDEKICK_TAB_BAR_OBJECT_NAME)
        self.tabs.setMovable(True)
        # Never elide tab labels: when the user-visible name is the only way
        # to tell tabs apart, truncation is worse than overflow. Scroll
        # buttons (enabled below) keep all tabs reachable when the bar runs
        # out of horizontal space.
        self.tabs.tabBar().setElideMode(QtCore.Qt.TextElideMode.ElideNone)
        self.tabs.setUsesScrollButtons(True)
        self.tabs.currentChanged.connect(self._emit_context)
        self.tabs.tabBar().tabMoved.connect(self._sync_tab_order_from_widget)

        policy_enum = getattr(QtCore.Qt, "ContextMenuPolicy", None)
        self.tabs.tabBar().setContextMenuPolicy(
            policy_enum.CustomContextMenu
            if policy_enum
            else QtCore.Qt.CustomContextMenu
        )
        self.tabs.tabBar().customContextMenuRequested.connect(
            self._show_tab_context_menu
        )

        self.setStyleSheet(sidekick_qss(self._design_tokens))

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(build_tab_settings_toolbar(self))
        layout.addWidget(self.tabs)

        self.configure_tabs(
            tab_definitions
            if tab_definitions is not None
            else self._default_tab_definitions()
        )

        self.apply_state(self._state)

    def minimumSizeHint(self) -> QtCore.QSize:
        """Override minimumSizeHint to allow aggressive resizing of the sidebar."""
        return QtCore.QSize(100, 0)

    @property
    def dock_widget(self) -> QtWidgets.QDockWidget | None:
        """Return the installed dock widget, if any."""
        return self._dock_widget

    @property
    def dock(self) -> QtWidgets.QDockWidget | None:
        """Alias for :attr:`dock_widget` — preferred by UI tests."""
        return self._dock_widget

    @property
    def project_root(self) -> Path:
        """Return the current project root used by runtime tabs."""
        return self._project_root

    # ── Dock chrome helpers (issue #2881) ─────────────────────────────────────

    def dock_title_widget(self) -> SidekickDockTitleBar | None:
        """Return the custom title-bar widget installed on the dock.

        Returns ``None`` if the dock has not been installed yet (i.e.
        :meth:`install_as_dock` has not been called).
        """
        return self._title_bar

    def is_collapsed(self) -> bool:
        """Return ``True`` when the sidebar is in collapsed (icon-strip) mode."""
        return self._is_collapsed

    def toggle_collapsed(self) -> None:
        """Toggle between collapsed (icon-strip) and expanded states.

        In the collapsed state the sidebar shrinks to a narrow icon strip
        (``≤ 56 px``) and the tab content is hidden.  Calling this method
        again restores the previous expanded width.
        """
        if self._is_collapsed:
            # Expand
            self._is_collapsed = False
            self.tabs.setVisible(True)
            self.setMaximumWidth(16777215)
            self._apply_expanded_width()
        else:
            # Collapse
            self._is_collapsed = True
            self._expanded_width = max(self.width(), self._expanded_width)
            self.tabs.setVisible(False)
            self.setMaximumWidth(56)
        self._emit_context()

    def _apply_expanded_width(self) -> None:
        target = max(self._expanded_width, 240)
        parent = self.parent()
        if isinstance(parent, QtWidgets.QSplitter):
            sizes = parent.sizes()
            idx = parent.indexOf(self)
            if idx != -1 and sum(sizes) > 0:
                diff = target - sizes[idx]
                sizes[idx] = target
                if idx > 0:
                    sizes[idx - 1] = max(0, sizes[idx - 1] - diff)
                parent.setSizes(sizes)
                return
        self.resize(target, self.height())

    def toggle_visibility(self) -> None:
        """Toggle the dock's visibility (Ctrl+B shortcut handler).

        Hides the dock when visible, shows it when hidden.

        Pre: dock has been installed via :meth:`install_as_dock`.
        Post: ``self.dock.isVisible()`` is the negation of the pre-call value.
        """
        if self._dock_widget is None:
            return
        if self._dock_widget.isVisible():
            self._dock_widget.hide()
        else:
            self._dock_widget.show()

    def register_shortcuts(self, main_window: QtWidgets.QMainWindow) -> None:
        """Wire Ctrl+B and Ctrl+Shift+B shortcuts onto ``main_window``.

        Args:
            main_window: The host main window that will receive the key events.

        Keyboard bindings registered:
            - ``Ctrl+B`` → :meth:`toggle_visibility`
            - ``Ctrl+Shift+B`` → :meth:`toggle_collapsed`
        """
        try:
            from PyQt6.QtGui import (  # type: ignore[attr-defined]
                QKeySequence,
                QShortcut,
            )
        except ImportError:
            try:
                from PyQt5.QtGui import QKeySequence  # type: ignore[no-redef]
                from PyQt5.QtWidgets import QShortcut  # type: ignore[no-redef]
            except ImportError:
                return  # Qt not available — gracefully skip shortcut registration

        sc_toggle = QShortcut(QKeySequence("Ctrl+B"), main_window)
        sc_toggle.activated.connect(self.toggle_visibility)  # type: ignore[union-attr]

        sc_collapse = QShortcut(QKeySequence("Ctrl+Shift+B"), main_window)
        sc_collapse.activated.connect(self.toggle_collapsed)  # type: ignore[union-attr]

    def re_dock(self, tab_id: str) -> bool:
        """Re-dock a tab that is currently in a floating pop-out window.

        Args:
            tab_id: The stable tab identifier to re-dock.

        Returns:
            ``True`` on success.

        Raises:
            RuntimeError: If ``tab_id`` is not currently floating (DbC
                precondition — the tab must be in a pop-out window).
        """
        if tab_id not in self._popout_windows:
            raise RuntimeError(
                f"re_dock precondition failed: tab '{tab_id}' is not floating. "
                "Only floating tabs can be re-docked."
            )
        return self.redock_tab(tab_id)

    def add_tab(self, tab_id: str, title: str, widget: QtWidgets.QWidget) -> None:
        """Add a tab with a stable persistence id."""
        if tab_id in self._tab_ids:
            raise ValueError(f"Duplicate sidebar tab id: {tab_id}")
        self._tab_ids.append(tab_id)
        self._tab_widgets[tab_id] = widget
        self.tabs.addTab(widget, self._tab_display_name(tab_id, title))

    def configure_tabs(self, definitions: list[SidebarTabDefinition]) -> None:
        """Configure the available tab set for this Sidekick instance."""
        self.tabs.clear()
        self._tab_ids.clear()
        self._tab_widgets.clear()
        self._tab_definitions = {
            definition.tab_id: definition for definition in definitions
        }
        self._configure_tab_settings()
        visible_defaults = initially_visible_tab_ids(definitions, self._state)
        for definition in definitions:
            if definition.tab_id in visible_defaults:
                self._add_defined_tab(definition)

    def available_tab_ids(self) -> list[str]:
        """Return all configured tab ids, including hidden tabs."""
        return list(self._tab_definitions)

    def visible_tab_ids(self) -> list[str]:
        """Return tab ids currently docked in the sidebar."""
        return list(self._tab_ids)

    def hidden_tab_ids(self) -> list[str]:
        """Return configured tabs that are currently hidden."""
        return [
            tab_id
            for tab_id in self._tab_definitions
            if tab_id not in self._tab_ids and tab_id not in self._popout_windows
        ]

    def get_tab_definition(self, tab_id: str) -> SidebarTabDefinition | None:
        """Return the definition for ``tab_id``, or *None* if not found."""
        return self._tab_definitions.get(tab_id)

    def get_tab_id_at(self, index: int) -> str | None:
        """Return the stable tab id at visual ``index``, or *None* if out of range."""
        if 0 <= index < len(self._tab_ids):
            return self._tab_ids[index]
        return None

    def get_tab_display_name(self, tab_id: str) -> str | None:
        """Return the persisted custom display name for ``tab_id``, or *None*."""
        result: str | None = self._state.tab_display_names.get(tab_id)
        return result

    def prompt_rename_tab(self, tab_id: str) -> None:
        """Open an input dialog so the user can rename ``tab_id``."""
        self._prompt_rename_tab(tab_id)

    def register_workspace_list_widget(self, widget: QtWidgets.QListWidget) -> None:
        """Register the workspace list widget used by the workspace tab."""
        self._workspace_list = widget

    def workspace_list_widget(self) -> QtWidgets.QListWidget | None:
        """Return the registered workspace list widget, or *None*."""
        return getattr(self, "_workspace_list", None)

    def register_workspace_table_widget(self, widget: QtWidgets.QWidget) -> None:
        """Register the MATLAB-style workspace table widget."""
        self._workspace_table = widget

    def workspace_table_widget(self) -> QtWidgets.QWidget | None:
        """Return the registered workspace table widget, or *None*."""
        return getattr(self, "_workspace_table", None)

    def layout_mode(self) -> LayoutMode:
        """Return the currently active sidebar layout strategy."""
        return self._layout_mode

    def set_layout_mode(self, mode: LayoutMode | str) -> None:
        """Switch between the classic sidebar and MATLAB-home layout.

        Persists the choice on ``_state.layout_mode`` so it is captured by
        :meth:`snapshot_state` and survives a host restart.

        Raises:
            TypeError: If ``mode`` is ``None`` or not a ``LayoutMode`` value.
            ValueError: If ``mode`` is an unknown string.
        """
        if mode is None:
            raise TypeError("mode must not be None")
        resolved = _coerce_layout_mode(mode)
        self._layout_mode = resolved
        self._state.layout_mode = resolved.value
        self._emit_context()

    def register_settings_button(self, button: QtWidgets.QToolButton) -> None:
        """Register the settings toolbar button for enabled-state management."""
        self._settings_button = button

    def move_tab(self, tab_id: str, index: int) -> bool:
        """Move a visible tab to ``index``."""
        if tab_id not in self._tab_ids:
            return False
        current = self._tab_ids.index(tab_id)
        target = max(0, min(index, len(self._tab_ids) - 1))
        if current == target:
            return True
        self.tabs.tabBar().moveTab(current, target)
        self._sync_tab_order_from_widget()
        return True

    def set_tab_visible(self, tab_id: str, visible: bool) -> bool:
        """Show or hide a configured tab."""
        if visible:
            if tab_id in self._tab_ids:
                return True
            definition = self._tab_definitions.get(tab_id)
            if definition is None:
                return False
            self._add_defined_tab(definition)
            self._sync_tab_order_from_widget()
            self._emit_context()
            return True

        if tab_id not in self._tab_ids:
            return tab_id in self._tab_definitions
        if len(self._tab_ids) == 1:
            return False
        index = self._tab_ids.index(tab_id)
        widget = self.tabs.widget(index)
        self.tabs.removeTab(index)
        self._tab_ids.pop(index)
        self._tab_widgets.pop(tab_id, None)
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()
        self._emit_context()
        return True

    def set_default_tab_visible(self, tab_id: str, visible: bool) -> bool:
        """Persist a default visibility preference for a configured tab."""
        candidate = with_default_tab_visibility(
            self._state,
            list(self._tab_definitions.values()),
            tab_id,
            visible,
        )
        if candidate is None:
            return False
        self._state = candidate
        self._emit_context()
        return True

    def reset_default_tab_visibility(self) -> None:
        """Reset persisted default visibility preferences to host defaults."""
        self._state = without_default_tab_visibility(self._state)
        self._emit_context()

    def set_minimized(self, minimized: bool) -> None:
        """Collapse or expand the sidebar without destroying tab state."""
        self._state.minimized = minimized
        if minimized:
            self._expanded_width = max(self.width(), self._expanded_width)
            self.tabs.setVisible(False)
            self.setMaximumWidth(56)
        else:
            self.tabs.setVisible(True)
            self.setMaximumWidth(16777215)
            self._apply_expanded_width()
        self._emit_context()

    def set_dock_area(self, area: str) -> bool:
        """Move the installed dock widget to the left or right side."""
        if area not in {"left", "right"}:
            return False
        self._state.dock_area = area
        if self._dock_widget is not None:
            host = self._dock_widget.parent()
            if hasattr(host, "addDockWidget"):
                host.addDockWidget(dock_area(area), self._dock_widget)
        self._emit_context()
        return True

    def install_as_dock(
        self,
        main_window: QtWidgets.QMainWindow,
        *,
        area: str | None = None,
        title: str = "Tools",
        state_path: str | Path | None = None,
    ) -> QtWidgets.QDockWidget:
        """Install this sidebar into ``main_window`` as a QDockWidget.

        Replaces the default QDockWidget title bar with a custom
        :class:`~.dock_title_bar.SidekickDockTitleBar` that exposes
        close (×) and collapse (—) buttons per issue #2881.
        """
        if state_path is not None:
            self.apply_state(SidebarState.load_json(state_path))

        dock = QtWidgets.QDockWidget(title, main_window)
        dock.setObjectName(SIDEKICK_DOCK_OBJECT_NAME)
        dock.setFeatures(all_sidebar_dock_features())
        dock.setWidget(self)
        dock.setFloating(self._state.floating)
        dock.resize(self._state.width, self._state.height)
        main_window.addDockWidget(dock_area(area or self._state.dock_area), dock)
        self._dock_widget = dock

        # Install custom title bar (issue #2881)
        self._title_bar = SidekickDockTitleBar(
            title,
            on_close=self.toggle_visibility,
            on_collapse=self.toggle_collapsed,
            parent=dock,
        )
        dock.setTitleBarWidget(self._title_bar)

        self._emit_context()
        return dock

    def snapshot_state(self) -> SidebarState:
        """Return current dock, size, and active-tab state."""
        dock_area_name = self._state.dock_area
        floating = self._state.floating
        width = self.width()
        height = self.height()

        if self._dock_widget is not None:
            floating = self._dock_widget.isFloating()
            width = self._dock_widget.width()
            height = self._dock_widget.height()
            host = self._dock_widget.parent()
            if hasattr(host, "dockWidgetArea"):
                area = host.dockWidgetArea(self._dock_widget)
                if area == dock_area("left"):
                    dock_area_name = "left"
                elif area == dock_area("right"):
                    dock_area_name = "right"

        return SidebarState(
            dock_area=dock_area_name,
            floating=floating,
            width=width,
            height=height,
            active_tab=self.active_tab_id(),
            layout_mode=self._layout_mode.value,
            minimized=self._state.minimized,
            tab_order=self.visible_tab_ids(),
            hidden_tabs=self.hidden_tab_ids(),
            default_visible_tabs=self._state.default_visible_tabs,
            default_hidden_tabs=self._state.default_hidden_tabs,
            popped_out_tabs=list(self._popout_windows),
            tab_display_names=self._state.tab_display_names,
            theme_settings=self._state.theme_settings,
            tab_settings=self._tab_settings_payload(),
            **calculator_state_fields(self._state),
        )

    def save_state(self, path: str | Path) -> SidebarState:
        state = self.snapshot_state()
        state.save_json(path)
        self._state = state
        return state

    def apply_state(self, state: SidebarState) -> None:
        self._state = state
        self._layout_mode = _coerce_layout_mode(state.layout_mode)
        self.resize(state.width, state.height)
        self._apply_tab_state(state)
        self.set_minimized(state.minimized)
        self.set_active_tab(state.active_tab)
        if self._dock_widget is not None:
            self._dock_widget.setFloating(state.floating)
            self._dock_widget.resize(state.width, state.height)

    def active_tab_id(self) -> str:
        index = int(self.tabs.currentIndex())
        if 0 <= index < len(self._tab_ids):
            return self._tab_ids[index]
        return self._tab_ids[0] if self._tab_ids else ""

    def set_active_tab(self, tab_id: str) -> bool:
        if tab_id not in self._tab_ids:
            return False
        self.tabs.setCurrentIndex(self._tab_ids.index(tab_id))
        return True

    def set_context_variable(self, name: str, value: Any) -> None:
        self.registry.set(name, value)
        self.refresh_workspace()

    def request_tool_launch(self, tool_id: str, payload: dict[str, Any]) -> None:
        """Emit a structured host-tool launch request."""
        self.tool_launch_requested.emit(tool_id, dict(payload))

    def refresh_workspace(self) -> None:
        """Refresh workspace-derived UI and emit the latest sidebar context."""
        refresh_workspace_list(self)
        self._emit_context()

    def set_design_tokens(self, design_tokens: SidekickDesignTokens) -> None:
        """Apply a new Sidekick token set to this sidebar."""
        self._design_tokens = design_tokens
        self.setStyleSheet(sidekick_qss(self._design_tokens))
        self._emit_context()

    def set_theme(self, theme_name: str) -> None:
        """Apply a shared fleet theme by name to this sidebar."""
        self.set_design_tokens(SidekickDesignTokens.from_shared_theme(theme_name))

    def set_project_root(self, project_root: str | Path) -> None:
        self._project_root = Path(project_root).expanduser().resolve()
        set_project_explorer_root(self._tab_widgets.get("files"), self._project_root)
        self._emit_context()

    def _default_tab_definitions(self) -> list[SidebarTabDefinition]:
        return cast(
            list[SidebarTabDefinition],
            build_default_tab_definitions(self, SidebarTabDefinition),
        )

    def _add_defined_tab(self, definition: SidebarTabDefinition) -> None:
        widget = definition.factory(self)
        self.add_tab(definition.tab_id, definition.title, widget)

    def _show_tab_context_menu(self, pos: QtCore.QPoint) -> None:
        show_tab_context_menu(self, pos)

    def tab_help_metadata(self, tab_id: str) -> dict[str, str]:
        """Return a copy of the configured help metadata for one tab id."""
        definition = self._tab_definitions.get(tab_id)
        if definition is None:
            return {}
        return dict(definition.help_metadata)

    def show_tab_help(self, tab_id: str | None = None) -> bool:
        """Open a compact dialog for one tab's help metadata."""
        resolved_tab_id = tab_id or self.active_tab_id()
        metadata = self.tab_help_metadata(resolved_tab_id)
        if not metadata:
            return False

        dialog = QtWidgets.QDialog(self)
        dialog.setObjectName(f"SidekickTabHelpDialog_{resolved_tab_id}")
        dialog.setWindowTitle(f"{self.tab_display_name(resolved_tab_id)} Help")
        dialog.resize(480, 360)
        layout = QtWidgets.QVBoxLayout(dialog)
        browser = QtWidgets.QTextBrowser(dialog)
        browser.setOpenExternalLinks(True)
        browser.setMarkdown(render_help_markdown(metadata))
        layout.addWidget(browser)
        dialog.show()
        self._help_dialog = dialog
        return True

    def _emit_context(self) -> None:
        self._refresh_settings_button()
        self.context_updated.emit(
            {
                "active_tab": self.active_tab_id(),
                "project_root": str(self._project_root),
                "dock_area": self._state.dock_area,
                "minimized": self._state.minimized,
                "visible_tabs": self.visible_tab_ids(),
                "hidden_tabs": self.hidden_tab_ids(),
                "popped_out_tabs": list(self._popout_windows),
                "tab_display_names": dict(self._state.tab_display_names),
                "tab_help": {
                    tab_id: dict(definition.help_metadata)
                    for tab_id, definition in self._tab_definitions.items()
                    if definition.help_metadata
                },
                "preferences": calculator_context_preferences(self._state),
                "tab_settings": self._tab_settings_payload(),
                "workspace_variables": [
                    variable.to_metadata() for variable in self.registry.variables()
                ],
            }
        )

    def _sync_tab_order_from_widget(self, *_args: Any) -> None:
        ordered: list[str] = []
        for index in range(self.tabs.count()):
            widget = self.tabs.widget(index)
            for tab_id, tab_widget in self._tab_widgets.items():
                if tab_widget is widget:
                    ordered.append(tab_id)
                    break
        if len(ordered) == len(self._tab_ids):
            self._tab_ids = ordered
        self._refresh_settings_button()
        self._emit_context()

    def _apply_tab_state(self, state: SidebarState) -> None:
        self._state = sanitize_tab_state(state, self._tab_definitions)
        state = self._state
        for tab_id in list(self._tab_ids):
            if tab_id in state.hidden_tabs:
                self.set_tab_visible(tab_id, False)
        for tab_id in state.tab_order:
            if tab_id in self._tab_definitions and tab_id not in self._tab_ids:
                self.set_tab_visible(tab_id, True)
        for index, tab_id in enumerate(state.tab_order):
            self.move_tab(tab_id, index)
        for tab_id in self._tab_definitions:
            self._refresh_tab_display_name(tab_id)


SidekickSidebar = UnifiedToolsSidebar


def _coerce_layout_mode(value: LayoutMode | str | None) -> LayoutMode:
    """Coerce ``value`` to a :class:`LayoutMode`, defaulting to SIDEBAR."""
    if value is None:
        return LayoutMode.SIDEBAR
    if isinstance(value, LayoutMode):
        return value
    if not isinstance(value, str):
        raise TypeError("layout mode must be a LayoutMode or str")
    try:
        return LayoutMode(value)
    except ValueError as exc:
        raise ValueError(f"unknown layout mode: {value!r}") from exc
