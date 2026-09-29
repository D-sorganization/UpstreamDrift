# mypy: ignore-errors
"""Unified dockable tools sidebar for Qt host applications."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.shared.python.compatibility import StrEnum

if TYPE_CHECKING:
    from .dock_title_bar import SidekickDockTitleBar

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
from .dock_chrome import DockChromeController
from .help_content import render_help_markdown
from .qt_compat import (
    QtCore,
    QtWidgets,
    Signal,
)
from .registry import WorkspaceRegistry
from .runtime_tabs import PythonReplWidget
from .state import SidebarState
from .state_profile_actions import StateProfileMixin
from .tab_collection import TabCollection
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
from .visibility_persistence import (
    _QS_APP,  # noqa: F401  # re-exported for backward compat
    _QS_ORG,  # noqa: F401  # re-exported for backward compat
    _QS_VISIBLE_TABS_KEY,  # noqa: F401  # re-exported for backward compat
    VisibilityPersistence,
)

_logger = logging.getLogger(__name__)

__all__ = [
    "LayoutMode",
    "MatlabHomeWidget",
    "SidekickSidebar",
    "UnifiedToolsSidebar",
]


class LayoutMode(StrEnum):
    """Sidebar layout strategies.

    ``SIDEBAR`` is the classic dockable tab strip; ``MATLAB_HOME`` is a
    two-pane layout (command window + workspace inspector) modelled on the
    MATLAB Home tab.
    """

    SIDEBAR = "sidebar"
    MATLAB_HOME = "matlab_home"


class _HostCloseFilter(QtCore.QObject):
    """Invoke one lifecycle callback when the owning host begins closing."""

    def __init__(
        self,
        on_close: Callable[[], None],
        parent: QtCore.QObject,
    ) -> None:
        if not callable(on_close):
            raise TypeError("on_close must be callable")
        super().__init__(parent)
        self._on_close = on_close

    def eventFilter(self, watched: object, event: object) -> bool:  # noqa: N802
        """Run host cleanup before Qt starts closing child widgets."""
        event_type = getattr(event, "type", None)
        if callable(event_type) and event_type() == QtCore.QEvent.Type.Close:
            self._on_close()
        return bool(super().eventFilter(watched, event))


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
        # Per-tab last pop-out positions (issue #2881)
        self._popout_positions: dict[str, tuple[int, int]] = {}
        self._popout_windows: dict[str, QtWidgets.QMainWindow] = {}
        self._duplicate_counts: dict[str, int] = {}
        self._help_dialog: QtWidgets.QDialog | None = None
        self._settings_dialog: QtWidgets.QDialog | None = None
        self._workspace_list: QtWidgets.QListWidget | None = None
        self._workspace_table: QtWidgets.QWidget | None = None
        self._settings_button: QtWidgets.QToolButton | None = None
        self._project_root = Path(project_root or Path.cwd()).expanduser().resolve()
        self._layout_mode = _coerce_layout_mode(self._state.layout_mode)
        self._shutdown_complete = False
        self._host_close_filter: _HostCloseFilter | None = None

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

        # F4: Collaborators ─────────────────────────────────────────────────
        self._tab_collection = TabCollection(self.tabs)
        self._dock_chrome = DockChromeController(
            sidebar_widget=self,
            tabs_widget=self.tabs,
            initial_state=self._state,
        )
        self._vis_persistence = VisibilityPersistence(
            project_root=self._project_root,
        )

        # Compatibility shims: some mixins and tests access private lists
        # directly.  Keep them as live aliases into the collaborator.
        self._tab_ids: list[str] = self._tab_collection._tab_ids  # noqa: SLF001
        self._tab_widgets: dict[str, QtWidgets.QWidget] = (
            self._tab_collection._tab_widgets  # noqa: SLF001
        )
        self._tab_definitions: dict[str, SidebarTabDefinition] = (
            self._tab_collection._tab_definitions  # noqa: SLF001
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

    def shutdown(self) -> None:
        """Stop runtime resources owned by live sidebar tabs.

        The operation is idempotent so both a host launcher and Qt's close
        lifecycle may call it. Runtime widgets expose a small public
        ``shutdown()`` contract; passive tabs require no special handling.
        """
        if getattr(self, "_shutdown_complete", False):
            return
        self._shutdown_complete = True

        widgets = list(getattr(self, "_tab_widgets", {}).values())
        popout_windows = getattr(self, "_popout_windows", {})
        for window in popout_windows.values():
            central_widget = getattr(window, "centralWidget", None)
            if callable(central_widget):
                widgets.append(central_widget())

        seen: set[int] = set()
        for widget in widgets:
            if widget is None or id(widget) in seen:
                continue
            seen.add(id(widget))
            shutdown = getattr(widget, "shutdown", None)
            if not callable(shutdown):
                continue
            try:
                shutdown()
            except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
                _logger.debug("Sidekick tab shutdown failed: %s", exc)

    def closeEvent(self, event: object) -> None:  # noqa: N802 - Qt API
        """Shut down runtime tabs before the sidebar closes."""
        self.shutdown()
        super().closeEvent(event)  # type: ignore[misc]

    @property
    def dock_widget(self) -> QtWidgets.QDockWidget | None:
        """Return the installed dock widget, if any."""
        return self._dock_chrome.dock_widget

    @property
    def dock(self) -> QtWidgets.QDockWidget | None:
        """Alias for :attr:`dock_widget` — preferred by UI tests."""
        return self._dock_chrome.dock_widget

    @property
    def project_root(self) -> Path:
        """Return the current project root used by runtime tabs."""
        return self._project_root

    # ── Dock chrome helpers (delegated to DockChromeController) ───────────────

    def dock_title_widget(self) -> SidekickDockTitleBar | None:
        """Return the custom title-bar widget installed on the dock.

        Returns ``None`` if the dock has not been installed yet (i.e.
        :meth:`install_as_dock` has not been called).
        """
        return self._dock_chrome.title_bar

    def is_collapsed(self) -> bool:
        """Return ``True`` when the sidebar is in collapsed (icon-strip) mode."""
        return self._dock_chrome.is_collapsed

    def toggle_collapsed(self) -> None:
        """Toggle between collapsed (icon-strip) and expanded states."""
        self._dock_chrome.toggle_collapsed()
        self._emit_context()

    def _apply_expanded_width(self) -> None:
        self._dock_chrome._apply_expanded_width()  # noqa: SLF001

    def toggle_visibility(self) -> None:
        """Toggle the dock's visibility (Ctrl+B shortcut handler)."""
        self._dock_chrome.toggle_visibility()

    def register_shortcuts(self, main_window: QtWidgets.QMainWindow) -> None:
        """Wire Ctrl+B and Ctrl+Shift+B shortcuts onto ``main_window``."""
        self._dock_chrome.register_shortcuts(
            main_window,
            on_toggle_visibility=self.toggle_visibility,
            on_toggle_collapsed=self.toggle_collapsed,
        )

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
        return bool(self.redock_tab(tab_id))

    def add_tab(self, tab_id: str, title: str, widget: QtWidgets.QWidget) -> None:
        """Add a tab with a stable persistence id."""
        self._tab_collection.add(tab_id, title, widget)

    def replace_tab_widget(
        self,
        old_widget: QtWidgets.QWidget,
        new_widget: QtWidgets.QWidget,
    ) -> bool:
        """Atomically swap ``old_widget`` for ``new_widget`` inside the tab bar."""
        return self._tab_collection.replace(old_widget, new_widget)

    def configure_tabs(self, definitions: list[SidebarTabDefinition]) -> None:
        """Configure the available tab set for this Sidekick instance."""
        self._tab_collection.clear()
        self._tab_collection.set_definitions(definitions)
        self._configure_tab_settings()

        # Load visibility overrides from VisibilityPersistence (F4/F5 delegate)
        saved_visible = self._vis_persistence.load(
            known_ids=set(self._tab_collection.all_ids())
        )
        if saved_visible is not None:
            visible_defaults = set(saved_visible)
        else:
            visible_defaults = initially_visible_tab_ids(definitions, self._state)

        if not visible_defaults:
            visible_defaults = initially_visible_tab_ids(definitions, self._state)

        for definition in definitions:
            if definition.tab_id in visible_defaults:
                self._add_defined_tab(definition)

    def available_tab_ids(self) -> list[str]:
        """Return all configured tab ids, including hidden tabs."""
        return self._tab_collection.all_ids()

    def visible_tab_ids(self) -> list[str]:
        """Return tab ids currently docked in the sidebar."""
        return self._tab_collection.visible_ids()

    def hidden_tab_ids(self) -> list[str]:
        """Return configured tabs that are currently hidden."""
        return self._tab_collection.hidden_ids(popout_ids=set(self._popout_windows))

    def get_tab_definition(self, tab_id: str) -> SidebarTabDefinition | None:
        """Return the definition for ``tab_id``, or *None* if not found."""
        return self._tab_collection.definition_for(tab_id)

    def get_tab_id_at(self, index: int) -> str | None:
        """Return the stable tab id at visual ``index``, or *None* if out of range."""
        return self._tab_collection.id_at(index)

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

    def tab_widget(self, tab_id: str) -> QtWidgets.QWidget | None:
        """Return the live widget for ``tab_id``, or *None* if not built."""
        return self._tab_collection.widget_for(tab_id)

    def chat_dock_widget(self) -> QtWidgets.QWidget | None:
        """Return the live Chat tab widget, or *None* if not built."""
        return self.tab_widget("chat")

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
            if self._tab_collection.contains(tab_id):
                return True
            definition = self._tab_collection.definition_for(tab_id)
            if definition is None:
                return False
            self._add_defined_tab(definition)
            self._sync_tab_order_from_widget()
            self._emit_context()
            self._persist_visible_tabs()
            return True

        if not self._tab_collection.contains(tab_id):
            return tab_id in self._tab_collection.all_ids()
        if len(self._tab_collection.visible_ids()) == 1:
            return False
        self._tab_collection.remove(tab_id)
        self._emit_context()
        self._persist_visible_tabs()
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
        self._dock_chrome.set_minimized(
            minimized=minimized,
            expanded_width=self._state.width,
        )
        self._emit_context()

    def set_dock_area(self, area: str) -> bool:
        """Move the installed dock widget to the left or right side."""
        result = self._dock_chrome.set_dock_area(area, self._state)
        if result:
            self._emit_context()
        return result

    def install_as_dock(
        self,
        main_window: QtWidgets.QMainWindow,
        *,
        area: str | None = None,
        title: str = "Tools",
        state_path: str | Path | None = None,
    ) -> QtWidgets.QDockWidget:
        """Install this sidebar into ``main_window`` as a QDockWidget."""
        if state_path is not None:
            self.apply_state(SidebarState.load_json(state_path))

        dock = self._dock_chrome.install_as_dock(
            main_window=main_window,
            state=self._state,
            area=area,
            title=title,
            on_collapse=self.toggle_collapsed,
        )
        # Keep backward-compatible dock-object-name constant.
        dock.setObjectName(SIDEKICK_DOCK_OBJECT_NAME)
        self._host_close_filter = _HostCloseFilter(self.shutdown, main_window)
        main_window.installEventFilter(self._host_close_filter)
        self._emit_context()
        return dock

    def snapshot_state(self) -> SidebarState:
        """Return current dock, size, and active-tab state."""
        dock_area_name, floating, width, height = self._dock_chrome.snapshot(
            self._state
        )

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
        _dw = self._dock_chrome.dock_widget
        if _dw is not None:
            _dw.setFloating(state.floating)
            _dw.resize(state.width, state.height)

    def active_tab_id(self) -> str:
        index = int(self.tabs.currentIndex())
        visible = self._tab_collection.visible_ids()
        if 0 <= index < len(visible):
            return visible[index]
        return visible[0] if visible else ""

    def set_active_tab(self, tab_id: str) -> bool:
        idx = self._tab_collection.index_of(tab_id)
        if idx < 0:
            return False
        self.tabs.setCurrentIndex(idx)
        return True

    def open_tab(self, tab_id: str) -> bool:
        """Show and focus a configured tab by stable launcher-facing id."""
        resolved = {"os_terminal": "terminal"}.get(tab_id, tab_id)
        if resolved not in self._tab_collection.all_ids():
            return False
        if not self._tab_collection.contains(resolved) and not self.set_tab_visible(
            resolved, True
        ):
            return False
        return self.set_active_tab(resolved)

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
        self._vis_persistence = VisibilityPersistence(
            project_root=self._project_root,
        )
        set_project_explorer_root(
            self._tab_collection.widget_for("files"), self._project_root
        )
        self._emit_context()

    def _default_tab_definitions(self) -> list[SidebarTabDefinition]:
        # Annotated local rather than cast(): keeps mypy happy whether
        # default_tabs is analysed (cast would be redundant) or skipped
        # via --follow-imports=skip (the annotation absorbs the Any return).
        definitions: list[SidebarTabDefinition] = build_default_tab_definitions(
            self, SidebarTabDefinition
        )
        return definitions

    def _add_defined_tab(self, definition: SidebarTabDefinition) -> None:
        widget = definition.factory(self)
        display_name = self._tab_display_name(definition.tab_id, definition.title)
        self._tab_collection.add(definition.tab_id, display_name, widget)

    def _show_tab_context_menu(self, pos: QtCore.QPoint) -> None:
        show_tab_context_menu(self, pos)

    def tab_help_metadata(self, tab_id: str) -> dict[str, str]:
        """Return a copy of the configured help metadata for one tab id."""
        definition = self._tab_definitions.get(tab_id)
        if definition is None:
            return {}
        return dict(definition.help_metadata)

    def show_tab_help(self, tab_id: str | None = None) -> bool:
        """Open a compact dialog for one tab's help metadata.

        If a help dialog is already open, raises/activates it instead of
        creating a duplicate (F7 — single help dialog instance).
        """
        resolved_tab_id = tab_id or self.active_tab_id()
        metadata = self.tab_help_metadata(resolved_tab_id)
        if not metadata:
            return False

        # Re-use existing dialog if still alive (F7).
        if self._help_dialog is not None and self._help_dialog.isVisible():
            self._help_dialog.raise_()
            self._help_dialog.activateWindow()
            return True

        from shared.python.ui import HoverCopyTextBrowser

        dialog = QtWidgets.QDialog(self)
        dialog.setObjectName(f"SidekickTabHelpDialog_{resolved_tab_id}")
        dialog.setWindowTitle(f"{self.tab_display_name(resolved_tab_id)} Help")
        dialog.resize(480, 360)
        layout = QtWidgets.QVBoxLayout(dialog)
        browser = HoverCopyTextBrowser(dialog)
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
        self._tab_collection.sync_order_from_widget()
        self._refresh_settings_button()
        self._emit_context()
        self._persist_visible_tabs()

    def _persist_visible_tabs(self) -> None:
        """Write the current visible-tab list to VisibilityPersistence (F4/F5)."""
        self._vis_persistence.save(self._tab_collection.visible_ids())

    def _apply_tab_state(self, state: SidebarState) -> None:
        self._state = sanitize_tab_state(state, self._tab_collection._tab_definitions)  # noqa: SLF001
        state = self._state
        for tab_id in list(self._tab_collection.visible_ids()):
            if tab_id in state.hidden_tabs:
                self.set_tab_visible(tab_id, False)
        for tab_id in state.tab_order:
            all_ids = self._tab_collection.all_ids()
            if tab_id in all_ids and not self._tab_collection.contains(tab_id):
                self.set_tab_visible(tab_id, True)
        for index, tab_id in enumerate(state.tab_order):
            self.move_tab(tab_id, index)
        for tab_id in self._tab_collection.all_ids():
            self._refresh_tab_display_name(tab_id)


SidekickSidebar = UnifiedToolsSidebar


def _coerce_layout_mode(value: LayoutMode | str | None) -> LayoutMode:
    """Coerce ``value`` to a :class:`LayoutMode`, defaulting to SIDEBAR."""
    if value is None:
        return LayoutMode.SIDEBAR  # type: ignore[return-value]
    if isinstance(value, LayoutMode):
        return value
    if not isinstance(value, str):
        raise TypeError("layout mode must be a LayoutMode or str")
    try:
        return LayoutMode(value)
    except ValueError as exc:
        raise ValueError(f"unknown layout mode: {value!r}") from exc
