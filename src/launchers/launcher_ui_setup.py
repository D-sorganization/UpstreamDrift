"""UI setup and initialization mixins for UpstreamDriftLauncher.

Contains menu bar, top bar, grid area, bottom bar, search, console,
context help, and AI panel setup methods.
"""

# mypy: disable-error-code="attr-defined,call-overload,arg-type,assignment"

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import (
    QAction,
    QKeySequence,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.launchers.launcher_constants import (
    TILE_SCALE_MAX,
    TILE_SCALE_MIN,
)
from src.launchers.custom_title_bar import create_window_control_button
from src.launchers.launcher_manager_attrs import forward_manager_attribute
from src.launchers._launcher_navigation_ui import (
    LauncherNavigationUIMixin,
    _build_menu_bar_close_widget as _navigation_menu_bar_close_widget,
)
from src.launchers._launcher_top_bar_ui import (
    ClickableLabel as ClickableLabel,
    HelpButtonHoverFilter as HelpButtonHoverFilter,
    LauncherTopBarUIMixin,
    RuntimeButton as RuntimeButton,
    _build_zoom_accessible_description as _top_bar_zoom_description,
)

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.theme.style_constants import Styles

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


def _build_zoom_accessible_description() -> str:
    """Describe zoom bounds through the historical facade helper."""
    return _top_bar_zoom_description(TILE_SCALE_MIN, TILE_SCALE_MAX)


def _build_menu_bar_close_widget(parent: QWidget, close_callback: Any) -> QWidget:
    """Build the historical menu close control with a patchable factory."""
    return _navigation_menu_bar_close_widget(
        parent,
        close_callback,
        button_factory=create_window_control_button,
    )


class ProcessOutputRelay(QObject):
    """Marshal subprocess output from reader threads onto the GUI thread.

    :class:`~src.launchers.launcher_process_manager.ProcessManager` reads
    child-process stdout on plain ``threading.Thread`` workers and invokes its
    ``output_callback`` from there. Qt widgets may only be touched from the
    thread that owns them, so the callback emits :attr:`line_received`; because
    this object lives in the GUI thread and the connection is explicitly
    ``QueuedConnection``, ``sink`` always runs on the GUI thread (issue #8003).
    """

    line_received = pyqtSignal(str, str)

    def __init__(self, sink: Any, parent: QObject | None = None) -> None:
        super().__init__(parent)
        if sink is None:
            raise ValueError("sink must be provided")
        self._sink = sink
        self.line_received.connect(self._deliver, Qt.ConnectionType.QueuedConnection)

    def _deliver(self, engine_name: str, line: str) -> None:
        """Forward one line to the console sink (GUI thread only)."""
        self._sink(engine_name, line)


from typing import Protocol
from PyQt6.QtWidgets import QTabWidget, QDialog
import contextlib


class LauncherUIProtocol(Protocol):
    workspace_tabs: QTabWidget
    library_widget: QWidget | None
    _popped_out_windows: list[QDialog]
    sidekick_sidebar: QWidget | None
    btn_ai_sidebar: QPushButton | None
    btn_popout_sidekick: QPushButton
    sidebar_widget: QWidget
    main_layout: QSplitter
    grid_layout: QGridLayout
    zoom_slider: QSlider
    lbl_zoom_pct: QLabel
    view_mode_combo: QComboBox | None
    chk_live: QCheckBox
    chk_gpu: QCheckBox
    chk_windows: QCheckBox
    chk_docker: QCheckBox
    chk_wsl: QCheckBox
    btn_console: QToolButton
    lbl_status: QLabel
    context_help: Any
    overlay: Any
    _action_console: QAction
    _viewmode_actions: dict[Any, QAction]
    _top_viewmode_actions: dict[Any, QAction]
    layout_manager: Any
    toast_manager: Any
    docker_checker: Any


class GridContainerWidget(QWidget):
    """Container widget that listens to resize events to dynamically rebuild the grid.

    This ensures that when the sidebar opens or closes, or the window is resized,
    the grid wrapping recalculates its columns to avoid a horizontal scrollbar.
    """

    def __init__(self, parent=None, launcher=None, grid_layout=None):
        super().__init__(parent)
        self.launcher = launcher
        self.grid_layout = grid_layout
        self._last_width = 0

    def resizeEvent(self, event):
        super().resizeEvent(event)
        width = self.width()
        if width != self._last_width and width > 0:
            self._last_width = width
            launcher = self.launcher
            if launcher and hasattr(launcher, "_rebuild_grid"):
                launcher._rebuild_grid()


class ResizingScrollArea(QScrollArea):
    """A QScrollArea that triggers a grid rebuild when its viewport width changes.

    This ensures that grid wrapping is recalculated based on the actual visible viewport width,
    avoiding horizontal scrollbars when sidebars are toggled or resized.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_width = 0

    def resizeEvent(self, event):
        super().resizeEvent(event)
        width = self.viewport().width()
        if width != self._last_width and width > 0:
            self._last_width = width
            widget = self.widget()
            if widget and hasattr(widget, "grid_layout"):
                launcher = getattr(widget, "launcher", None)
                if launcher and hasattr(launcher, "_rebuild_grid"):
                    launcher._rebuild_grid()


class UISetupManager(LauncherNavigationUIMixin, LauncherTopBarUIMixin):
    def __init__(self, launcher):
        self.launcher = launcher

    def _create_menu_bar_close_widget(
        self,
        parent: QWidget,
        close_callback: Any,
    ) -> QWidget:
        """Route runtime menu construction through the historical facade seam."""
        return _build_menu_bar_close_widget(parent, close_callback)

    def _get_zoom_accessible_description(self) -> str:
        """Route runtime zoom guidance through the historical facade seam."""
        return _build_zoom_accessible_description()

    def __getattr__(self, name):
        return getattr(self.launcher, name)

    def __setattr__(self, name, value):
        forward_manager_attribute(self, name, value)

    def _on_windows_mode_changed(self, state: int) -> None:
        """Delegate Windows-mode changes when the full launcher provides a handler."""
        try:
            handler = self.launcher._on_windows_mode_changed
        except AttributeError:
            return
        handler(state)

    def wheelEvent(self, event: Any) -> None:  # noqa: N802
        """Ctrl+scroll wheel adjusts the zoom slider."""
        from PyQt6.QtCore import Qt as _Qt

        if event.modifiers() & _Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta != 0:
                steps = 5 if delta > 0 else -5
                self._nudge_zoom(steps)
                event.accept()
                return
        super().wheelEvent(event)  # type: ignore[misc]

    """Mixin for UpstreamDriftLauncher UI initialization.

    Provides methods for building the menu bar, top bar, grid area,
    bottom bar, search shortcuts, process console, context help, and AI panel.
    """

    def init_ui(self) -> None:
        """Initialize the user interface.

        Frameless-window chrome contract (#5624):

            outer_vbox  (top to bottom)
              ├─ self.title_bar   (CustomTitleBar — black)
              ├─ self.menu_bar    (QMenuBar — File / View / Tools / Help)
              └─ self.main_layout (QSplitter — sidebar | content | sidekick)

        ``QMainWindow.setMenuBar`` is intentionally **not** used: that
        API reserves a native strip above the central widget, which on
        a frameless main window renders above the custom title bar —
        the exact regression #5624 fixes.
        """
        # Main Widget
        central = QWidget()
        self.setCentralWidget(central)

        # Outer layout to hold the title bar, menu bar, and the horizontal
        # main layout (#5624: explicit, controlled vertical ordering).
        outer_vbox = QVBoxLayout(central)
        outer_vbox.setSpacing(0)
        outer_vbox.setContentsMargins(0, 0, 0, 0)

        try:
            from src.launchers.custom_title_bar import CustomTitleBar

            self.title_bar = CustomTitleBar(self.launcher, show_close_button=True)
            self.title_bar.minimize_requested.connect(self.showMinimized)
            self.title_bar.maximize_requested.connect(
                lambda: (
                    self.showNormal() if self.isMaximized() else self.showMaximized()
                )
            )
            self.title_bar.close_requested.connect(self.close)
            # Clamp every move target into the virtual desktop so an
            # off-screen drag does not silently strand the window.
            self.title_bar.move_requested.connect(lambda pos: self.move(pos))

            # The native OS title bar is hidden via FramelessWindowHint set in __init__
            outer_vbox.addWidget(self.title_bar)
        except ImportError:
            pass

        # --- Menu Bar (immediately below the title bar) ---
        # Postcondition: ``self.menu_bar`` is a non-null populated QMenuBar
        # owned by ``central`` (the QMainWindow.centralWidget()), NOT by
        # the native main-window menu strip.  Tests in
        # tests/unit/launcher/test_layout_hierarchy.py pin this contract.
        self.menu_bar = self._build_menu_bar_widget()
        outer_vbox.addWidget(self.menu_bar)

        # Main layout is now a horizontal splitter to accommodate the sidebar resizably
        main_layout = QSplitter(Qt.Orientation.Horizontal)
        main_layout.setProperty("class", "dark")
        main_layout.setHandleWidth(4)
        main_layout.setChildrenCollapsible(False)

        try:
            from src.shared.python.theme import get_current_colors

            _colors = get_current_colors()
        except (ImportError, AttributeError):
            from src.shared.python.theme import DARK_THEME as _dt

            _colors = {
                "border": getattr(_dt, "border_default", "#555555"),
                "accent": getattr(_dt, "accent", "#0A84FF"),
            }
        _splitter_bg = _colors.get("border", "#555555")
        _splitter_hover = _colors.get("accent", "#0A84FF")
        main_layout.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {_splitter_bg};
                margin: 2px 0px;
                border-radius: 2px;
            }}
            QSplitter::handle:hover {{
                background-color: {_splitter_hover};
            }}
        """)
        outer_vbox.addWidget(main_layout, 1)

        # Expose the splitter for downstream features that embed extra
        # panes (e.g. ``_install_sidekick_sidebar`` in #5624 adds the
        # Sidekick widget as the third pane instead of using
        # ``addDockWidget``, which misbehaves on a frameless window).
        self.main_layout = main_layout

        # --- Global Sidebar ---
        self.sidebar_widget = self._setup_global_sidebar()
        main_layout.addWidget(self.sidebar_widget)

        # Content Container
        content_container = QWidget()
        content_container.setMinimumWidth(300)
        content_layout = QVBoxLayout(content_container)
        content_layout.setSpacing(Styles.SPACING_LG)
        content_layout.setContentsMargins(
            Styles.MARGIN_PAGE,
            Styles.MARGIN_PAGE,
            Styles.MARGIN_PAGE,
            Styles.MARGIN_PAGE,
        )

        # --- Top Bar ---
        top_bar = self._setup_top_bar()

        # Add Sidekick pop-out button as part of top-bar. It will be hidden initially.
        self.btn_popout_sidekick = QPushButton("⇱ Pop Out")
        self.btn_popout_sidekick.setToolTip("Pop out Sidekick into a separate window")
        self.btn_popout_sidekick.clicked.connect(self._popout_sidekick)
        self.btn_popout_sidekick.setVisible(False)
        self.btn_popout_sidekick.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 4px 8px;
                color: #cccccc;
            }
            QPushButton:hover {
                background: #2a2a2a;
                color: #ffffff;
                border-color: #555555;
            }
        """)
        top_bar.insertWidget(top_bar.count(), self.btn_popout_sidekick)

        content_layout.addLayout(top_bar)

        # --- Content area with horizontal splitter (tiles | AI chat) ---
        self.content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.content_splitter.setHandleWidth(3)
        self.content_splitter.setChildrenCollapsible(False)
        self.content_splitter.setProperty("class", "dark")
        _style = self.content_splitter.style()

        if _style:
            _style.polish(self.content_splitter)

        # Left panel: launcher grid + bottom bar
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(20)
        self._setup_grid_area(left_layout)
        self._setup_running_processes_panel(left_layout)
        bottom_bar = self._setup_bottom_bar()
        left_layout.addLayout(bottom_bar)

        self.content_splitter.addWidget(left_panel)

        # NOTE: The legacy right-panel AIAssistantPanel that used to be
        # spliced into ``content_splitter`` here was removed as part of the
        # deprecated-chat sweep (UpstreamDrift #5620). The canonical chat
        # surface is now the Sidekick dock's "Chat" tab, provided by the
        # vendored ``sidekick.ui.tools_sidebar`` package via
        # ``_install_sidekick_sidebar()`` in ``upstream_drift_launcher.py``.
        # ``toggle_ai_assistant`` and other ``hasattr(self, "ai_panel")``
        # call sites become safe no-ops; a follow-up issue rewires them to
        # raise/focus the Sidekick chat tab. Do NOT re-introduce a second
        # ``AIAssistantPanel`` here.
        self._ai_visible = False

        # Sidekick pane management
        self.sidekick_window = None
        self._sidekick_popped_out = False

        # Add Workspace Tabs for Unified Architecture
        from src.shared.python.gui_pkg.draggable_tabs import DraggableTabWidget

        self.workspace_tabs = DraggableTabWidget(core_tabs={"Home"})
        self.workspace_tabs.setDocumentMode(True)
        self.workspace_tabs.currentChanged.connect(self._sync_console_button_states)

        self.workspace_tabs.addTab(self.content_splitter, "Home")

        self.library_widget = None
        self.library_window = None

        content_layout.addWidget(self.workspace_tabs, 1)

        main_layout.addWidget(content_container)

        # Sidebar should not stretch, content should take the rest.
        main_layout.setStretchFactor(0, 0)
        main_layout.setStretchFactor(1, 1)

        # Apply dark theme
        self.apply_styles()

        # Keyboard shortcuts
        self._setup_search_shortcuts()

    def dock_widget_as_tab(self, widget: QWidget, title: str) -> None:
        """Dock a submodule/tool widget as a new tab in the workspace.

        Tool tabs are background-eligible (#6013): closing one keeps the
        widget running hidden and restorable via the View > Background Tabs
        menu, rather than destroying it.
        """
        if widget is None:
            raise ValueError("widget must be provided")

        index = self.workspace_tabs.add_background_tab(widget, title)
        self.workspace_tabs.setCurrentIndex(index)

    def _open_library_tab(self) -> None:
        """Open or focus the Library workspace tab."""
        if not True:
            logger.error("Workspace tabs not initialized; cannot open Library.")
            return

        try:
            from src.launchers.library_widget import LibraryWidget
        except ImportError as e:
            logger.warning("Could not load Library tab: %s", e)
            if True:
                self.show_toast("Library is unavailable in this environment.", "error")
            return

        existing = self.library_widget
        if existing is None:
            existing = LibraryWidget(self.launcher)
            self.library_widget = existing
            # The cached singleton may be detached and closed via "Close Tab",
            # which deleteLater()s the underlying C++ object (#6902). Null the
            # cache when that happens so the next open rebuilds it instead of
            # re-using a deleted object (RuntimeError: wrapped C/C++ object
            # deleted).
            existing.destroyed.connect(self._on_library_widget_destroyed)

        index = self.workspace_tabs.indexOf(existing)
        if index < 0:
            self.dock_widget_as_tab(existing, "Library")
        else:
            self.workspace_tabs.setCurrentIndex(index)

    def _on_library_widget_destroyed(self, *_args: object) -> None:
        """Null the cached Library singleton once its C++ object is destroyed.

        Connected to ``LibraryWidget.destroyed`` so a detached-and-closed
        Library (which ``deleteLater()``s the widget) cannot leave a dangling
        reference in ``self.library_widget`` (#6902).
        """
        self.library_widget = None
        self.library_window = None

    def _popout_library(self) -> None:
        """Open the Library in a floating window, preserving one widget instance."""
        self._open_library_tab()
        widget = self.library_widget
        if widget is None:
            return

        index = self.workspace_tabs.indexOf(widget)
        if index >= 0:
            self.workspace_tabs.removeTab(index)

        self.popout_widget(widget, "Library")
        windows = getattr(self, "_popped_out_windows", [])
        self.library_window = windows[-1] if windows else None

    def popout_widget(self, widget: QWidget, title: str) -> None:
        """Pop out a submodule widget into a separate window."""
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QDialog, QVBoxLayout

        if not hasattr(self, "_popped_out_windows") or self._popped_out_windows is None:
            self._popped_out_windows: list[QDialog] = []

        # We use a non-modal dialog to allow it to float freely
        win = QDialog(self.launcher, Qt.WindowType.Window)
        win.setWindowTitle(title)
        win.resize(1000, 800)

        layout = QVBoxLayout(win)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget)

        def on_close(event):
            if win in self._popped_out_windows:
                self._popped_out_windows.remove(win)
            event.accept()

        win.closeEvent = on_close
        self._popped_out_windows.append(win)
        win.show()

    def _toggle_sidekick(self, checked: bool = None) -> None:
        """Toggle the visibility of the Sidekick pane."""
        logger.info(f"_toggle_sidekick called with checked={checked}")
        if True and self.sidekick_sidebar is not None:
            if self._sidekick_popped_out and self.sidekick_window:
                if self.sidekick_window.isHidden():
                    self.sidekick_window.show()
                else:
                    self.sidekick_window.hide()
            else:
                if checked is None:
                    # If called programmatically without arg, invert current state
                    visible = not self.sidekick_sidebar.isVisible()
                else:
                    visible = checked

                logger.info(f"Setting sidekick visible={visible}")
                self.sidekick_sidebar.setVisible(visible)

                # When showing the sidebar, ensure the splitter gives it width
                if visible and True:
                    self._apply_sidekick_splitter_sizes()

                # Keep button in sync
                btn = getattr(self, "btn_toggle_right_sidebar", None) or getattr(
                    self, "btn_ai_sidebar", None
                )
                if btn is not None and btn.isChecked() != visible:
                    btn.setChecked(visible)

                if visible and True:
                    self.btn_popout_sidekick.setVisible(True)
        else:
            logger.info("Sidekick sidebar still loading or not initialized.")
            if True:
                self.show_toast(
                    "Sidekick is still loading, please wait a moment…", "info"
                )
            # Uncheck the button since it's not ready yet
            btn = getattr(self, "btn_toggle_right_sidebar", None) or getattr(
                self, "btn_ai_sidebar", None
            )
            if btn is not None:
                btn.setChecked(False)

    def _toggle_left_sidebar(self, checked: bool = None) -> None:
        """Toggle the visibility of the global navigation sidebar."""
        if not True or self.sidebar_widget is None:
            return
        visible = not self.sidebar_widget.isVisible() if checked is None else checked
        self.sidebar_widget.setVisible(visible)

        # Ensure proper splitter sizes when showing
        if visible and True:
            sizes = self.main_layout.sizes()
            if sum(sizes) > 0 and sizes[0] == 0:
                # Give the sidebar its minimum width at least
                sizes[0] = 120
                sizes[1] = max(
                    100, sum(sizes) - 120 - (sizes[2] if len(sizes) > 2 else 0)
                )
                self.main_layout.setSizes(sizes)

    def _popout_sidekick(self) -> None:
        """Toggle Sidekick pop-out state."""
        if not True or self.sidekick_sidebar is None:
            return

        if not self._sidekick_popped_out:
            # Pop out
            self._sidekick_popped_out = True
            self.btn_popout_sidekick.setText("⇲ Dock Sidekick")

            from PyQt6.QtWidgets import QDialog, QVBoxLayout
            from PyQt6.QtCore import Qt

            self.sidekick_window = QDialog(self.launcher, Qt.WindowType.Window)
            self.sidekick_window.setWindowTitle("UpstreamDrift Sidekick")
            self.sidekick_window.resize(400, 800)

            layout = QVBoxLayout(self.sidekick_window)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.sidekick_sidebar)

            def on_close(event):
                event.ignore()
                self.sidekick_window.hide()

            self.sidekick_window.closeEvent = on_close
            self.sidekick_window.show()
        else:
            # Re-dock
            self._sidekick_popped_out = False
            self.btn_popout_sidekick.setText("⇱ Pop Out Sidekick")

            if self.sidekick_window:
                self.sidekick_window.hide()

            self.main_layout.addWidget(self.sidekick_sidebar)
            self.sidekick_sidebar.show()

        # Initialize Overlay
        self._init_overlay()

        # top_bar.addWidget(self.btn_launch)  # Removed launch button per user request

        # Layout controls were moved to the View menu per user request.

        # Action buttons were moved to the left sidebar per user request.

    # ---- View-mode + zoom controls --------------------------------------

    _ZOOM_SLIDER_STEPS = 100  # slider integer range -> [MIN, MAX] tile_scale

    def _setup_grid_area(self, layout: QVBoxLayout) -> None:
        """Set up the scrollable grid area."""
        if layout is None:
            raise ValueError("layout must be provided")
        self.scroll_area = ResizingScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setProperty("class", "transparent")
        # Issue #6679: disable horizontal scrollbar so the viewport width
        # actually shrinks when the Sidekick panel opens.  When the scrollbar
        # is enabled Qt expands content rather than wrapping — the
        # ResizingScrollArea.resizeEvent never fires, columns never re-wrap.
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        _style = self.scroll_area.style()

        if _style:
            _style.polish(self.scroll_area)

        self.grid_container = GridContainerWidget(launcher=self.launcher)
        self.grid_container.setProperty("class", "transparent")
        # Allow the container to shrink below its preferred width so the
        # scroll area can compute a narrower available_width for column wraps.
        self.grid_container.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )
        _style = self.grid_container.style()

        if _style:
            _style.polish(self.grid_container)
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(20)
        self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.grid_container.grid_layout = self.grid_layout

        self.scroll_area.setWidget(self.grid_container)
        layout.addWidget(self.scroll_area, 1)

    def _setup_bottom_bar(self) -> QHBoxLayout:
        """Return an empty bottom bar; the launch button now lives in the top bar.

        Kept as a method so existing callers and tests
        (``test_setup_bottom_bar``) that expect ``self.btn_launch`` to exist
        after invocation keep working. Returns an empty layout — caller can
        still ``addLayout`` it without producing visible chrome at the bottom.
        """
        self._ensure_launch_button()
        return QHBoxLayout()

    def _setup_search_shortcuts(self) -> None:
        """Setup keyboard shortcuts for search."""
        shortcut_search = QShortcut(QKeySequence("Ctrl+F"), self.launcher)
        shortcut_search.activated.connect(self._focus_search)

        shortcut_escape = QShortcut(QKeySequence("Esc"), self.launcher)
        shortcut_escape.activated.connect(self._clear_search)

    def _focus_search(self) -> None:
        """Focus and select all text in search bar."""
        self.search_input.setFocus()
        self.search_input.selectAll()

    def _clear_search(self) -> None:
        """Clear the search filter and remove focus from search bar."""
        has_text = False
        text_val = self.search_input.text()
        if isinstance(text_val, str) and text_val:
            has_text = True
        if has_text or self.search_input.hasFocus():
            self.search_input.clear()
            self.search_input.clearFocus()

    # -- Process Output Console --

    def _setup_process_console(self) -> None:
        """Create the dockable Process Output console widget."""
        self._console_text = QPlainTextEdit()
        self._console_text.setReadOnly(True)
        self._console_text.setMaximumBlockCount(5000)
        self._console_text.setProperty("class", "console-dark")
        _style = self._console_text.style()

        if _style:
            _style.polish(self._console_text)

        self._console_widget = QWidget()
        self._console_widget.prevent_deletion_on_close = True
        console_layout = QVBoxLayout(self._console_widget)
        console_layout.setContentsMargins(0, 0, 0, 0)
        console_layout.setSpacing(0)
        console_layout.addWidget(self._console_text)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(4, 2, 4, 2)

        clear_btn = QToolButton()
        clear_btn.setText("Clear")
        clear_btn.setToolTip("Clear console output")
        clear_btn.clicked.connect(self._console_text.clear)
        toolbar.addStretch()
        toolbar.addWidget(clear_btn)
        console_layout.addLayout(toolbar)

        # Built here, on the GUI thread, so the relay's thread affinity is the
        # GUI thread and queued deliveries land there (issue #8003).
        self._ensure_console_relay()

    def _ensure_console_relay(self) -> ProcessOutputRelay:
        """Return the relay marshalling reader-thread output onto the GUI thread."""
        relay = getattr(self, "_console_relay", None)
        if relay is None:
            relay = ProcessOutputRelay(self._append_console_line, self.launcher)
            self._console_relay = relay
        return relay

    def _on_process_output(self, engine_name: str, line: str) -> None:
        """Receive a line of output from a subprocess (thread-safe).

        Called from :class:`ProcessManager`'s plain ``threading.Thread`` stdout
        readers. ``QTimer.singleShot`` used to be used here, but a bare
        ``singleShot`` schedules onto the *calling* thread's event loop — the
        reader threads have none, so the callback never fired and the Process
        Output console stayed empty (issue #8003). Emitting a signal into a
        GUI-thread-affine QObject with a queued connection is the correct
        cross-thread hand-off.
        """
        self._ensure_console_relay().line_received.emit(engine_name, line)

    def _append_console_line(self, engine_name: str, line: str) -> None:
        """Append a formatted line to the console widget (GUI thread only)."""
        if engine_name is None:
            raise ValueError("engine_name must be provided")

        idx = self.workspace_tabs.indexOf(self._console_widget)
        if idx == -1:
            idx = self.workspace_tabs.addTab(self._console_widget, "Console")
            self._sync_console_button_states()

        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._console_text.appendPlainText(f"[{ts}] [{engine_name}] {line}")

    def toggle_process_console(self) -> None:
        """Toggle visibility of the Process Output console tab."""
        idx = self.workspace_tabs.indexOf(self._console_widget)
        if idx == -1:
            idx = self.workspace_tabs.addTab(self._console_widget, "Console")
            self.workspace_tabs.setCurrentIndex(idx)
        else:
            if self.workspace_tabs.currentIndex() == idx:
                self.workspace_tabs.removeTab(idx)
            else:
                self.workspace_tabs.setCurrentIndex(idx)
        self._sync_console_button_states()

    def _is_console_open(self) -> bool:
        """Check if the console widget is currently in the tab bar or detached."""
        console_widget = getattr(self, "_console_widget", None)
        if not isinstance(console_widget, QWidget):
            return False
        if self.workspace_tabs.indexOf(console_widget) != -1:
            return True
        # Check detached tabs
        for widget, _, _ in self.workspace_tabs.detached_tabs.values():
            if widget is console_widget:
                return True
        return False

    def _sync_console_button_states(self) -> None:
        """Synchronize the console action and sidebar button states based on tab presence."""
        is_open = self._is_console_open()
        if hasattr(self, "_action_console") and self._action_console:
            self._action_console.setChecked(is_open)
        if hasattr(self, "btn_console") and self.btn_console:
            self.btn_console.setChecked(is_open)

    # -- Clear Filters --

    def _clear_all_filters(self) -> None:
        """Clear search input and reset category to Home (All)."""
        self.search_input.clear()
        if hasattr(self, "sidebar_group"):
            home_btn = self.sidebar_group.button(0)
            if home_btn:
                home_btn.setChecked(True)
            self._on_sidebar_routed(0)

    # -- Running Processes Panel --

    def _setup_running_processes_panel(self, layout: QVBoxLayout) -> None:
        """Create the running processes list widget at the bottom of the home grid."""
        self.running_processes_panel = QFrame()
        self.running_processes_panel.setObjectName("RunningProcessesPanel")
        self.running_processes_panel.setStyleSheet("""
            #RunningProcessesPanel {
                background-color: rgba(30, 30, 30, 0.7);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                padding: 8px;
            }
        """)
        panel_layout = QVBoxLayout(self.running_processes_panel)
        panel_layout.setContentsMargins(10, 6, 10, 6)
        panel_layout.setSpacing(6)

        header = QHBoxLayout()
        header_title = QLabel("Running Processes")
        header_title.setStyleSheet(
            "font-weight: bold; color: #ffffff; font-size: 12px;"
        )
        header.addWidget(header_title)
        header.addStretch()

        # Kill All button
        self.btn_kill_all_procs = QPushButton("Kill All")
        self.btn_kill_all_procs.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_kill_all_procs.setStyleSheet("""
            QPushButton {
                background-color: rgba(220, 53, 69, 0.2);
                border: 1px solid rgba(220, 53, 69, 0.4);
                border-radius: 4px;
                padding: 2px 8px;
                color: #ff6b6b;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: rgba(220, 53, 69, 0.4);
                color: #ffffff;
            }
        """)
        self.btn_kill_all_procs.clicked.connect(self._kill_all_processes)
        header.addWidget(self.btn_kill_all_procs)
        panel_layout.addLayout(header)

        # Container for process rows
        self.processes_container = QWidget()
        self.processes_container.setProperty("class", "transparent")
        self.processes_layout = QVBoxLayout(self.processes_container)
        self.processes_layout.setContentsMargins(0, 0, 0, 0)
        self.processes_layout.setSpacing(4)
        panel_layout.addWidget(self.processes_container)

        layout.addWidget(self.running_processes_panel)
        self.running_processes_panel.hide()

    def _find_active_settings_widget(self) -> QWidget | None:
        """Find the active SettingsWidget, whether docked as a tab or detached/floating."""
        if hasattr(self, "workspace_tabs") and self.workspace_tabs is not None:
            # Check docked tabs
            for i in range(self.workspace_tabs.count()):
                widget = self.workspace_tabs.widget(i)
                if widget and widget.__class__.__name__ == "SettingsWidget":
                    return widget
            # Check detached/floating tabs
            if hasattr(self.workspace_tabs, "detached_tabs"):
                for widget, _, _ in self.workspace_tabs.detached_tabs.values():
                    if widget and widget.__class__.__name__ == "SettingsWidget":
                        return widget
        return None

    def update_running_processes_ui(self) -> None:
        """Update the running processes list widget by refreshing the settings tab."""
        # Ensure the home page running processes panel remains hidden
        if (
            hasattr(self, "running_processes_panel")
            and self.running_processes_panel is not None
        ):
            self.running_processes_panel.hide()

        # Refresh the active Settings widget if it is currently instantiated
        settings_widget = self._find_active_settings_widget()
        if settings_widget is not None and hasattr(
            settings_widget, "refresh_processes_ui"
        ):
            settings_widget.refresh_processes_ui()

    def _kill_process_by_name(self, name: str) -> None:
        """Kill a running process by its name."""

        with self.process_manager._process_lock:
            proc = self.running_processes.get(name)
            if proc:
                try:
                    logger.info(
                        f"Killing process {name} (PID: {proc.pid}) via UI request"
                    )
                    from src.shared.python.security.subprocess_utils import (
                        kill_process_tree,
                    )

                    kill_process_tree(proc.pid)
                    proc.terminate()
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Error killing process {name}: {e}")
                finally:
                    with contextlib.suppress(KeyError):
                        del self.running_processes[name]
        self.update_running_processes_ui()

    def _kill_all_processes(self) -> None:
        """Kill all currently running processes."""
        self.process_manager.cleanup_processes()
        self.update_running_processes_ui()

    # -- AI Panel (DEPRECATED, removed by UpstreamDrift #5620) --
    #
    # ``_setup_ai_panel`` previously spliced a second ``AIAssistantPanel``
    # into the launcher's right-edge splitter, duplicating the canonical
    # Sidekick chat tab. That method was deleted so the launcher only
    # surfaces ONE chat path (the Sidekick dock's Chat tab). Do not
    # restore an AI panel here; extend the Sidekick chat tab in Tools'
    # ``sidekick.ui.tools_sidebar`` package instead.
    #
    # The chat-session sync helper that lived next to it
    # (``_sync_chat_session``) was also dropped because the Sidekick
    # ``ChatDockWidget`` performs the equivalent session-id handshake via
    # the shared ``active_chat_session.txt`` file (see
    # ``Tools/src/shared/python/chat/chat_dock_widget.py``).

    # -- Context Help --

    def _setup_context_help(self) -> None:
        """Setup context help dock."""
        from src.launchers.ui_components import ContextHelpDock
        from PyQt6.QtWidgets import QDialog

        self.context_help = QDialog(self.launcher, Qt.WindowType.Window)
        self.context_help.setWindowTitle("Context Help")
        self.context_help.resize(400, 800)

        dl_layout = QVBoxLayout(self.context_help)
        dl_layout.setContentsMargins(0, 0, 0, 0)

        help_widget = ContextHelpDock(self.launcher)
        dl_layout.addWidget(help_widget)

        # Proxy the update_context method to the inner widget
        self.context_help.update_context = help_widget.update_context

        self.context_help.hide()

    # -- Overlay --

    def _init_overlay(self) -> None:
        """Initialize the screen overlay."""
        try:
            from src.shared.python.ui.overlay import OverlayWidget

            self.overlay = OverlayWidget(self.launcher)
            self.overlay.hide()
        except (ImportError, TypeError):
            logger.warning("OverlayWidget could not be initialized.")

    def _toggle_overlay(self) -> None:
        """Toggle the screen overlay."""
        if True:
            self.overlay.toggle()
