"""UI setup and initialization mixins for UpstreamDriftLauncher.

Contains menu bar, top bar, grid area, bottom bar, search, console,
context help, and AI panel setup methods.
"""

# mypy: disable-error-code="attr-defined,call-overload,arg-type,assignment"

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import QSize, Qt, QTimer
from PyQt6.QtGui import (
    QAction,
    QKeySequence,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMenuBar,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.launchers.launcher_constants import (
    TILE_SCALE_MAX,
    TILE_SCALE_MIN,
    ViewMode,
)
from src.launchers.custom_title_bar import create_window_control_button

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.theme.style_constants import Styles
from src.shared.python.theme.typography import Weights, get_display_font
from src.shared.python.ui.auto_complete import AutoCompleteLineEdit
from src.shared.python.ui.completion_vocab import build_vocabulary

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


def _build_zoom_accessible_description() -> str:
    """Describe the zoom slider using the configured tile-scale bounds."""
    minimum_pct = int(round(TILE_SCALE_MIN * 100))
    maximum_pct = int(round(TILE_SCALE_MAX * 100))
    return (
        f"Adjust tile size from {minimum_pct}% to {maximum_pct}%. "
        "Use arrow keys or drag to adjust."
    )


def _build_menu_bar_close_widget(parent: QWidget, close_callback: Any) -> QWidget:
    """Create the top-row close control for the launcher menu bar."""
    container = QWidget(parent)
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 1, 6, 1)
    layout.setSpacing(0)

    close_button = create_window_control_button(
        "close",
        "X",
        tooltip="Close the launcher",
        accessible_name="Close launcher window",
        object_name="menu-bar-close-button",
        parent=container,
    )
    close_button.clicked.connect(lambda _checked=False: close_callback())
    layout.addWidget(close_button)
    return container


class LauncherUISetupMixin:
    """Mixin for UpstreamDriftLauncher UI initialization.

    Provides methods for building the menu bar, top bar, grid area,
    bottom bar, search shortcuts, process console, context help, and AI panel.
    """

    def _build_sidebar_button(
        self,
        label: str,
        icon_name: str,
        *,
        checkable: bool = False,
    ) -> QToolButton:
        """Create an icon-first sidebar control with accessible labeling.

        Provides both icon and visible text label for accessibility.
        """
        button = QToolButton()
        button.setText(label)
        button.setToolTip(label)
        button.setAccessibleName(label)
        button.setAccessibleDescription(f"Navigate to {label} section")
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        # Show both icon and text for better accessibility
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)

        # Sidebar buttons must always render a visible icon (#5624).
        # If the requested glyph is missing from SVG_REGISTRY, fall back
        # to a registered glyph so the icon is never null — null icons
        # produce the text-only sidebar regression visible in #5624.
        try:
            from src.shared.python.theme.icon_utils import (
                SVG_REGISTRY,
                IconColorizer,
            )

            resolved_name = icon_name if icon_name in SVG_REGISTRY else "settings"
            button.setIcon(IconColorizer.get_icon(resolved_name, "#d4d4d4"))
        except (ImportError, ValueError):
            # Last-resort fallback: any QIcon (even with a tinted blank
            # pixmap) is preferable to a null icon for hit-testing and
            # screen-reader semantics.
            from PyQt6.QtGui import QIcon, QPixmap

            pixmap = QPixmap(22, 22)
            pixmap.fill(Qt.GlobalColor.transparent)
            button.setIcon(QIcon(pixmap))

        button.setIconSize(QSize(22, 22))
        button.setCheckable(checkable)
        button.setAutoRaise(True)
        # Set focus policy for keyboard navigation
        button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return button

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

            self.title_bar = CustomTitleBar(self, show_close_button=True)
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
        from PyQt6.QtWidgets import QTabWidget

        self.workspace_tabs = QTabWidget()
        self.workspace_tabs.setTabsClosable(True)
        self.workspace_tabs.setDocumentMode(True)

        def _on_tab_close_requested(index: int) -> None:
            if index > 0:
                widget = self.workspace_tabs.widget(index)
                self.workspace_tabs.removeTab(index)
                if widget is not None:
                    widget.deleteLater()

        self.workspace_tabs.tabCloseRequested.connect(_on_tab_close_requested)

        self.workspace_tabs.addTab(self.content_splitter, "Home")
        # Prevent closing the Home tab
        tab_bar = self.workspace_tabs.tabBar()
        if tab_bar is not None:
            tab_bar.setTabButton(0, tab_bar.ButtonPosition.RightSide, None)
            tab_bar.setTabButton(0, tab_bar.ButtonPosition.LeftSide, None)

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
        """Dock a submodule widget as a new tab in the workspace."""
        if not hasattr(self, "workspace_tabs"):
            logger.error("Workspace tabs not initialized; cannot dock widget.")
            return

        index = self.workspace_tabs.addTab(widget, title)
        self.workspace_tabs.setCurrentIndex(index)

    def _open_library_tab(self) -> None:
        """Open or focus the Library workspace tab."""
        if not hasattr(self, "workspace_tabs"):
            logger.error("Workspace tabs not initialized; cannot open Library.")
            return

        try:
            from src.launchers.library_widget import LibraryWidget
        except ImportError as e:
            logger.warning("Could not load Library tab: %s", e)
            if hasattr(self, "show_toast"):
                self.show_toast("Library is unavailable in this environment.", "error")
            return

        existing = getattr(self, "library_widget", None)
        if existing is None:
            existing = LibraryWidget(self)
            self.library_widget = existing

        index = self.workspace_tabs.indexOf(existing)
        if index < 0:
            self.dock_widget_as_tab(existing, "Library")
        else:
            self.workspace_tabs.setCurrentIndex(index)

    def _popout_library(self) -> None:
        """Open the Library in a floating window, preserving one widget instance."""
        self._open_library_tab()
        widget = getattr(self, "library_widget", None)
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

        if not hasattr(self, "_popped_out_windows"):
            self._popped_out_windows: list[QDialog] = []

        # We use a non-modal dialog to allow it to float freely
        win = QDialog(self, Qt.WindowType.Window)
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
        if hasattr(self, "sidekick_sidebar") and self.sidekick_sidebar is not None:
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
                if visible and hasattr(self, "_apply_sidekick_splitter_sizes"):
                    self._apply_sidekick_splitter_sizes()

                # Keep button in sync
                if (
                    hasattr(self, "btn_ai_sidebar")
                    and self.btn_ai_sidebar.isChecked() != visible
                ):
                    self.btn_ai_sidebar.setChecked(visible)

                if visible and hasattr(self, "btn_popout_sidekick"):
                    self.btn_popout_sidekick.setVisible(True)
        else:
            logger.info("Sidekick sidebar still loading or not initialized.")
            if hasattr(self, "show_toast"):
                self.show_toast(
                    "Sidekick is still loading, please wait a moment…", "info"
                )
            # Uncheck the button since it's not ready yet
            if hasattr(self, "btn_ai_sidebar"):
                self.btn_ai_sidebar.setChecked(False)

    def _toggle_left_sidebar(self, checked: bool = None) -> None:
        """Toggle the visibility of the global navigation sidebar."""
        if not hasattr(self, "sidebar_widget") or self.sidebar_widget is None:
            return
        visible = not self.sidebar_widget.isVisible() if checked is None else checked
        self.sidebar_widget.setVisible(visible)

        # Ensure proper splitter sizes when showing
        if visible and hasattr(self, "main_layout"):
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
        if not hasattr(self, "sidekick_sidebar") or self.sidekick_sidebar is None:
            return

        if not self._sidekick_popped_out:
            # Pop out
            self._sidekick_popped_out = True
            self.btn_popout_sidekick.setText("⇲ Dock Sidekick")

            from PyQt6.QtWidgets import QDialog, QVBoxLayout
            from PyQt6.QtCore import Qt

            self.sidekick_window = QDialog(self, Qt.WindowType.Window)
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

    def _setup_global_sidebar(self) -> QWidget:
        """Create the global sidebar navigation."""
        sidebar = QWidget()
        sidebar.setMinimumWidth(Styles.SIDEBAR_MIN_WIDTH)
        sidebar.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )

        try:
            from src.shared.python.theme import get_current_colors

            colors = get_current_colors()
        except ImportError:
            from src.shared.python.theme import DARK_THEME

            colors = DARK_THEME

        sidebar.setStyleSheet(f"""
            QWidget {{
                background-color: {colors.bg_elevated};
                border-right: 1px solid {colors.border_default};
            }}
            QToolButton {{
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 8px;
                color: {colors.text_secondary};
                padding: 12px 0;
            }}
            QToolButton:hover {{
                background-color: {colors.bg_highlight};
                color: {colors.text_primary};
                border: 1px solid {colors.border_default};
            }}
            QToolButton:checked {{
                background-color: {colors.primary};
                color: {colors.bg};
                border: 1px solid {colors.primary};
            }}
        """)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(8, 20, 8, 20)
        layout.setSpacing(15)

        btn_home = self._build_sidebar_button(
            "Home",
            "home",
            checkable=True,
        )
        btn_home.setChecked(True)

        btn_engines = self._build_sidebar_button(
            "Engines",
            "computer",
            checkable=True,
        )

        btn_biomechanics = self._build_sidebar_button(
            "Biomechanics",
            "accessibility",
            checkable=True,
        )
        btn_biomechanics.setAccessibleDescription(
            "Filter tiles to show biomechanics and motion analysis tools"
        )

        btn_simulation = self._build_sidebar_button(
            "Simulation",
            "sports_golf",
            checkable=True,
        )

        btn_motion_matching = self._build_sidebar_button(
            "Motion Match",
            "directions_run",
            checkable=True,
        )

        btn_motion_capture = self._build_sidebar_button(
            "MoCap",
            "videocam",
            checkable=True,
        )

        btn_tools = self._build_sidebar_button(
            "Tools",
            "build",
            checkable=True,
        )

        btn_library = self._build_sidebar_button(
            "Library",
            "book",
            checkable=True,
        )
        btn_library.setAccessibleDescription("Open the document library tab")

        # If _show_preferences exists in the mixed-in class, use it.
        # Otherwise, we gracefully handle it to avoid crashes in tests.
        btn_settings = self._build_sidebar_button(
            "Settings",
            "settings",
            checkable=False,
        )
        if hasattr(self, "_show_preferences"):
            btn_settings.clicked.connect(self._show_preferences)

        # Setup mutually exclusive active-state routing for navigation
        self.sidebar_group = QButtonGroup(self)
        self.sidebar_group.addButton(btn_home, 0)
        self.sidebar_group.addButton(btn_engines, 1)
        self.sidebar_group.addButton(btn_biomechanics, 2)
        self.sidebar_group.addButton(btn_simulation, 3)
        self.sidebar_group.addButton(btn_motion_matching, 4)
        self.sidebar_group.addButton(btn_motion_capture, 5)
        self.sidebar_group.addButton(btn_tools, 6)
        self.sidebar_group.addButton(btn_library, 7)
        self.sidebar_group.idClicked.connect(self._on_sidebar_routed)

        self.btn_library_sidebar = btn_library

        # Space navigation buttons evenly to fill available height
        layout.addWidget(btn_home)
        layout.addStretch(1)
        layout.addWidget(btn_engines)
        layout.addStretch(1)
        layout.addWidget(btn_biomechanics)
        layout.addStretch(1)
        layout.addWidget(btn_simulation)
        layout.addStretch(1)
        layout.addWidget(btn_motion_matching)
        layout.addStretch(1)
        layout.addWidget(btn_motion_capture)
        layout.addStretch(1)
        layout.addWidget(btn_tools)
        layout.addStretch(1)
        layout.addWidget(btn_library)
        layout.addStretch(3)  # larger gap before bottom group
        layout.addWidget(btn_settings)

        # Set explicit focus order for keyboard navigation
        sidebar.setFocusProxy(btn_home)
        QWidget.setTabOrder(btn_home, btn_engines)
        QWidget.setTabOrder(btn_engines, btn_biomechanics)
        QWidget.setTabOrder(btn_biomechanics, btn_simulation)
        QWidget.setTabOrder(btn_simulation, btn_motion_matching)
        QWidget.setTabOrder(btn_motion_matching, btn_motion_capture)
        QWidget.setTabOrder(btn_motion_capture, btn_tools)
        QWidget.setTabOrder(btn_tools, btn_library)
        QWidget.setTabOrder(btn_library, btn_settings)

        scroll_area = QScrollArea()
        scroll_area.setWidget(sidebar)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setMinimumWidth(Styles.SIDEBAR_MIN_WIDTH)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setStyleSheet(f"""
            QScrollArea {{
                background-color: {colors.bg_elevated};
                border-right: 1px solid {colors.border_default};
            }}
        """)

        return scroll_area

    def _on_sidebar_routed(self, button_id: int) -> None:
        """Route sidebar navigation to filter the grid layout.

        Button IDs
        ----------
        0  All (Home)
        1  Physics Engines
        2  Biomechanics
        """
        if button_id == 7:
            self._open_library_tab()
            return
        if not hasattr(self, "layout_manager"):
            return

        _CATEGORY_MAP: dict[int, str] = {
            0: "All",
            1: "Physics Engines",
            2: "Biomechanics",
            3: "Simulation",
            4: "Motion Matching",
            5: "Motion Capture",
            6: "Tools & Data",
        }
        self.layout_manager.current_category_filter = _CATEGORY_MAP.get(
            button_id, "All"
        )

        if hasattr(self, "_rebuild_grid"):
            self._rebuild_grid()

    def _build_menu_bar_widget(self) -> QMenuBar:
        """Build a populated ``QMenuBar`` for the frameless launcher (#5624).

        Returns a standalone ``QMenuBar`` (parent ``self``) that the
        caller adds to the central widget's outer ``QVBoxLayout`` so it
        sits **below** the custom title bar.

        Postcondition: the returned widget is non-null, parented to
        ``self``, and populated with the File/View/Tools/Help menus.

        We deliberately do not call ``QMainWindow.setMenuBar`` — that
        reserves the native top strip above the central widget, which
        on a frameless window draws above the custom title bar.
        """
        menubar = QMenuBar(self)
        # Postcondition (DbC): a non-null QMenuBar is returned.
        assert menubar is not None, (
            "QMenuBar construction returned None — should be impossible"
        )

        self._setup_file_menu(menubar)
        self._setup_view_menu(menubar)
        self._setup_tools_menu(menubar)
        self._setup_help_menu(menubar)
        return menubar

    def _setup_menu_bar(self) -> None:
        """Set up the application menu bar.

        .. deprecated:: #5624
            Kept for backwards compatibility with any caller still
            reaching for the legacy hook.  Prefer
            ``_build_menu_bar_widget`` + adding the result to the
            central widget's outer layout.
        """
        menubar = self.menuBar()

        self._setup_file_menu(menubar)
        self._setup_view_menu(menubar)
        self._setup_tools_menu(menubar)
        self._setup_help_menu(menubar)
        menubar.setCornerWidget(
            _build_menu_bar_close_widget(self, self.close),
            Qt.Corner.TopRightCorner,
        )

    def _setup_file_menu(self, menubar: Any) -> None:
        if menubar is None:
            raise ValueError("menubar must be provided")
        file_menu = menubar.addMenu("&File")

        action_preferences = QAction("&Preferences...", self)
        action_preferences.setShortcut("Ctrl+,")
        action_preferences.setToolTip("Edit application preferences and theme")
        action_preferences.setStatusTip("Opens preferences")
        action_preferences.triggered.connect(self._show_preferences)
        file_menu.addAction(action_preferences)

        file_menu.addSeparator()

        action_exit = QAction("E&xit", self)
        action_exit.setShortcut("Ctrl+Q")
        action_exit.setToolTip("Close the launcher")
        action_exit.setStatusTip("Quits the application")
        action_exit.triggered.connect(self.close)
        file_menu.addAction(action_exit)

    def _setup_view_menu(self, menubar: Any) -> None:
        if menubar is None:
            raise ValueError("menubar must be provided")
        view_menu = menubar.addMenu("&View")

        from PyQt6.QtGui import QActionGroup

        self._viewmode_action_group = QActionGroup(self)
        self._viewmode_action_group.setExclusive(True)
        self._viewmode_actions: dict[ViewMode, QAction] = {}
        for label, mode, shortcut in (
            ("Tile &Small", ViewMode.SMALL, "Ctrl+1"),
            ("Tile &Medium", ViewMode.MEDIUM, "Ctrl+2"),
            ("Tile &Large", ViewMode.LARGE, "Ctrl+3"),
            ("List &Small", ViewMode.LIST_SMALL, "Ctrl+4"),
            ("List &Large", ViewMode.LIST_LARGE, "Ctrl+5"),
        ):
            act = QAction(label, self)
            act.setCheckable(True)
            act.setShortcut(shortcut)
            act.setToolTip(f"Switch tile layout to {label.replace('&', '')} mode")
            act.setStatusTip(act.toolTip())
            act.triggered.connect(
                lambda _checked=False, m=mode: self._set_view_mode_from_menu(m)
            )
            self._viewmode_action_group.addAction(act)
            view_menu.addAction(act)
            self._viewmode_actions[mode] = act
        # Default checkmark on LIST_LARGE.
        self._viewmode_actions[ViewMode.LIST_LARGE].setChecked(True)

        view_menu.addSeparator()

        action_layout_mode = QAction("&Edit Layout Mode", self)
        action_layout_mode.setCheckable(True)
        action_layout_mode.setToolTip("Toggle drag-and-drop reordering of model tiles")
        action_layout_mode.setStatusTip("Toggles layout-edit mode")
        action_layout_mode.triggered.connect(self._toggle_layout_mode_from_menu)
        view_menu.addAction(action_layout_mode)
        self._action_layout_mode = action_layout_mode

        action_customize_tiles = QAction("&Select Visible Tiles...", self)
        action_customize_tiles.setToolTip(
            "Select which tiles are visible in the layout"
        )
        action_customize_tiles.setStatusTip("Select visible tiles")
        action_customize_tiles.triggered.connect(self.open_layout_manager)
        view_menu.addAction(action_customize_tiles)
        self.action_customize_tiles = action_customize_tiles

        view_menu.addSeparator()

        action_context_help = QAction("Context &Help Panel", self)
        action_context_help.setCheckable(True)
        action_context_help.setToolTip("Show or hide the side help panel")
        action_context_help.setStatusTip("Toggles context-help panel")
        action_context_help.triggered.connect(self._toggle_context_help)
        view_menu.addAction(action_context_help)
        self._action_context_help = action_context_help

        action_console = QAction("&Process Output Console", self)
        action_console.setCheckable(True)
        action_console.setChecked(False)
        action_console.setShortcut("Ctrl+`")
        action_console.setToolTip("Show or hide the launched-process output console")
        action_console.setStatusTip("Toggles process-output console")
        action_console.triggered.connect(
            lambda checked: self._console_dock.setVisible(checked)
        )
        view_menu.addAction(action_console)
        self._action_console = action_console

        view_menu.addSeparator()
        theme_menu = view_menu.addMenu("&Theme")
        self._setup_theme_menu(theme_menu)

    def _setup_tools_menu(self, menubar: Any) -> None:
        if menubar is None:
            raise ValueError("menubar must be provided")
        tools_menu = menubar.addMenu("&Tools")

        action_env = QAction("&Environment Manager...", self)
        action_env.setToolTip("Inspect Python environments and engine availability")
        action_env.setStatusTip("Opens environment manager")
        action_env.triggered.connect(lambda: self._open_settings(tab=1))
        tools_menu.addAction(action_env)

        action_diag = QAction("&Diagnostics...", self)
        action_diag.setToolTip("Run a diagnostic sweep of the install")
        action_diag.setStatusTip("Opens diagnostics")
        action_diag.triggered.connect(lambda: self._open_settings(tab=2))
        tools_menu.addAction(action_diag)

    def _setup_help_menu(self, menubar: Any) -> None:
        if menubar is None:
            raise ValueError("menubar must be provided")
        help_menu = menubar.addMenu("&Help")

        # User Manual + topic-help actions live under the legacy in-app help
        # dialog.  The new structured Help submenu (User Guide, Loaders,
        # Keyboard Shortcuts, Report a Bug, About) is appended after it via
        # build_help_menu_into so the entries gain consistent tooltips and
        # status tips.
        from src.launchers.about_dialog import (
            open_issues_page,
            open_motion_match_loaders_doc,
            open_user_guide,
            show_about_dialog,
        )
        from src.launchers.help_menu import show_keyboard_shortcuts_modal

        action_manual = QAction("&User Manual", self)
        action_manual.setShortcut("F1")
        action_manual.setToolTip("Open the in-app help dialog")
        action_manual.setStatusTip("Opens user manual")
        action_manual.triggered.connect(lambda: self._show_help_dialog())
        help_menu.addAction(action_manual)

        action_context_docs = QAction("Context &Documentation", self)
        action_context_docs.setToolTip("Open context-aware documentation")
        action_context_docs.setStatusTip("Opens Context Help")
        if hasattr(self, "_toggle_context_help"):
            action_context_docs.triggered.connect(self._toggle_context_help)
        help_menu.addAction(action_context_docs)

        action_user_guide = QAction("User &Guide (online)", self)
        action_user_guide.setToolTip(
            "Open the bundled user guide in the system browser"
        )
        action_user_guide.setStatusTip("Opens user guide")
        action_user_guide.triggered.connect(lambda: open_user_guide())
        help_menu.addAction(action_user_guide)

        action_loaders = QAction("&Motion-Match Loaders", self)
        action_loaders.setToolTip("Reference for loading motion-target files")
        action_loaders.setStatusTip("Opens motion-match loader reference")
        action_loaders.triggered.connect(lambda: open_motion_match_loaders_doc())
        help_menu.addAction(action_loaders)

        action_project_map = QAction("&Project Map", self)
        action_project_map.setToolTip("Open the project-map document")
        action_project_map.setStatusTip("Opens project map")
        action_project_map.triggered.connect(self._open_project_map)
        help_menu.addAction(action_project_map)

        help_menu.addSeparator()

        topic_tips: dict[str, tuple[str, str]] = {
            "engine_selection": (
                "How to choose between the bundled physics engines",
                "Opens engine-selection help",
            ),
            "simulation_controls": (
                "Reference for simulation playback and stepping controls",
                "Opens simulation-controls help",
            ),
            "motion_capture": (
                "Loading and aligning motion-capture targets",
                "Opens motion-capture help",
            ),
            "visualization": (
                "Camera, traces, and scene-element controls",
                "Opens visualization help",
            ),
            "analysis_tools": (
                "Built-in analysis and post-processing tools",
                "Opens analysis-tools help",
            ),
        }
        for label, topic in [
            ("Engine &Selection Guide", "engine_selection"),
            ("Simulation &Controls", "simulation_controls"),
            ("Motion &Capture", "motion_capture"),
            ("&Visualization", "visualization"),
            ("&Analysis Tools", "analysis_tools"),
        ]:
            action = QAction(label, self)
            tip, status = topic_tips[topic]
            action.setToolTip(tip)
            action.setStatusTip(status)
            action.triggered.connect(lambda checked, t=topic: self._show_help_dialog(t))
            help_menu.addAction(action)

        help_menu.addSeparator()

        action_shortcuts = QAction("&Keyboard Shortcuts...", self)
        action_shortcuts.setShortcut("Ctrl+?")
        action_shortcuts.setToolTip("Show every registered keyboard shortcut")
        action_shortcuts.setStatusTip("Opens keyboard-shortcuts table")

        # Prefer the structured modal that scrapes live actions; fall
        # back to the legacy overlay if the modal raises.
        def _open_shortcuts() -> None:
            try:
                show_keyboard_shortcuts_modal(self)
            except Exception:  # noqa: BLE001
                self._show_shortcuts_overlay()

        action_shortcuts.triggered.connect(_open_shortcuts)
        help_menu.addAction(action_shortcuts)

        action_report_bug = QAction("&Report a Bug...", self)
        action_report_bug.setToolTip("Open the public issue tracker in your browser")
        action_report_bug.setStatusTip("Opens issue tracker")
        action_report_bug.triggered.connect(lambda: open_issues_page())
        help_menu.addAction(action_report_bug)

        help_menu.addSeparator()

        action_about = QAction("&About UpstreamDrift", self)
        action_about.setToolTip("Show version and runtime information")
        action_about.setStatusTip("Opens About dialog")

        # Prefer the new dialog with live versions; fall back to legacy
        # if anything fails (keeps menu working in trimmed environments).
        def _open_about() -> None:
            try:
                show_about_dialog(self)
            except Exception:  # noqa: BLE001
                self._show_about_dialog()

        action_about.triggered.connect(_open_about)
        help_menu.addAction(action_about)

    def _setup_top_bar_status_and_search(self, top_bar: QHBoxLayout) -> None:
        """Add status indicator, execution mode label, and search bar to top bar."""
        # Status Indicator
        if top_bar is None:
            raise ValueError("top_bar must be provided")
        self.lbl_status = QLabel("Checking Docker...")
        self.lbl_status.setProperty("status", "inactive-bold")
        _style = self.lbl_status.style()

        if _style:
            _style.polish(self.lbl_status)
        top_bar.addWidget(self.lbl_status)

        # Engine-runtime indicator. Shows where physics engines run:
        # Native Windows (host Python), Docker container, or WSL2.
        # The accompanying ``?`` button opens a single help dialog
        # shared with the Settings → Engine Runtime group, so the
        # explanation lives in exactly one place.
        from src.launchers.runtime_mode_help import make_runtime_mode_help_button

        self.lbl_execution_mode = QLabel("Runtime: Native Windows")
        self.lbl_execution_mode.setProperty("exec_mode", "warning")
        _style = self.lbl_execution_mode.style()

        if _style:
            _style.polish(self.lbl_execution_mode)
        self.lbl_execution_mode.setToolTip(
            "Where physics engines execute — Native Windows, Docker "
            "container, or WSL2 Ubuntu. Click the ? for full details."
        )
        top_bar.addWidget(self.lbl_execution_mode)

        self.btn_runtime_help = make_runtime_mode_help_button(self)
        top_bar.addWidget(self.btn_runtime_help)

        top_bar.addStretch()

        # Search Bar
        self.search_input = AutoCompleteLineEdit(words=build_vocabulary())
        self.search_input.setPlaceholderText("Search models...")
        try:
            from src.shared.python.theme.responsive import (
                set_text_minimum_width,
                TextWidthSpec,
            )

            set_text_minimum_width(
                self.search_input,
                TextWidthSpec(minimum_px=250),
            )
        except ImportError:
            self.search_input.setFixedWidth(250)
        self.search_input.setToolTip("Filter models by name or description (Ctrl+F)")
        self.search_input.setAccessibleName("Search models")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.textChanged.connect(self.update_search_filter)
        top_bar.addWidget(self.search_input)

    def _setup_top_bar_config_checkboxes(self, top_bar: QHBoxLayout) -> None:
        """Create config checkboxes and layout controls, adding them to top bar."""
        from PyQt6.QtCore import QSettings

        settings = QSettings("UpstreamDrift", "Launcher")

        self.chk_live = QCheckBox("Live Viz")
        self.chk_live.setChecked(settings.value("chk_live", True, type=bool))

        self.chk_gpu = QCheckBox("GPU")
        self.chk_gpu.setChecked(settings.value("chk_gpu", False, type=bool))

        self.chk_docker = QCheckBox("Docker")
        docker_default = getattr(self, "docker_available", False)
        self.chk_docker.setChecked(
            settings.value("chk_docker", docker_default, type=bool)
        )
        self.chk_docker.stateChanged.connect(self._on_docker_mode_changed)

        self.chk_wsl = QCheckBox("WSL")
        self.chk_wsl.setChecked(settings.value("chk_wsl", False, type=bool))
        self.chk_wsl.stateChanged.connect(self._on_wsl_mode_changed)

        # Layout controls were moved to the View menu per user request.

    def _setup_top_bar_action_buttons(self, top_bar: QHBoxLayout) -> None:
        """Add Help, Settings, and AI Assistant buttons to top bar."""
        # Action buttons were moved to the left sidebar per user request.

    def _register_top_bar_tooltips(self) -> None:
        """Register enhanced tooltips for configuration checkboxes."""
        from src.launchers.launcher_constants import HELP_SYSTEM_AVAILABLE

        if not HELP_SYSTEM_AVAILABLE:
            return

        from src.shared.python.gui_pkg.help_system import TooltipManager

        TooltipManager.register_tooltip(
            self.chk_live,
            "Live Visualization",
            "Enable real-time 3D visualization during simulation.",
            "visualization",
        )
        TooltipManager.register_tooltip(
            self.chk_gpu,
            "GPU Acceleration",
            "Use GPU for physics computation when available.",
            "engine_selection",
        )
        TooltipManager.register_tooltip(
            self.chk_docker,
            "Docker container runtime",
            "Run physics engines inside the upstream-drift:engine Linux "
            "container. Full Drake/Pinocchio support; requires Docker "
            "installed and the image built (Settings → Docker Image).",
            "engine_selection",
        )
        TooltipManager.register_tooltip(
            self.chk_wsl,
            "WSL2 Ubuntu runtime",
            "Run physics engines in your WSL2 Ubuntu user environment. "
            "Same Linux wheels as Docker mode but no container layer — "
            "faster file I/O and easier interactive debugging.",
            "engine_selection",
        )

    # ---- View-mode + zoom controls --------------------------------------

    _ZOOM_SLIDER_STEPS = 100  # slider integer range -> [MIN, MAX] tile_scale

    def _slider_to_scale(self, value: int) -> float:
        """Map a slider integer ``value`` to a tile_scale float."""
        v = max(0, min(self._ZOOM_SLIDER_STEPS, int(value)))
        frac = v / float(self._ZOOM_SLIDER_STEPS)
        return TILE_SCALE_MIN + (TILE_SCALE_MAX - TILE_SCALE_MIN) * frac

    def _scale_to_slider(self, scale: float) -> int:
        """Map a tile_scale float back to slider integer steps."""
        s = max(TILE_SCALE_MIN, min(TILE_SCALE_MAX, float(scale)))
        frac = (s - TILE_SCALE_MIN) / (TILE_SCALE_MAX - TILE_SCALE_MIN)
        return int(round(frac * self._ZOOM_SLIDER_STEPS))

    def _setup_view_mode_and_zoom(self, top_bar: QHBoxLayout) -> None:
        """Add discrete view-mode dropdown and a compact, elegant zoom slider to top bar."""

        # Ctrl+= / Ctrl+- shortcuts adjust zoom by one step (~1.75% scale).
        sc_in = QShortcut(QKeySequence("Ctrl+="), self)
        sc_in.activated.connect(lambda: self._nudge_zoom(+5))
        sc_in_alt = QShortcut(QKeySequence("Ctrl++"), self)
        sc_in_alt.activated.connect(lambda: self._nudge_zoom(+5))
        sc_out = QShortcut(QKeySequence("Ctrl+-"), self)
        sc_out.activated.connect(lambda: self._nudge_zoom(-5))

    def _nudge_zoom(self, delta_steps: int) -> None:
        """Adjust the zoom slider by ``delta_steps`` integer ticks."""
        slider = getattr(self, "zoom_slider", None)
        if slider is None:
            return
        slider.setValue(slider.value() + delta_steps)

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

    def _on_view_mode_changed(self, index: int) -> None:
        """Apply the selected view mode to the layout manager + grid."""
        combo = getattr(self, "view_mode_combo", None)
        if combo is None:
            return
        mode = combo.itemData(index)
        if not isinstance(mode, ViewMode):
            return
        self._apply_view_mode(mode, sync_combo=False)

    def _set_view_mode_from_menu(self, mode: ViewMode) -> None:
        """Drive the view-mode change from the View menu's submenu."""
        self._apply_view_mode(mode, sync_combo=True)

    def _apply_view_mode(self, mode: ViewMode, *, sync_combo: bool) -> None:
        """Single source of truth for changing tile layout mode.

        Keeps the menubar action group, the top-bar dropdown, the zoom
        slider, and the grid in sync regardless of which surface
        triggered the change.
        """
        lm = getattr(self, "layout_manager", None)
        if lm is None:
            return
        lm.set_view_mode(mode)
        # Sync menu action checkmarks regardless.
        actions = getattr(self, "_viewmode_actions", None)
        if actions and mode in actions and not actions[mode].isChecked():
            actions[mode].setChecked(True)
        # Sync top-bar dropdown menu action checkmarks regardless.
        top_actions = getattr(self, "_top_viewmode_actions", None)
        if top_actions and mode in top_actions and not top_actions[mode].isChecked():
            top_actions[mode].setChecked(True)
        # Update zoom slider/label to reflect the mode's default scale.
        if hasattr(self, "zoom_slider"):
            self.zoom_slider.blockSignals(True)
            self.zoom_slider.setValue(self._scale_to_slider(lm.tile_scale))
            self.zoom_slider.blockSignals(False)
        if hasattr(self, "lbl_zoom_pct"):
            self.lbl_zoom_pct.setText(f"{int(round(lm.tile_scale * 100))}%")
        if hasattr(self, "grid_layout"):
            lm.rebuild_grid(self.grid_layout)
        if hasattr(self, "_save_layout"):
            self._save_layout()

    def _on_zoom_slider_changed(self, value: int) -> None:
        """Live-resize all model cards to match the new slider position."""
        lm = getattr(self, "layout_manager", None)
        scale = self._slider_to_scale(value)
        if hasattr(self, "lbl_zoom_pct"):
            self.lbl_zoom_pct.setText(f"{int(round(scale * 100))}%")
        if lm is None:
            return
        lm.set_tile_scale(scale)
        if hasattr(self, "_rebuild_grid"):
            self._rebuild_grid()
        if hasattr(self, "_save_layout"):
            self._save_layout()

    def _setup_top_bar(self) -> QHBoxLayout:
        """Set up the top tool bar."""
        top_bar = QHBoxLayout()

        # Modern toggles for the sidebars (left nav and right sidekick)
        self.btn_toggle_left_sidebar = QToolButton(self)
        try:
            from src.shared.python.theme.icon_utils import IconColorizer

            self.btn_toggle_left_sidebar.setIcon(
                IconColorizer.get_icon("menu", "#cccccc")
            )
        except ImportError:
            self.btn_toggle_left_sidebar.setText("☰")
        self.btn_toggle_left_sidebar.setToolTip("Toggle Navigation Sidebar")
        self.btn_toggle_left_sidebar.setCheckable(True)
        self.btn_toggle_left_sidebar.setChecked(True)
        self.btn_toggle_left_sidebar.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_toggle_left_sidebar.setStyleSheet(
            "QToolButton { background: transparent; padding: 4px 8px; border-radius: 4px; } QToolButton:hover { background: #2a2a2a; }"
        )
        self.btn_toggle_left_sidebar.clicked.connect(self._toggle_left_sidebar)
        top_bar.addWidget(self.btn_toggle_left_sidebar)

        self._setup_top_bar_status_and_search(top_bar)
        self._setup_top_bar_config_checkboxes(top_bar)
        self._setup_top_bar_action_buttons(top_bar)

        self.btn_toggle_right_sidebar = QToolButton(self)
        try:
            from src.shared.python.theme.icon_utils import IconColorizer

            self.btn_toggle_right_sidebar.setIcon(
                IconColorizer.get_icon("chat", "#cccccc")
            )
        except ImportError:
            self.btn_toggle_right_sidebar.setText("💬")
        self.btn_toggle_right_sidebar.setToolTip("Toggle Sidekick Chat")
        self.btn_toggle_right_sidebar.setCheckable(True)
        self.btn_toggle_right_sidebar.setChecked(True)
        self.btn_toggle_right_sidebar.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_toggle_right_sidebar.setStyleSheet(
            "QToolButton { background: transparent; padding: 4px 8px; border-radius: 4px; } QToolButton:hover { background: #2a2a2a; }"
        )
        self.btn_toggle_right_sidebar.clicked.connect(self._toggle_sidekick)
        top_bar.addWidget(self.btn_toggle_right_sidebar)

        # Context Help Dock
        self._setup_context_help()

        # Register enhanced tooltips
        self._register_top_bar_tooltips()

        return top_bar

    def _setup_grid_area(self, layout: QVBoxLayout) -> None:
        """Set up the scrollable grid area."""
        if layout is None:
            raise ValueError("layout must be provided")
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setProperty("class", "transparent")
        _style = self.scroll_area.style()

        if _style:
            _style.polish(self.scroll_area)

        self.grid_container = QWidget()
        self.grid_container.setProperty("class", "transparent")
        _style = self.grid_container.style()

        if _style:
            _style.polish(self.grid_container)
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(20)
        self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.scroll_area.setWidget(self.grid_container)
        layout.addWidget(self.scroll_area, 1)

    def _setup_bottom_bar(self) -> QHBoxLayout:
        """Set up the bottom bar with launch button."""
        bottom_bar = QHBoxLayout()
        bottom_bar.addStretch()

        self.btn_launch = QPushButton("Select a Model")
        self.btn_launch.setEnabled(False)
        self.btn_launch.setFixedHeight(50)
        self.btn_launch.setFont(get_display_font(size=12, weight=Weights.BOLD))
        self.btn_launch.setProperty("class", "launch-ready")
        _style = self.btn_launch.style()

        if _style:
            _style.polish(self.btn_launch)
        self.btn_launch.clicked.connect(self.launch_simulation)
        self.btn_launch.setCursor(Qt.CursorShape.PointingHandCursor)
        bottom_bar.addWidget(self.btn_launch)

        return bottom_bar

    def _setup_search_shortcuts(self) -> None:
        """Setup keyboard shortcuts for search."""
        shortcut_search = QShortcut(QKeySequence("Ctrl+F"), self)
        shortcut_search.activated.connect(self._focus_search)

        shortcut_escape = QShortcut(QKeySequence("Esc"), self)
        shortcut_escape.activated.connect(self._clear_search)

    def _focus_search(self) -> None:
        """Focus and select all text in search bar."""
        self.search_input.setFocus()
        self.search_input.selectAll()

    def _clear_search(self) -> None:
        """Clear the search filter and remove focus from search bar."""
        if self.search_input.hasFocus():
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

        console_container = QWidget()
        console_layout = QVBoxLayout(console_container)
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

        from PyQt6.QtWidgets import QDialog

        self._console_dock = QDialog(self, Qt.WindowType.Window)
        self._console_dock.setWindowTitle("Process Output")
        self._console_dock.resize(800, 300)

        dl_layout = QVBoxLayout(self._console_dock)
        dl_layout.setContentsMargins(0, 0, 0, 0)
        dl_layout.addWidget(console_container)

        self._console_dock.hide()

    def _on_process_output(self, engine_name: str, line: str) -> None:
        """Receive a line of output from a subprocess (thread-safe)."""
        QTimer.singleShot(
            0,
            lambda: self._append_console_line(engine_name, line),
        )

    def _append_console_line(self, engine_name: str, line: str) -> None:
        """Append a formatted line to the console widget (GUI thread only)."""
        if engine_name is None:
            raise ValueError("engine_name must be provided")
        if not self._console_dock.isVisible():
            self._console_dock.show()
            if hasattr(self, "_action_console"):
                self._action_console.setChecked(True)

        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._console_text.appendPlainText(f"[{ts}] [{engine_name}] {line}")

    def toggle_process_console(self) -> None:
        """Toggle visibility of the Process Output dock."""
        self._console_dock.setVisible(not self._console_dock.isVisible())

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

        self.context_help = QDialog(self, Qt.WindowType.Window)
        self.context_help.setWindowTitle("Context Help")
        self.context_help.resize(400, 800)

        dl_layout = QVBoxLayout(self.context_help)
        dl_layout.setContentsMargins(0, 0, 0, 0)

        help_widget = ContextHelpDock(self)
        dl_layout.addWidget(help_widget)

        # Proxy the update_context method to the inner widget
        self.context_help.update_context = help_widget.update_context

        self.context_help.hide()

    # -- Overlay --

    def _init_overlay(self) -> None:
        """Initialize the screen overlay."""
        try:
            from src.shared.python.ui.overlay import OverlayWidget

            self.overlay = OverlayWidget(self)
            self.overlay.hide()
        except (ImportError, TypeError):
            logger.warning("OverlayWidget could not be initialized.")

    def _toggle_overlay(self) -> None:
        """Toggle the screen overlay."""
        if hasattr(self, "overlay"):
            self.overlay.toggle()
