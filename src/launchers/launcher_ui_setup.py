"""UI setup and initialization mixins for GolfLauncher.

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
    """Mixin for GolfLauncher UI initialization.

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

            self.title_bar = CustomTitleBar(self, show_close_button=False)
            self.title_bar.minimize_requested.connect(self.showMinimized)
            self.title_bar.maximize_requested.connect(
                lambda: (
                    self.showNormal() if self.isMaximized() else self.showMaximized()
                )
            )
            self.title_bar.close_requested.connect(self.close)
            self.title_bar.move_requested.connect(self.move)

            # Hide the native OS title bar since we are using a custom one
            self.setWindowFlags(Qt.WindowType.FramelessWindowHint | self.windowFlags())

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
        outer_vbox.addWidget(main_layout)

        # Expose the splitter for downstream features that embed extra
        # panes (e.g. ``_install_sidekick_sidebar`` in #5624 adds the
        # Sidekick widget as the third pane instead of using
        # ``addDockWidget``, which misbehaves on a frameless window).
        self.main_layout = main_layout

        # --- Global Sidebar ---
        sidebar = self._setup_global_sidebar()
        main_layout.addWidget(sidebar)

        # Content Container
        content_container = QWidget()
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
        content_layout.addLayout(top_bar)

        # --- Content area with horizontal splitter (tiles | AI chat) ---
        self.content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.content_splitter.setHandleWidth(3)
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

        # Right panel: AI chat (added to splitter, visible by default)
        self._ai_visible = True
        from src.launchers.launcher_constants import AI_AVAILABLE

        if AI_AVAILABLE:
            self._setup_ai_panel()

        content_layout.addWidget(self.content_splitter, 1)

        main_layout.addWidget(content_container)

        # Proportional sizing: sidebar gets ~1/6, content ~5/6.
        main_layout.setStretchFactor(0, 1)
        main_layout.setStretchFactor(1, 5)

        # Apply dark theme
        self.apply_styles()

        # Keyboard shortcuts
        self._setup_search_shortcuts()

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
                border: none;
                border-radius: 8px;
                color: {colors.text_secondary};
                padding: 12px 0;
            }}
            QToolButton:hover {{
                background-color: {colors.bg_highlight};
                color: {colors.text_primary};
            }}
            QToolButton:checked {{
                background-color: {colors.primary};
                color: {colors.bg};
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

        # If open_settings exists in the mixed-in class, use it.
        # Otherwise, we gracefully handle it to avoid crashes in tests.
        btn_settings = self._build_sidebar_button(
            "Settings",
            "settings",
        )
        if hasattr(self, "_open_settings"):
            btn_settings.clicked.connect(self._open_settings)

        from src.launchers.launcher_constants import AI_AVAILABLE

        if AI_AVAILABLE:
            self.btn_ai_sidebar = self._build_sidebar_button(
                "Chat",
                "chat",
                checkable=True,
            )
            if hasattr(self, "toggle_ai_assistant"):
                self.btn_ai_sidebar.clicked.connect(self.toggle_ai_assistant)

        btn_docs = self._build_sidebar_button(
            "Documentation",
            "help",
            checkable=True,
        )
        if hasattr(self, "_toggle_context_help"):
            btn_docs.clicked.connect(self._toggle_context_help)

        # Setup mutually exclusive active-state routing for navigation
        self.sidebar_group = QButtonGroup(self)
        self.sidebar_group.addButton(btn_home, 0)
        self.sidebar_group.addButton(btn_engines, 1)
        self.sidebar_group.addButton(btn_biomechanics, 2)
        self.sidebar_group.addButton(btn_simulation, 3)
        self.sidebar_group.addButton(btn_motion_matching, 4)
        self.sidebar_group.addButton(btn_motion_capture, 5)
        self.sidebar_group.addButton(btn_tools, 6)
        self.sidebar_group.idClicked.connect(self._on_sidebar_routed)

        layout.addWidget(btn_home)
        layout.addWidget(btn_engines)
        layout.addWidget(btn_biomechanics)
        layout.addWidget(btn_simulation)
        layout.addWidget(btn_motion_matching)
        layout.addWidget(btn_motion_capture)
        layout.addWidget(btn_tools)
        layout.addStretch()
        if AI_AVAILABLE:
            layout.addWidget(self.btn_ai_sidebar)
        layout.addWidget(btn_docs)
        layout.addWidget(btn_settings)

        # Set explicit focus order for keyboard navigation
        sidebar.setFocusProxy(btn_home)
        QWidget.setTabOrder(btn_home, btn_engines)
        QWidget.setTabOrder(btn_engines, btn_biomechanics)
        QWidget.setTabOrder(btn_biomechanics, btn_simulation)
        QWidget.setTabOrder(btn_simulation, btn_motion_matching)
        QWidget.setTabOrder(btn_motion_matching, btn_motion_capture)
        QWidget.setTabOrder(btn_motion_capture, btn_tools)
        QWidget.setTabOrder(btn_tools, btn_settings)
        QWidget.setTabOrder(btn_settings, btn_docs)

        return sidebar

    def _on_sidebar_routed(self, button_id: int) -> None:
        """Route sidebar navigation to filter the grid layout.

        Button IDs
        ----------
        0  All (Home)
        1  Physics Engines
        2  Biomechanics
        """
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
        menubar.setCornerWidget(
            _build_menu_bar_close_widget(self, self.close),
            Qt.Corner.TopRightCorner,
        )
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

        # ---- View-mode submenu (Comfortable / Compact / Dense / List) ----
        # The same four modes the top-bar combo exposes, but discoverable
        # through the menu with keyboard shortcuts.
        viewmode_menu = view_menu.addMenu("Tile &Layout")
        from PyQt6.QtGui import QActionGroup

        self._viewmode_action_group = QActionGroup(self)
        self._viewmode_action_group.setExclusive(True)
        self._viewmode_actions: dict[ViewMode, QAction] = {}
        for label, mode, shortcut in (
            ("&Comfortable", ViewMode.COMFORTABLE, "Ctrl+1"),
            ("Co&mpact", ViewMode.COMPACT, "Ctrl+2"),
            ("&Dense", ViewMode.DENSE, "Ctrl+3"),
            ("&List", ViewMode.LIST, "Ctrl+4"),
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
            viewmode_menu.addAction(act)
            self._viewmode_actions[mode] = act
        # Default checkmark on LIST (matches combo default).
        self._viewmode_actions[ViewMode.LIST].setChecked(True)

        view_menu.addSeparator()

        action_layout_mode = QAction("&Edit Layout Mode", self)
        action_layout_mode.setCheckable(True)
        action_layout_mode.setToolTip("Toggle drag-and-drop reordering of model tiles")
        action_layout_mode.setStatusTip("Toggles layout-edit mode")
        action_layout_mode.triggered.connect(self._toggle_layout_mode_from_menu)
        view_menu.addAction(action_layout_mode)
        self._action_layout_mode = action_layout_mode

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
        self.chk_live = QCheckBox("Live Viz")
        self.chk_live.setChecked(True)

        self.chk_gpu = QCheckBox("GPU")
        self.chk_gpu.setChecked(False)

        self.chk_docker = QCheckBox("Docker")
        # Default to Docker mode if available
        self.chk_docker.setChecked(getattr(self, "docker_available", False))
        self.chk_docker.stateChanged.connect(self._on_docker_mode_changed)

        self.chk_wsl = QCheckBox("WSL")
        self.chk_wsl.setChecked(False)
        self.chk_wsl.stateChanged.connect(self._on_wsl_mode_changed)

        # Layout controls (combined toggle + dropdown)
        from PyQt6.QtWidgets import QToolButton, QMenu

        self.btn_modify_layout = QToolButton()
        self.btn_modify_layout.setText("Layout: Locked 🔒")
        self.btn_modify_layout.setPopupMode(
            QToolButton.ToolButtonPopupMode.MenuButtonPopup
        )
        self.btn_modify_layout.setCheckable(True)
        self.btn_modify_layout.setChecked(False)
        self.btn_modify_layout.clicked.connect(self.toggle_layout_mode)

        self.layout_menu = QMenu(self.btn_modify_layout)
        self.action_customize_tiles = self.layout_menu.addAction("Edit Tiles...")
        if self.action_customize_tiles:
            self.action_customize_tiles.setEnabled(False)
            self.action_customize_tiles.triggered.connect(self.open_layout_manager)
        self.btn_modify_layout.setMenu(self.layout_menu)

        # Only Layout controls remain in the top bar (config options moved to settings)
        top_bar.addWidget(self.btn_modify_layout)

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
        """Add view-mode combobox, zoom slider, and percent label to top bar."""
        if top_bar is None:
            raise ValueError("top_bar must be provided")

        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItem("Comfortable", ViewMode.COMFORTABLE)
        self.view_mode_combo.addItem("Compact", ViewMode.COMPACT)
        self.view_mode_combo.addItem("Dense", ViewMode.DENSE)
        self.view_mode_combo.addItem("List", ViewMode.LIST)
        self.view_mode_combo.setCurrentIndex(1)  # Compact default
        self.view_mode_combo.setToolTip("Choose how the model tiles are arranged")
        self.view_mode_combo.setAccessibleName("View mode")
        self.view_mode_combo.currentIndexChanged.connect(self._on_view_mode_changed)
        top_bar.addWidget(self.view_mode_combo)

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(0, self._ZOOM_SLIDER_STEPS)
        self.zoom_slider.setMinimumWidth(140)
        from PyQt6.QtWidgets import QSizePolicy

        self.zoom_slider.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding,
            self.zoom_slider.sizePolicy().verticalPolicy(),
        )
        self.zoom_slider.setToolTip("Adjust the size of the model tiles")
        self.zoom_slider.setAccessibleName("Tile zoom")
        self.zoom_slider.setAccessibleDescription(_build_zoom_accessible_description())
        # Set focus policy for keyboard accessibility
        self.zoom_slider.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Initial position from layout_manager if available, else compact 0.5.
        from src.launchers.launcher_constants import TILE_SCALE_DEFAULT

        initial_scale = TILE_SCALE_DEFAULT
        lm = getattr(self, "layout_manager", None)
        if lm is not None and hasattr(lm, "tile_scale"):
            initial_scale = float(lm.tile_scale)
        self.zoom_slider.setValue(self._scale_to_slider(initial_scale))
        self.zoom_slider.valueChanged.connect(self._on_zoom_slider_changed)
        top_bar.addWidget(self.zoom_slider)

        self.lbl_zoom_pct = QLabel(f"{int(round(initial_scale * 100))}%")
        self.lbl_zoom_pct.setToolTip("Current tile size as a percentage of base")
        top_bar.addWidget(self.lbl_zoom_pct)

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

        Keeps the menubar action group, the top-bar combo, the zoom
        slider, and the grid in sync regardless of which surface
        triggered the change.
        """
        lm = getattr(self, "layout_manager", None)
        if lm is None:
            return
        lm.set_view_mode(mode)
        # Sync combo box if the change came from the menu.
        if sync_combo:
            combo = getattr(self, "view_mode_combo", None)
            if combo is not None:
                idx = combo.findData(mode)
                if idx >= 0 and combo.currentIndex() != idx:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(idx)
                    combo.blockSignals(False)
        # Sync menu action checkmarks regardless.
        actions = getattr(self, "_viewmode_actions", None)
        if actions and mode in actions and not actions[mode].isChecked():
            actions[mode].setChecked(True)
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
        if hasattr(self, "_save_layout"):
            self._save_layout()

    def _setup_top_bar(self) -> QHBoxLayout:
        """Set up the top tool bar."""
        top_bar = QHBoxLayout()

        self._setup_top_bar_status_and_search(top_bar)
        self._setup_view_mode_and_zoom(top_bar)
        self._setup_top_bar_config_checkboxes(top_bar)
        self._setup_top_bar_action_buttons(top_bar)

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

        self._console_dock = QDockWidget("Process Output", self)
        self._console_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self._console_dock.setWidget(console_container)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._console_dock)
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

    # -- AI Panel --

    def _setup_ai_panel(self) -> None:
        """Set up the AI Assistant panel inside the content splitter."""
        from src.launchers.launcher_constants import AI_AVAILABLE

        if not AI_AVAILABLE:
            return

        try:
            from src.shared.python.ai.gui import AIAssistantPanel

            self.ai_panel = AIAssistantPanel(self)
            self.ai_panel.setMinimumWidth(0)
            self.content_splitter.addWidget(self.ai_panel)
            self.ai_panel.setMaximumWidth(16777215)  # Make it open by default
            self.ai_panel.settings_requested.connect(self._open_ai_settings)
            self.ai_panel.close_requested.connect(
                lambda: self.ai_panel.setMaximumWidth(0)
            )
            self._sync_chat_session()
        except ImportError as e:
            logger.error(f"Failed to initialize AI panel: {e}")
            if hasattr(self, "btn_ai"):
                self.btn_ai.setEnabled(False)
                self.btn_ai.setToolTip(f"AI Assistant unavailable: {e}")
            if hasattr(self, "btn_ai_sidebar"):
                self.btn_ai_sidebar.setEnabled(False)
                self.btn_ai_sidebar.setToolTip(f"AI Assistant unavailable: {e}")

    def _sync_chat_session(self) -> None:
        """Sync the launcher's chat session with the shared FastAPI server."""
        import json
        from pathlib import Path

        try:
            import urllib.request

            # The URL is a hardcoded literal pointing at the launcher's
            # locally-spawned FastAPI server on 127.0.0.1; there is no
            # path through which user-controlled data can influence it.
            # The companion `# nosec B310` keeps Bandit happy; the
            # `# nosemgrep` line below silences the matching Semgrep
            # `dynamic-urllib-use-detected` rule with the same rationale.
            url = "http://127.0.0.1:8000/api/chat/sessions"
            req = urllib.request.Request(url, method="GET")
            # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
            with urllib.request.urlopen(req, timeout=2) as resp:  # nosec B310 - hardcoded localhost URL, no external input
                sessions = json.loads(resp.read().decode("utf-8"))

            session_id = sessions[0]["session_id"] if sessions else None

            session_file = (
                Path.home() / ".golf_modeling_suite" / "active_chat_session.txt"
            )
            session_file.parent.mkdir(parents=True, exist_ok=True)
            if session_id:
                session_file.write_text(session_id, encoding="utf-8")
                logger.info("Synced chat session: %s", session_id)
        except (ImportError, OSError) as e:
            logger.debug("Chat server sync skipped (server may not be running): %s", e)
        except (ValueError, KeyError, IndexError, json.JSONDecodeError) as e:
            logger.debug("Chat session sync failed: %s", e)

    # -- Context Help --

    def _setup_context_help(self) -> None:
        """Setup context help dock."""
        from src.launchers.ui_components import ContextHelpDock

        self.context_help = ContextHelpDock(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.context_help)
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
