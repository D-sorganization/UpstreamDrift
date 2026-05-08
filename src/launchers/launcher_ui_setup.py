"""UI setup and initialization mixins for GolfLauncher.

Contains menu bar, top bar, grid area, bottom bar, search, console,
context help, and AI panel setup methods.
"""

# mypy: disable-error-code="attr-defined,call-overload,arg-type"

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
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
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

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.theme.style_constants import Styles
from src.shared.python.theme.typography import Weights, get_display_font

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class LauncherUISetupMixin:
    """Mixin for GolfLauncher UI initialization.

    Provides methods for building the menu bar, top bar, grid area,
    bottom bar, search shortcuts, process console, context help, and AI panel.
    """

    def _build_sidebar_button(
        self,
        label: str,
        icon_name: QStyle.StandardPixmap,
        *,
        checkable: bool = False,
    ) -> QToolButton:
        """Create an icon-first sidebar control with accessible labeling."""
        button = QToolButton()
        button.setText("")
        button.setToolTip(label)
        button.setAccessibleName(label)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        button.setIcon(self.style().standardIcon(icon_name))
        button.setIconSize(QSize(22, 22))
        button.setCheckable(checkable)
        button.setAutoRaise(True)
        return button

    def init_ui(self) -> None:
        """Initialize the user interface."""
        # --- Menu Bar ---
        self._setup_menu_bar()

        # Main Widget
        central = QWidget()
        self.setCentralWidget(central)

        # Main layout is now horizontal to accommodate the sidebar
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # --- Global Sidebar ---
        sidebar = self._setup_global_sidebar()
        main_layout.addWidget(sidebar)

        # Content Container
        content_container = QWidget()
        content_layout = QVBoxLayout(content_container)
        content_layout.setSpacing(20)
        content_layout.setContentsMargins(30, 30, 30, 30)

        # --- Top Bar ---
        top_bar = self._setup_top_bar()
        content_layout.addLayout(top_bar)

        # --- Content area with horizontal splitter (tiles | AI chat) ---
        self.content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.content_splitter.setHandleWidth(3)
        self.content_splitter.setStyleSheet(Styles.SPLITTER_HANDLE)

        # Left panel: launcher grid + bottom bar
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(20)
        self._setup_grid_area(left_layout)
        bottom_bar = self._setup_bottom_bar()
        left_layout.addLayout(bottom_bar)

        self.content_splitter.addWidget(left_panel)

        # Right panel: AI chat (added to splitter, hidden by default)
        self._ai_visible = False
        from src.launchers.launcher_constants import AI_AVAILABLE

        if AI_AVAILABLE:
            self._setup_ai_panel()

        content_layout.addWidget(self.content_splitter, 1)

        main_layout.addWidget(content_container, 1)

        # Apply dark theme
        self.apply_styles()

        # Keyboard shortcuts
        self._setup_search_shortcuts()

        # Initialize Overlay
        self._init_overlay()

    def _setup_global_sidebar(self) -> QWidget:
        """Create the thin global sidebar navigation."""
        sidebar = QWidget()
        sidebar.setFixedWidth(70)

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
            QStyle.StandardPixmap.SP_DirHomeIcon,
            checkable=True,
        )
        btn_home.setChecked(True)

        btn_engines = self._build_sidebar_button(
            "Engines",
            QStyle.StandardPixmap.SP_ComputerIcon,
            checkable=True,
        )

        # If open_settings exists in the mixed-in class, use it.
        # Otherwise, we gracefully handle it to avoid crashes in tests.
        btn_settings = self._build_sidebar_button(
            "Settings",
            QStyle.StandardPixmap.SP_FileDialogDetailedView,
        )
        if hasattr(self, "_open_settings"):
            btn_settings.clicked.connect(self._open_settings)

        btn_docs = self._build_sidebar_button(
            "Documentation",
            QStyle.StandardPixmap.SP_DialogHelpButton,
        )
        if hasattr(self, "_show_help_dialog"):
            btn_docs.clicked.connect(lambda: self._show_help_dialog())

        layout.addWidget(btn_home)
        layout.addWidget(btn_engines)
        layout.addStretch()
        layout.addWidget(btn_settings)
        layout.addWidget(btn_docs)

        return sidebar

    def _setup_menu_bar(self) -> None:
        """Set up the application menu bar."""
        menubar = self.menuBar()

        self._setup_file_menu(menubar)
        self._setup_view_menu(menubar)
        self._setup_tools_menu(menubar)
        self._setup_help_menu(menubar)

    def _setup_file_menu(self, menubar: Any) -> None:
        if not (menubar is not None):
            raise ValueError("menubar must be provided")
        if not (menubar is not None):
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
        if not (menubar is not None):
            raise ValueError("menubar must be provided")
        if not (menubar is not None):
            raise ValueError("menubar must be provided")
        view_menu = menubar.addMenu("&View")

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
        if not (menubar is not None):
            raise ValueError("menubar must be provided")
        if not (menubar is not None):
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
        if not (menubar is not None):
            raise ValueError("menubar must be provided")
        if not (menubar is not None):
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
        if not (top_bar is not None):
            raise ValueError("top_bar must be provided")
        if not (top_bar is not None):
            raise ValueError("top_bar must be provided")
        self.lbl_status = QLabel("Checking Docker...")
        self.lbl_status.setStyleSheet(Styles.STATUS_INACTIVE_BOLD)
        top_bar.addWidget(self.lbl_status)

        # Engine-runtime indicator. Shows where physics engines run:
        # Native Windows (host Python), Docker container, or WSL2.
        # The accompanying ``?`` button opens a single help dialog
        # shared with the Settings → Engine Runtime group, so the
        # explanation lives in exactly one place.
        from src.launchers.runtime_mode_help import make_runtime_mode_help_button

        self.lbl_execution_mode = QLabel("Runtime: Native Windows")
        self.lbl_execution_mode.setStyleSheet(Styles.EXEC_MODE_WARNING)
        self.lbl_execution_mode.setToolTip(
            "Where physics engines execute — Native Windows, Docker "
            "container, or WSL2 Ubuntu. Click the ? for full details."
        )
        top_bar.addWidget(self.lbl_execution_mode)

        self.btn_runtime_help = make_runtime_mode_help_button(self)
        top_bar.addWidget(self.btn_runtime_help)

        top_bar.addStretch()

        # Search Bar
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search models...")
        self.search_input.setFixedWidth(250)
        self.search_input.setToolTip("Filter models by name or description (Ctrl+F)")
        self.search_input.setAccessibleName("Search models")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.textChanged.connect(self.update_search_filter)
        top_bar.addWidget(self.search_input)

    def _setup_top_bar_config_checkboxes(self) -> None:
        """Create hidden configuration checkboxes and layout controls."""
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

        # Layout controls
        self.btn_modify_layout = QPushButton("Layout: Locked")
        self.btn_modify_layout.setCheckable(True)
        self.btn_modify_layout.setChecked(False)
        self.btn_modify_layout.clicked.connect(self.toggle_layout_mode)

        self.btn_customize_tiles = QPushButton("Edit Tiles")
        self.btn_customize_tiles.setEnabled(False)
        self.btn_customize_tiles.clicked.connect(self.open_layout_manager)

    def _setup_top_bar_action_buttons(self, top_bar: QHBoxLayout) -> None:
        """Add Help, Settings, and AI Assistant buttons to top bar."""
        if not (top_bar is not None):
            raise ValueError("top_bar must be provided")
        if not (top_bar is not None):
            raise ValueError("top_bar must be provided")
        from src.launchers.launcher_constants import AI_AVAILABLE

        btn_help = QPushButton("Help")
        btn_help.setToolTip("View documentation and user guide (F1)")
        btn_help.clicked.connect(lambda: self._show_help_dialog())
        btn_help.setStyleSheet(Styles.BTN_PRIMARY)
        top_bar.addWidget(btn_help)

        btn_settings = QPushButton("\u2699 Settings")
        btn_settings.setToolTip("Diagnostics, environment, and build settings")
        btn_settings.setStyleSheet(Styles.BTN_SECONDARY)
        btn_settings.clicked.connect(self._open_settings)
        top_bar.addWidget(btn_settings)

        # AI Assistant Button
        if AI_AVAILABLE:
            self.btn_ai = QPushButton("AI Chat [...]")
            self.btn_ai.setToolTip("Open AI Assistant for help with analysis")
            self.btn_ai.setCheckable(True)
            self.btn_ai.clicked.connect(self.toggle_ai_assistant)
            self.btn_ai.setStyleSheet(Styles.BTN_AI_CHAT)
            top_bar.addWidget(self.btn_ai)

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
        self.zoom_slider.setFixedWidth(140)
        self.zoom_slider.setToolTip("Adjust the size of the model tiles")
        self.zoom_slider.setAccessibleName("Tile zoom")

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
        lm = getattr(self, "layout_manager", None)
        if lm is None:
            return
        lm.set_view_mode(mode)
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
        self._setup_top_bar_config_checkboxes()
        self._setup_top_bar_action_buttons(top_bar)

        # Context Help Dock
        self._setup_context_help()

        # Register enhanced tooltips
        self._register_top_bar_tooltips()

        return top_bar

    def _setup_grid_area(self, layout: QVBoxLayout) -> None:
        """Set up the scrollable grid area."""
        if not (layout is not None):
            raise ValueError("layout must be provided")
        if not (layout is not None):
            raise ValueError("layout must be provided")
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setStyleSheet(Styles.SCROLL_AREA_TRANSPARENT)

        self.grid_container = QWidget()
        self.grid_container.setStyleSheet(Styles.TRANSPARENT_BG)
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
        self.btn_launch.setStyleSheet(Styles.BTN_LAUNCH_READY)
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
        self._console_text.setStyleSheet(Styles.CONSOLE_DARK)

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
        if not (engine_name is not None):
            raise ValueError("engine_name must be provided")
        if not (engine_name is not None):
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
            self.content_splitter.setCollapsible(1, True)
            self.ai_panel.setMaximumWidth(0)
            self.ai_panel.settings_requested.connect(self._open_ai_settings)
            self.ai_panel.close_requested.connect(
                lambda: self.toggle_ai_assistant(False)
            )
            self._sync_chat_session()
        except ImportError as e:
            logger.error(f"Failed to initialize AI panel: {e}")
            self.btn_ai.setEnabled(False)
            self.btn_ai.setToolTip(f"AI Assistant unavailable: {e}")

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
