"""UI setup and initialization mixins for GolfLauncher.

Contains menu bar, top bar, grid area, bottom bar, search, console,
context help, and AI panel setup methods.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import (
    QAction,
    QFont,
    QKeySequence,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QCheckBox,
    QDockWidget,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.logging_config import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class LauncherUISetupMixin:
    """Mixin for GolfLauncher UI initialization.

    Provides methods for building the menu bar, top bar, grid area,
    bottom bar, search shortcuts, process console, context help, and AI panel.
    """

    def init_ui(self) -> None:
        """Initialize the user interface."""
        # --- Menu Bar ---
        self._setup_menu_bar()

        # Main Widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)

        # --- Top Bar ---
        top_bar = self._setup_top_bar()
        main_layout.addLayout(top_bar)

        # --- Content area with horizontal splitter (tiles | AI chat) ---
        self.content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.content_splitter.setHandleWidth(3)
        self.content_splitter.setStyleSheet(
            "QSplitter::handle { background-color: #484f58; }"
        )

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

        main_layout.addWidget(self.content_splitter, 1)

        # Apply dark theme
        self.apply_styles()

        # Keyboard shortcuts
        self._setup_search_shortcuts()

        # Initialize Overlay
        self._init_overlay()

    def _setup_menu_bar(self) -> None:
        """Set up the application menu bar."""
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu("&File")

        action_preferences = QAction("&Preferences...", self)
        action_preferences.setShortcut("Ctrl+,")
        action_preferences.triggered.connect(self._show_preferences)
        file_menu.addAction(action_preferences)

        file_menu.addSeparator()

        action_exit = QAction("E&xit", self)
        action_exit.setShortcut("Ctrl+Q")
        action_exit.triggered.connect(self.close)
        file_menu.addAction(action_exit)

        # View Menu
        view_menu = menubar.addMenu("&View")

        action_layout_mode = QAction("&Edit Layout Mode", self)
        action_layout_mode.setCheckable(True)
        action_layout_mode.triggered.connect(self._toggle_layout_mode_from_menu)
        view_menu.addAction(action_layout_mode)
        self._action_layout_mode = action_layout_mode

        view_menu.addSeparator()

        action_context_help = QAction("Context &Help Panel", self)
        action_context_help.setCheckable(True)
        action_context_help.triggered.connect(self._toggle_context_help)
        view_menu.addAction(action_context_help)
        self._action_context_help = action_context_help

        action_console = QAction("&Process Output Console", self)
        action_console.setCheckable(True)
        action_console.setChecked(False)
        action_console.setShortcut("Ctrl+`")
        action_console.triggered.connect(
            lambda checked: self._console_dock.setVisible(checked)
        )
        view_menu.addAction(action_console)
        self._action_console = action_console

        # Theme submenu under View
        view_menu.addSeparator()
        theme_menu = view_menu.addMenu("&Theme")
        self._setup_theme_menu(theme_menu)

        # Tools Menu
        tools_menu = menubar.addMenu("&Tools")

        action_env = QAction("&Environment Manager...", self)
        action_env.triggered.connect(lambda: self._open_settings(tab=1))
        tools_menu.addAction(action_env)

        action_diag = QAction("&Diagnostics...", self)
        action_diag.triggered.connect(lambda: self._open_settings(tab=2))
        tools_menu.addAction(action_diag)

        # Help Menu
        help_menu = menubar.addMenu("&Help")

        action_manual = QAction("&User Manual", self)
        action_manual.setShortcut("F1")
        action_manual.triggered.connect(lambda: self._show_help_dialog())
        help_menu.addAction(action_manual)

        action_project_map = QAction("&Project Map", self)
        action_project_map.triggered.connect(self._open_project_map)
        help_menu.addAction(action_project_map)

        # Topic-specific help items
        help_menu.addSeparator()

        for label, topic in [
            ("Engine &Selection Guide", "engine_selection"),
            ("Simulation &Controls", "simulation_controls"),
            ("&Motion Capture", "motion_capture"),
            ("&Visualization", "visualization"),
            ("&Analysis Tools", "analysis_tools"),
        ]:
            action = QAction(label, self)
            action.triggered.connect(lambda checked, t=topic: self._show_help_dialog(t))
            help_menu.addAction(action)

        help_menu.addSeparator()

        action_shortcuts = QAction("&Keyboard Shortcuts...", self)
        action_shortcuts.setShortcut("Ctrl+?")
        action_shortcuts.triggered.connect(self._show_shortcuts_overlay)
        help_menu.addAction(action_shortcuts)

        help_menu.addSeparator()

        action_about = QAction("&About UpstreamDrift", self)
        action_about.triggered.connect(self._show_about_dialog)
        help_menu.addAction(action_about)

    def _setup_top_bar(self) -> QHBoxLayout:
        """Set up the top tool bar."""
        from src.launchers.launcher_constants import (
            AI_AVAILABLE,
            HELP_SYSTEM_AVAILABLE,
        )

        top_bar = QHBoxLayout()

        # Status Indicator
        self.lbl_status = QLabel("Checking Docker...")
        self.lbl_status.setStyleSheet("color: #aaaaaa; font-weight: bold;")
        top_bar.addWidget(self.lbl_status)

        # Execution Mode Label
        self.lbl_execution_mode = QLabel("Mode: Local (Windows)")
        self.lbl_execution_mode.setStyleSheet(
            "color: #FFD60A; font-weight: bold; margin-left: 10px;"
        )
        self.lbl_execution_mode.setToolTip(
            "Current execution environment (Local, Docker, or WSL)"
        )
        top_bar.addWidget(self.lbl_execution_mode)

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

        # Hidden configuration checkboxes
        self.chk_live = QCheckBox("Live Viz")
        self.chk_live.setChecked(True)

        self.chk_gpu = QCheckBox("GPU")
        self.chk_gpu.setChecked(False)

        self.chk_docker = QCheckBox("Docker")
        self.chk_docker.setChecked(False)
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

        btn_help = QPushButton("Help")
        btn_help.setToolTip("View documentation and user guide (F1)")
        btn_help.clicked.connect(lambda: self._show_help_dialog())
        btn_help.setStyleSheet("""
            QPushButton {
                background-color: #0A84FF;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0077E6;
            }
        """)
        top_bar.addWidget(btn_help)

        btn_settings = QPushButton("\u2699 Settings")
        btn_settings.setToolTip("Diagnostics, environment, and build settings")
        btn_settings.setStyleSheet("""
            QPushButton {
                background-color: #484f58;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #6e7681;
            }
        """)
        btn_settings.clicked.connect(self._open_settings)
        top_bar.addWidget(btn_settings)

        # AI Assistant Button
        if AI_AVAILABLE:
            self.btn_ai = QPushButton("AI Chat [...]")
            self.btn_ai.setToolTip("Open AI Assistant for help with analysis")
            self.btn_ai.setCheckable(True)
            self.btn_ai.clicked.connect(self.toggle_ai_assistant)
            self.btn_ai.setStyleSheet("""
                QPushButton {
                    background-color: #1976d2;
                    color: white;
                    padding: 8px 16px;
                    font-weight: bold;
                    border-radius: 4px;
                }
                QPushButton:hover { background-color: #1565c0; }
                QPushButton:checked { background-color: #0d47a1; }
                """)
            top_bar.addWidget(self.btn_ai)

        # Context Help Dock
        self._setup_context_help()

        # Register enhanced tooltips
        if HELP_SYSTEM_AVAILABLE:
            from src.shared.python.help_system import TooltipManager

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
                "Docker Mode",
                "Run physics engines in Docker containers.",
                "engine_selection",
            )
            TooltipManager.register_tooltip(
                self.chk_wsl,
                "WSL Mode",
                "Run in WSL2 Ubuntu environment for full Linux engine support.",
                "engine_selection",
            )

        return top_bar

    def _setup_grid_area(self, layout: QVBoxLayout) -> None:
        """Set up the scrollable grid area."""
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setStyleSheet("QScrollArea { background: transparent; }")

        self.grid_container = QWidget()
        self.grid_container.setStyleSheet("background: transparent;")
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
        self.btn_launch.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.btn_launch.setStyleSheet("""
            QPushButton {
                background-color: #2da44e;
                color: white;
                border-radius: 6px;
                padding: 0 40px;
            }
            QPushButton:disabled {
                background-color: #444444;
                color: #888888;
            }
            QPushButton:hover:!disabled {
                background-color: #2c974b;
            }
            """)
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
        self._console_text.setStyleSheet(
            "QPlainTextEdit {"
            "  background-color: #1e1e1e;"
            "  color: #d4d4d4;"
            "  font-family: 'Cascadia Code', 'Consolas', 'Courier New', monospace;"
            "  font-size: 11px;"
            "  border: none;"
            "}"
        )

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

            url = "http://127.0.0.1:8000/api/chat/sessions"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                sessions = json.loads(resp.read().decode("utf-8"))

            if sessions:
                session_id = sessions[0]["session_id"]
            else:
                session_id = None

            session_file = (
                Path.home() / ".golf_modeling_suite" / "active_chat_session.txt"
            )
            session_file.parent.mkdir(parents=True, exist_ok=True)
            if session_id:
                session_file.write_text(session_id, encoding="utf-8")
                logger.info("Synced chat session: %s", session_id)
        except (ImportError, OSError) as e:
            logger.debug("Chat server sync skipped (server may not be running): %s", e)
        except (ImportError, RuntimeError, OSError):
            # Catch any other unexpected errors (e.g. urllib.error.HTTPError)
            # to prevent crashing the launcher during initialization
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
