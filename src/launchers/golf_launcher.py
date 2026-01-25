#!/usr/bin/env python3
"""
Unified Golf Modeling Suite Launcher (PyQt6)
Features:
- Modern UI with rounded corners.
- Modular Docker Environment Management.
- Integrated Help and Documentation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Add current directory to path so we can import ui_components if needed locally
sys.path.append(str(Path(__file__).parent))

from src.launchers.ui_components import (
    ASSETS_DIR,
    AsyncStartupWorker,
    ContextHelpDock,
    DockerCheckThread,
    DraggableModelCard,
    EnvironmentDialog,
    GolfSplashScreen,
    HelpDialog,
    LayoutManagerDialog,
    StartupResults,
)
from src.shared.python.logging_config import configure_gui_logging, get_logger

if TYPE_CHECKING:
    from src.shared.python.ui import ToastManager

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import (
    QCloseEvent,
    QFont,
    QIcon,
    QKeySequence,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDockWidget,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.secure_subprocess import (
    secure_popen,
)

# Lazy imports for heavy modules - these are loaded during async startup
# to avoid blocking the UI thread during application launch
_EngineManager: Any = None
_EngineType: Any = None
_ModelRegistry: Any = None


def _lazy_load_engine_manager() -> tuple[Any, Any]:
    """Lazily load EngineManager to speed up initial import."""
    global _EngineManager, _EngineType
    if _EngineManager is None:
        from src.shared.python.engine_manager import EngineManager as _EM
        from src.shared.python.engine_manager import EngineType as _ET

        _EngineManager = _EM
        _EngineType = _ET
    return _EngineManager, _EngineType


def _lazy_load_model_registry() -> Any:
    """Lazily load ModelRegistry to speed up initial import."""
    global _ModelRegistry
    if _ModelRegistry is None:
        from src.shared.python.model_registry import ModelRegistry as _MR

        _ModelRegistry = _MR
    return _ModelRegistry


# Import unified theme system for consistent styling
try:
    THEME_AVAILABLE = True
except ImportError:
    THEME_AVAILABLE = False

# Optional AI Assistant import (graceful degradation if not available)
try:
    from src.shared.python.ai.gui import AIAssistantPanel, AISettingsDialog

    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# Optional UI components import (graceful degradation)
try:
    from src.shared.python.ui import (
        PreferencesDialog,
        ShortcutsOverlay,
        ToastManager,
    )

    UI_COMPONENTS_AVAILABLE = True
except ImportError:
    UI_COMPONENTS_AVAILABLE = False

# Windows-specific subprocess constants
CREATE_NO_WINDOW: int
CREATE_NEW_CONSOLE: int

if os.name == "nt":
    try:
        CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        CREATE_NEW_CONSOLE = subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]
    except AttributeError:
        CREATE_NO_WINDOW = 0x08000000
        CREATE_NEW_CONSOLE = 0x00000010
else:
    CREATE_NO_WINDOW = 0
    CREATE_NEW_CONSOLE = 0

# Configure Logging using centralized module
configure_gui_logging()
logger = get_logger(__name__)

# Constants
# Constants
REPOS_ROOT = Path(__file__).parent.parent.parent.resolve()
CONFIG_DIR = REPOS_ROOT / ".kiro" / "launcher"
LAYOUT_CONFIG_FILE = CONFIG_DIR / "layout.json"
GRID_COLUMNS = 4  # Changed to 3x4 grid (12 tiles total)

DOCKER_STAGES = ["all", "mujoco", "pinocchio", "drake", "base"]


class GolfLauncher(QMainWindow):
    """Main application window for the launcher."""

    def __init__(self, startup_results: StartupResults | None = None) -> None:
        """Initialize the main window.

        Args:
            startup_results: Optional pre-loaded startup results from AsyncStartupWorker.
                            If provided, skips redundant loading of registry and engines.
        """
        super().__init__()
        self.setWindowTitle("Golf Modeling Suite - GolfingRobot")
        self.resize(1400, 900)
        self.center_window()

        # Store startup metrics for status display
        self._startup_time_ms = (
            startup_results.startup_time_ms if startup_results else 0
        )

        # Set Icon - Use Windows-optimized icon for maximum clarity on Windows
        icon_candidates = [
            ASSETS_DIR
            / "golf_robot_windows_optimized.png",  # Windows-optimized (best for Windows)
            ASSETS_DIR / "golf_robot_ultra_sharp.png",  # Ultra-sharp version
            ASSETS_DIR / "golf_robot_cropped_icon.png",  # Cropped version
            ASSETS_DIR / "golf_robot_icon.png",  # High-quality standard
            ASSETS_DIR / "golf_icon.png",  # Original fallback
        ]

        icon_loaded = False
        for icon_path in icon_candidates:
            if icon_path.exists():
                self.setWindowIcon(QIcon(str(icon_path)))
                logger.info("Loaded icon: %s", icon_path.name)
                icon_loaded = True
                break

        if not icon_loaded:
            logger.warning("No icon files found")

        # State
        self.docker_available = (
            startup_results.docker_available if startup_results else False
        )
        self.selected_model: str | None = None
        self.model_cards: dict[str, Any] = {}
        self.model_order: list[str] = []  # Track model order for drag-and-drop
        self.layout_edit_mode = False  # Track if layout editing is enabled
        self.running_processes: dict[str, subprocess.Popen] = (
            {}
        )  # Track running instances
        self.available_models: dict[str, Any] = {}
        self.special_app_lookup: dict[str, Any] = {}
        self.current_filter_text = ""

        # Use pre-loaded registry from startup results, or load fresh
        if startup_results and startup_results.registry is not None:
            self.registry = startup_results.registry
            logger.info("Using pre-loaded model registry from async startup")
        else:
            # Fallback to loading registry synchronously
            try:
                MR = _lazy_load_model_registry()
                self.registry = MR(REPOS_ROOT / "config/models.yaml")
            except (ImportError, Exception) as e:
                logger.error(f"Failed to load ModelRegistry: {e}")
                self.registry = None

        # Use pre-loaded engine manager from startup results, or load fresh
        if startup_results and startup_results.engine_manager is not None:
            self.engine_manager = startup_results.engine_manager
            logger.info("Using pre-loaded engine manager from async startup")
        else:
            # Fallback to loading engine manager synchronously
            try:
                EM, _ = _lazy_load_engine_manager()
                self.engine_manager = EM(REPOS_ROOT)
            except Exception as e:
                logger.warning(f"Failed to initialize EngineManager: {e}")
                self.engine_manager = None

        self._build_available_models()
        self._initialize_model_order()

        self.init_ui()

        # Use pre-loaded Docker status or check asynchronously
        if startup_results:
            # Docker status already known from async startup
            self._apply_docker_status(startup_results.docker_available)
        else:
            # Fallback to async check
            self.check_docker()

        # Load saved layout
        self._load_layout()

        # Set up periodic process cleanup
        self.cleanup_timer = QTimer(self)
        self.cleanup_timer.timeout.connect(self._cleanup_processes)
        self.cleanup_timer.start(10000)  # Clean up every 10 seconds

        # Initialize UI components if available
        self.toast_manager: ToastManager | None = None
        self._init_ui_components()

        # Log startup performance
        if self._startup_time_ms > 0:
            logger.info(f"Application startup completed in {self._startup_time_ms}ms")

    def _init_ui_components(self) -> None:
        """Initialize optional UI components (toast, shortcuts, etc.)."""
        # Toast notification manager
        if UI_COMPONENTS_AVAILABLE:
            self.toast_manager = ToastManager(self)

            # Setup keyboard shortcuts
            self._setup_keyboard_shortcuts()
        else:
            self.toast_manager = None

    def _setup_keyboard_shortcuts(self) -> None:
        """Set up global keyboard shortcuts."""
        # Ctrl+? or F1 for shortcuts overlay
        shortcut_help = QShortcut(QKeySequence("Ctrl+?"), self)
        shortcut_help.activated.connect(self._show_shortcuts_overlay)

        shortcut_f1 = QShortcut(QKeySequence("F1"), self)
        shortcut_f1.activated.connect(self._show_shortcuts_overlay)

        # Ctrl+, for preferences
        shortcut_prefs = QShortcut(QKeySequence("Ctrl+,"), self)
        shortcut_prefs.activated.connect(self._show_preferences)

        # Ctrl+Q to quit
        shortcut_quit = QShortcut(QKeySequence("Ctrl+Q"), self)
        shortcut_quit.activated.connect(self.close)

    def _show_shortcuts_overlay(self) -> None:
        """Show the keyboard shortcuts overlay."""
        if UI_COMPONENTS_AVAILABLE:
            overlay = ShortcutsOverlay(self)
            overlay.show()
            overlay.setFocus()

    def _show_preferences(self) -> None:
        """Show the preferences dialog."""
        if UI_COMPONENTS_AVAILABLE:
            dialog = PreferencesDialog(self)
            dialog.exec()

    def show_toast(self, message: str, toast_type: str = "info") -> None:
        """Show a toast notification.

        Args:
            message: Message to display
            toast_type: Type of toast ("success", "error", "warning", "info")
        """
        if self.toast_manager:
            if toast_type == "success":
                self.toast_manager.show_success(message)
            elif toast_type == "error":
                self.toast_manager.show_error(message)
            elif toast_type == "warning":
                self.toast_manager.show_warning(message)
            else:
                self.toast_manager.show_info(message)

    def _apply_docker_status(self, available: bool) -> None:
        """Apply Docker availability status to UI.

        Args:
            available: Whether Docker is available
        """
        self.docker_available = available
        if available:
            self.lbl_status.setText("● System Ready")
            self.lbl_status.setStyleSheet("color: #30D158; font-weight: bold;")
        else:
            self.lbl_status.setText("● Docker Not Found")
            self.lbl_status.setStyleSheet("color: #FF375F; font-weight: bold;")
        self.update_launch_button()

    def _build_available_models(self) -> None:
        """Collect all known models and auxiliary applications."""

        if self.registry:
            for model in self.registry.get_all_models():
                self.available_models[model.id] = model
                if model.type in ("special_app", "utility", "matlab_app"):
                    self.special_app_lookup[model.id] = model

    def _initialize_model_order(self) -> None:
        """Set a sensible default grid ordering."""

        default_ids: list[str] = []
        if self.registry:
            default_ids.extend([m.id for m in self.registry.get_all_models()[:10]])

        default_ids.extend(
            [
                "urdf_generator",
                "c3d_viewer",
                "matlab_dataset_gui",
                "matlab_golf_gui",
                "matlab_code_analyzer",
            ]
        )

        self.model_order = [
            model_id for model_id in default_ids if model_id in self.available_models
        ]

    def _save_layout(self) -> None:
        """Save the current model layout to configuration file."""
        try:
            # Ensure config directory exists
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)

            layout_data = {
                "model_order": self.model_order,
                "selected_model": self.selected_model,
                "window_geometry": {
                    "x": self.x(),
                    "y": self.y(),
                    "width": self.width(),
                    "height": self.height(),
                },
                "options": {
                    "live_visualization": (
                        self.chk_live.isChecked() if hasattr(self, "chk_live") else True
                    ),
                    "gpu_acceleration": (
                        self.chk_gpu.isChecked() if hasattr(self, "chk_gpu") else False
                    ),
                },
            }

            with open(LAYOUT_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(layout_data, f, indent=2)

            logger.info(f"Layout saved to {LAYOUT_CONFIG_FILE}")

        except Exception as e:
            logger.error(f"Failed to save layout: {e}")

    def _sync_model_cards(self) -> None:
        """Ensure widgets match the current model order."""

        # Remove cards that are no longer selected
        for model_id in list(self.model_cards.keys()):
            if model_id not in self.model_order:
                widget = self.model_cards.pop(model_id)
                widget.setParent(None)
                widget.deleteLater()

        # Create cards for any newly added models
        for model_id in self.model_order:
            if model_id not in self.model_cards:
                model = self._get_model(model_id)
                if model:
                    self.model_cards[model_id] = DraggableModelCard(model, self)

    def _apply_model_selection(self, selected_ids: list[str]) -> None:
        """Apply a new set of selected models from the layout dialog."""

        ordered_selection = [
            model_id for model_id in self.model_order if model_id in selected_ids
        ]

        for model_id in selected_ids:
            if model_id not in ordered_selection and model_id in self.available_models:
                ordered_selection.append(model_id)

        self.model_order = ordered_selection
        self._sync_model_cards()
        self._rebuild_grid()
        self._save_layout()

        if self.selected_model not in self.model_order:
            self.selected_model = self.model_order[0] if self.model_order else None

        # Copilot AI Review Change:
        # Start with the existing model_order filtered to the newly selected IDs so
        # that previously selected models keep their relative positions in the grid.
        # ordered_selection already handled this by iterating self.model_order first.
        # Append any newly selected models (not already in model_order) to the end.
        self.update_launch_button()

    def _get_model(self, model_id: str) -> Any | None:
        """Retrieve a model or application by ID."""

        if model_id in self.available_models:
            return self.available_models[model_id]

        if self.registry:
            return self.registry.get_model(model_id)

        return None

    def center_window(self) -> None:
        """Center the window on the primary screen."""
        screen = self.screen()
        if not screen:
            return

        geometry = self.frameGeometry()
        available_geometry = screen.availableGeometry()
        center_point = available_geometry.center()
        geometry.moveCenter(center_point)
        self.move(geometry.topLeft())

    def _load_layout(self) -> None:
        """Load the saved model layout from configuration file."""
        try:
            if not LAYOUT_CONFIG_FILE.exists():
                logger.info("No saved layout found, using default")
                return

            with open(LAYOUT_CONFIG_FILE, encoding="utf-8") as f:
                layout_data = json.load(f)

            # Restore model order if valid
            saved_order = [
                model_id
                for model_id in layout_data.get("model_order", [])
                if model_id in self.available_models
            ]
            if saved_order:
                self.model_order = saved_order
                self._sync_model_cards()
                self._rebuild_grid()
                logger.info("Model layout restored from saved configuration")

            # Restore window geometry
            geo = layout_data.get("window_geometry", {})
            if geo:
                # Ensure window title bar is visible (y >= 30)
                # And center if it looks weird
                x = geo.get("x", 100)
                y = geo.get("y", 100)
                w = geo.get("width", 1280)
                h = geo.get("height", 800)

                # Clamp Y to avoid being off-screen top
                if y < 30:
                    y = 50

                self.setGeometry(x, y, w, h)
            else:
                self._center_window()

            # Restore options
            options = layout_data.get("options", {})
            if hasattr(self, "chk_live"):
                self.chk_live.setChecked(options.get("live_visualization", True))
            if hasattr(self, "chk_gpu"):
                self.chk_gpu.setChecked(options.get("gpu_acceleration", False))

            # Restore selected model
            saved_selection = layout_data.get("selected_model")
            if saved_selection and saved_selection in self.model_cards:
                self.select_model(saved_selection)

            self._rebuild_grid()  # Use _rebuild_grid as it exists
            logger.info("Layout loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load layout: {e}")
            self._center_window()

    def _center_window(self) -> None:
        """Center the window on the primary screen."""
        screen = QApplication.primaryScreen()
        if screen:
            screen_geo = screen.availableGeometry()
            # Ensure width is treated as int, handling potential Mock objects from tests
            current_width = self.width()
            if hasattr(current_width, "return_value"):  # Handle MagicMock
                current_width = 1280
            width = (
                int(current_width) if isinstance(current_width, int | float) else 1280
            )

            w = width if width > 100 else 1280

            # Ensure height is treated as int, handling potential Mock objects from tests
            current_height = self.height()
            if hasattr(current_height, "return_value"):  # Handle MagicMock
                current_height = 800
            height = (
                int(current_height) if isinstance(current_height, int | float) else 800
            )
            h = height if height > 100 else 800

            # Handle Mock objects for screen geometry
            screen_x = screen_geo.x()
            if hasattr(screen_x, "return_value"):
                screen_x = 0
            screen_x = int(screen_x) if isinstance(screen_x, int | float) else 0

            screen_y = screen_geo.y()
            if hasattr(screen_y, "return_value"):
                screen_y = 0
            screen_y = int(screen_y) if isinstance(screen_y, int | float) else 0

            screen_width = screen_geo.width()
            if hasattr(screen_width, "return_value"):
                screen_width = 1920
            screen_width = (
                int(screen_width) if isinstance(screen_width, int | float) else 1920
            )

            screen_height = screen_geo.height()
            if hasattr(screen_height, "return_value"):
                screen_height = 1080
            screen_height = (
                int(screen_height) if isinstance(screen_height, int | float) else 1080
            )

            x = screen_x + (screen_width - w) // 2
            y = screen_y + (screen_height - h) // 2

            # Ensure not too high
            y = max(y, 50)

            self.setGeometry(x, y, w, h)

    def closeEvent(self, event: QCloseEvent | None) -> None:
        """Handle window close event to save layout.

        UX FIX: Added confirmation dialog when processes are still running
        to prevent accidental termination of simulations.
        """
        # Check for running processes
        running_count = sum(
            1 for p in self.running_processes.values() if p.poll() is None
        )

        if running_count > 0:
            # UX FIX: Confirm before closing with running processes
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                f"There {'is' if running_count == 1 else 'are'} {running_count} "
                f"running process{'es' if running_count > 1 else ''}.\n\n"
                "Closing will terminate all running simulations.\n"
                "Are you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                if event:
                    event.ignore()
                return

        self._save_layout()

        # Stop cleanup timer
        if hasattr(self, "cleanup_timer"):
            self.cleanup_timer.stop()

        # Terminate running processes
        for key, process in list(self.running_processes.items()):
            if process.poll() is None:
                logger.info(f"Terminating child process: {key}")
                try:
                    process.terminate()
                except Exception as e:
                    logger.error(f"Failed to terminate {key}: {e}")

        super().closeEvent(event)

    def init_ui(self) -> None:
        """Initialize the user interface."""
        # Main Widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)

        # --- Top Bar ---
        top_bar = QHBoxLayout()

        # Status Indicator
        self.lbl_status = QLabel("Checking Docker...")
        self.lbl_status.setStyleSheet("color: #aaaaaa; font-weight: bold;")
        top_bar.addWidget(self.lbl_status)
        top_bar.addStretch()

        # Search Bar
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 Search models...")
        self.search_input.setFixedWidth(200)
        self.search_input.setToolTip("Filter models by name or description (Ctrl+F)")
        self.search_input.setAccessibleName("Search models")
        self.search_input.setClearButtonEnabled(True)  # Add clear button
        self.search_input.textChanged.connect(self.update_search_filter)
        top_bar.addWidget(self.search_input)

        # Modify Layout toggle button
        self.btn_modify_layout = QPushButton("🔒 Layout Locked")
        self.btn_modify_layout.setCheckable(True)
        self.btn_modify_layout.setChecked(False)
        self.btn_modify_layout.setToolTip("Toggle to enable/disable tile rearrangement")
        self.btn_modify_layout.clicked.connect(self.toggle_layout_mode)
        self.btn_modify_layout.setStyleSheet("""
            QPushButton {
                background-color: #444444;
                color: #cccccc;
                padding: 8px 16px;
            }
            QPushButton:checked {
                background-color: #007acc;
                color: white;
            }
            """)
        top_bar.addWidget(self.btn_modify_layout)

        self.btn_customize_tiles = QPushButton("🧩 Edit Tiles")
        self.btn_customize_tiles.setEnabled(False)
        self.btn_customize_tiles.setToolTip("Add or remove launcher tiles in edit mode")
        self.btn_customize_tiles.clicked.connect(self.open_layout_manager)
        self.btn_customize_tiles.setCursor(Qt.CursorShape.PointingHandCursor)
        top_bar.addWidget(self.btn_customize_tiles)

        btn_env = QPushButton("⚙️ Environment")
        btn_env.setToolTip("Manage Docker environment and dependencies")
        btn_env.clicked.connect(self.open_environment_manager)
        top_bar.addWidget(btn_env)

        btn_help = QPushButton("📖 Help")
        btn_help.setToolTip("View documentation and user guide")
        btn_help.clicked.connect(self.open_help)
        top_bar.addWidget(btn_help)

        # AI Assistant Button (if available)
        if AI_AVAILABLE:
            self.btn_ai = QPushButton("🤖 AI Assistant")
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
                QPushButton:hover {
                    background-color: #1565c0;
                }
                QPushButton:checked {
                    background-color: #0d47a1;
                }
                """)
            top_bar.addWidget(self.btn_ai)

            # Setup AI Dock Widget (Hidden by default)
            self._setup_ai_dock()

        # Context Help Dock
        self._setup_context_help()

        main_layout.addLayout(top_bar)

        # --- Launcher Grid ---
        # Scroll Area for Grid
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setStyleSheet("QScrollArea { background: transparent; }")

        # Container for Grid
        self.grid_container = QWidget()
        self.grid_container.setStyleSheet("background: transparent;")
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(20)
        self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.scroll.setWidget(self.grid_container)
        main_layout.addWidget(self.scroll, 1)

        # --- Bottom Bar ---
        bottom_bar = QHBoxLayout()

        # Configuration options
        config_group = QHBoxLayout()
        config_group.setSpacing(15)

        self.chk_live = QCheckBox("Live Visualization")
        self.chk_live.setChecked(True)
        self.chk_live.setToolTip("Enable real-time 3D visualization during simulation")
        config_group.addWidget(self.chk_live)

        self.chk_gpu = QCheckBox("GPU Acceleration")
        self.chk_gpu.setChecked(False)
        self.chk_gpu.setToolTip(
            "Use GPU for physics computation (requires supported hardware)"
        )
        config_group.addWidget(self.chk_gpu)

        bottom_bar.addLayout(config_group)
        bottom_bar.addStretch()

        # Launch Button
        self.btn_launch = QPushButton("🚀 Select a Model")
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

        main_layout.addLayout(bottom_bar)

        # Apply dark theme
        self.apply_styles()

        # Add Search Shortcut
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

    def _setup_ai_dock(self) -> None:
        """Set up the AI Assistant dock widget."""
        if not AI_AVAILABLE:
            return

        self.ai_dock = QDockWidget("AI Assistant", self)
        self.ai_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )

        # Create AI Panel
        try:
            self.ai_panel = AIAssistantPanel(self)
            self.ai_dock.setWidget(self.ai_panel)
            self.ai_dock.hide()  # Hidden by default
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.ai_dock)

            # Connect dock visibility change
            self.ai_dock.visibilityChanged.connect(self._on_ai_dock_visibility_changed)

            # Connect settings request
            self.ai_panel.settings_requested.connect(self._open_ai_settings)

        except Exception as e:
            logger.error(f"Failed to initialize AI panel: {e}")
            self.btn_ai.setEnabled(False)
            self.btn_ai.setToolTip(f"AI Assistant unavailable: {e}")

    def toggle_ai_assistant(self, checked: bool) -> None:
        """Toggle the AI Assistant dock visibility.

        Args:
            checked: Whether the button is checked.
        """
        if not AI_AVAILABLE or not hasattr(self, "ai_dock"):
            return

        if checked:
            self.ai_dock.show()
        else:
            self.ai_dock.hide()

    def _on_ai_dock_visibility_changed(self, visible: bool) -> None:
        """Handle AI dock visibility change.

        Args:
            visible: Whether the dock is visible.
        """
        self.btn_ai.setChecked(visible)

    def _open_ai_settings(self) -> None:
        """Open the AI settings dialog."""
        if not AI_AVAILABLE:
            return

        dialog = AISettingsDialog(self)
        if dialog.exec():
            # Reload settings in panel
            if hasattr(self, "ai_panel"):
                # Ideally panel would reload automatically or via signal
                pass

    def _swap_models(self, source_id: str, target_id: str) -> None:
        """Swap two models in the grid layout."""
        if not self.layout_edit_mode:
            return

        try:
            idx1 = self.model_order.index(source_id)
            idx2 = self.model_order.index(target_id)

            # Swap in list
            self.model_order[idx1], self.model_order[idx2] = (
                self.model_order[idx2],
                self.model_order[idx1],
            )

            # Rebuild grid
            self._rebuild_grid()

            # Save layout
            self._save_layout()

        except ValueError:
            pass  # ID not found

    def update_search_filter(self, text: str) -> None:
        """Update the search filter and rebuild grid."""
        self.current_filter_text = text.lower()
        self._rebuild_grid()

    def _rebuild_grid(self) -> None:
        """Rebuild the grid layout based on current model order."""
        # Clean current layout
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)

        # Filter models if search is active
        filtered_order = []
        for model_id in self.model_order:
            if not self.current_filter_text:
                filtered_order.append(model_id)
                continue

            model = self._get_model(model_id)
            if not model:
                continue

            # Search in name, id, and description
            search_content = f"{model.name} {model.id} {model.description}".lower()
            if self.current_filter_text in search_content:
                filtered_order.append(model_id)

        # Get or create widgets
        widgets = []
        for model_id in filtered_order:
            if model_id not in self.model_cards:
                model = self._get_model(model_id)
                if model:
                    self.model_cards[model_id] = DraggableModelCard(model, self)

            if model_id in self.model_cards:
                widgets.append(self.model_cards[model_id])

        # Add to grid
        row = 0
        col = 0
        for widget in widgets:
            self.grid_layout.addWidget(widget, row, col)
            col += 1
            if col >= GRID_COLUMNS:
                col = 0
                row += 1

    def create_model_card(self, model: Any) -> QFrame:
        """Creates a clickable card widget."""
        # Legacy method kept for reference but unused given DraggableModelCard
        # If any old references exist, they should be updated.
        return QFrame()

    def launch_model_direct(self, model_id: str) -> None:
        """Selects and immediately launches the model (for double-click)."""
        self.select_model(model_id)
        # Process events to ensure UI updates before launch
        QApplication.processEvents()
        self.launch_simulation()

    def _launch_urdf_generator(self) -> None:
        """Launch the URDF generator application."""
        from shared.python.constants import URDF_GENERATOR_SCRIPT

        script_path = REPOS_ROOT / URDF_GENERATOR_SCRIPT

        # Check if already running
        if "urdf_generator" in self.running_processes:
            proc = self.running_processes["urdf_generator"]
            if proc.poll() is None:
                self.show_toast("URDF Generator is already running.", "warning")
                # Bring to front logic if possible?
                return

        self.lbl_status.setText("● Launching URDF Generator...")
        self.lbl_status.setStyleSheet("color: #FFD60A;")
        QApplication.processEvents()

        try:
            logger.info("Launching URDF Generator: %s", script_path)

            # Launch detached
            process = secure_popen(
                [sys.executable, str(script_path)],
                cwd=str(REPOS_ROOT),
                creationflags=CREATE_NEW_CONSOLE if os.name == "nt" else 0,
            )

            self.running_processes["urdf_generator"] = process
            self.show_toast("URDF Generator launched.", "success")
            self.lbl_status.setText("● URDF Generator Running")
            self.lbl_status.setStyleSheet("color: #30D158;")

        except Exception as e:
            logger.error(f"Failed to launch URDF Generator: {e}")
            self.show_toast(f"Launch failed: {e}", "error")
            self.lbl_status.setText("● Launch Error")
            self.lbl_status.setStyleSheet("color: #FF375F;")

    def _launch_c3d_viewer(self) -> None:
        """Launch the C3D motion viewer application."""
        # Assuming script is in tools/c3d_viewer/c3d_viewer.py or similar
        # Need to verify path. Based on imports, maybe we can assume it's available.
        c3d_script = REPOS_ROOT / "tools" / "c3d_viewer" / "c3d_viewer.py"

        if not c3d_script.exists():
            # Fallback or check if it's a module
            c3d_script = REPOS_ROOT / "tools" / "c3d_viewer_app.py"

        if not c3d_script.exists():
            self.show_toast("C3D Viewer script not found.", "error")
            return

        if "c3d_viewer" in self.running_processes:
            if self.running_processes["c3d_viewer"].poll() is None:
                self.show_toast("C3D Viewer is already running.", "warning")
                return

        try:
            logger.info("Launching C3D Viewer: %s", c3d_script)
            process = secure_popen(
                [sys.executable, str(c3d_script)],
                cwd=str(c3d_script.parent),
                creationflags=CREATE_NEW_CONSOLE if os.name == "nt" else 0,
            )

            self.running_processes["c3d_viewer"] = process
            self.show_toast("C3D Viewer launched.", "success")

        except Exception as e:
            logger.error(f"Failed to launch C3D Viewer: {e}")
            self.show_toast(f"Launch failed: {e}", "error")

    def _launch_matlab_app(self, app: Any) -> None:
        """Launch a MATLAB-based application using batch mode."""
        # This requires MATLAB to be installed and in PATH
        # Or we use a specific batch script

        app_path = getattr(app, "path", None)
        if not app_path:
            self.show_toast("Invalid MATLAB configuration.", "error")
            return

        self.show_toast(f"Launching MATLAB: {app.name}...", "info")

        try:
            # Construct command
            # matlab -r "run('script.m');"
            cmd = ["matlab", "-nosplash", "-nodesktop", "-r", f"run('{app_path}');"]

            # Check if using batch script wrapper
            if str(app_path).endswith(".bat") or str(app_path).endswith(".sh"):
                cmd = [str(app_path)]

            process = secure_popen(
                cmd,
                cwd=str(Path(app_path).parent),
                creationflags=CREATE_NEW_CONSOLE if os.name == "nt" else 0,
            )

            self.running_processes[app.id] = process
            self.show_toast(f"{app.name} launch initiated.", "success")

        except FileNotFoundError:
            self.show_toast("MATLAB executable not found in PATH.", "error")
        except Exception as e:
            logger.error(f"Failed to launch MATLAB app: {e}")
            self.show_toast(f"Launch failed: {e}", "error")

    def select_model(self, model_id: str) -> None:
        """Select a model and update UI."""
        self.selected_model = model_id

        # Update visual selection state
        for mid, card in self.model_cards.items():
            if mid == model_id:
                card.setStyleSheet("""
                    QFrame#ModelCard {
                        background-color: #383838;
                        border: 2px solid #0A84FF;
                        border-radius: 12px;
                    }
                    """)
            else:
                card.setStyleSheet("""
                    QFrame#ModelCard {
                        background-color: #2D2D2D;
                        border: 1px solid #3A3A3A;
                        border-radius: 12px;
                    }
                    QFrame#ModelCard:hover {
                        background-color: #333333;
                        border: 1px solid #555555;
                    }
                    """)

        # Update launch button
        model = self._get_model(model_id)
        if model:
            self.update_launch_button(model.name)

            # Update Help Context
            if hasattr(self, "context_help"):
                self.context_help.update_context(model_id)

    def update_launch_button(self, model_name: str | None = None) -> None:
        """Update the launch button state."""
        if not self.selected_model:
            self.btn_launch.setText("🚀 Select a Model")
            self.btn_launch.setEnabled(False)
            self.btn_launch.setStyleSheet("""
                QPushButton {
                    background-color: #3a3a3a;
                    color: #888888;
                    border-radius: 6px;
                }
                """)
            return

        name = model_name or self.selected_model

        # Check requirements
        model = self._get_model(self.selected_model)

        # Check Docker dependency
        if model and getattr(model, "requires_docker", False):
            if not self.docker_available:
                self.btn_launch.setText("⚠️ Docker Required")
                self.btn_launch.setStyleSheet("""
                    QPushButton {
                        background-color: #3a3a3a;
                        color: #ff453a;
                        border: 2px solid #ff453a;
                        border-radius: 6px;
                    }
                    """)
                self.btn_launch.setEnabled(False)
                return

        self.btn_launch.setText(f"🚀 Launch {name}")
        self.btn_launch.setEnabled(True)
        self.btn_launch.setStyleSheet("""
            QPushButton {
                background-color: #2da44e;
                color: white;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2c974b;
            }
            """)

    def _get_engine_type(self, model_type: str) -> _EngineType:
        """Map model type to EngineType."""
        # Use lazy loaded types
        EngineType = _EngineType

        if "mujoco" in model_type:
            return EngineType.MUJOCO
        elif "drake" in model_type:
            return EngineType.DRAKE
        elif "pinocchio" in model_type:
            return EngineType.PINOCCHIO
        elif "opensim" in model_type:
            return EngineType.OPENSIM
        elif "myosim" in model_type:
            return EngineType.MYOSIM
        return EngineType.MUJOCO  # Default

    def apply_styles(self) -> None:
        """Apply custom stylesheets."""
        # Global dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1E1E1E;
            }
            QWidget {
                color: #FFFFFF;
                font-family: 'Segoe UI', sans-serif;
            }
            QLineEdit {
                background-color: #252526;
                color: white;
                border: 1px solid #3E3E42;
                border-radius: 4px;
                padding: 6px;
            }
            QLineEdit:focus {
                border: 1px solid #007ACC;
            }
            QScrollArea {
                border: none;
            }
            QPushButton {
                background-color: #333333;
                border: 1px solid #333333;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #3E3E42;
            }
            """)

    def check_docker(self) -> None:
        """Start the docker check thread."""
        logger.info("Checking Docker status...")
        # Since we moved DockerCheckThread to ui_components, we need to import or reimplement?
        # Actually it's part of StartupWorker.
        # But if we need standalone check:
        # We can use a simple QThread here or use AsyncStartupWorker again?
        # Let's verify if DockerCheckThread is in ui_components. It is not.
        # But AsyncStartupWorker does check docker.

        # Let's perform a lightweight threaded check inline since it's simple
        self.docker_checker = DockerCheckThread()
        self.docker_checker.result.connect(self.on_docker_check_complete)
        self.docker_checker.start()

    def on_docker_check_complete(self, available: bool) -> None:
        """Handle docker check result."""
        self._apply_docker_status(available)

    def open_help(self) -> None:
        """Toggle the help drawer."""
        help_dialog = HelpDialog(self)
        help_dialog.exec()

    def open_environment_manager(self) -> None:
        """Open the environment manager dialog."""
        dialog = EnvironmentDialog(self)
        dialog.exec()

    def launch_simulation(self) -> None:
        """Launch the selected simulation."""
        if not self.selected_model:
            return

        model_id = self.selected_model

        # Handle Utility/Special Apps first
        if "urdf_generator" in model_id:
            self._launch_urdf_generator()
            return
        elif "c3d_viewer" in model_id:
            self._launch_c3d_viewer()
            return

        model = self._get_model(model_id)
        if not model:
            self.show_toast("Model configuration not found.", "error")
            return

        if model.type == "matlab_app":
            self._launch_matlab_app(model)
            return

        # Handle Standard Physics Models
        self.lbl_status.setText(f"● Launching {model.name}...")
        self.lbl_status.setStyleSheet("color: #FFD60A;")
        QApplication.processEvents()

        try:
            # Determine launch strategy
            repo_path = getattr(model, "repo_path", None)

            # If path provided, use it
            if repo_path:
                abs_repo_path = REPOS_ROOT / repo_path

                # If custom GUI app provided
                if model.type == "custom_humanoid":
                    self._custom_launch_humanoid(abs_repo_path)
                elif model.type == "custom_dashboard":
                    self._custom_launch_comprehensive(abs_repo_path)
                elif model.type == "drake":
                    self._custom_launch_drake(abs_repo_path)
                elif model.type == "pinocchio":
                    self._custom_launch_pinocchio(abs_repo_path)
                elif model.type == "openpose":
                    self._custom_launch_openpose(abs_repo_path)
                elif model.type == "mjcf" or str(repo_path).endswith(".xml"):
                    self._launch_generic_mjcf(abs_repo_path)
                else:
                    self.show_toast(f"Unknown launch type: {model.type}", "warning")
            else:
                self.show_toast("Model path missing.", "error")

        except Exception as e:
            logger.error(f"Launch failed: {e}")
            self.show_toast(f"Launch Failed: {e}", "error")
            self.lbl_status.setText("● Ready")
            self.lbl_status.setStyleSheet("color: #aaaaaa;")

    def _launch_generic_mjcf(self, path: Path) -> None:
        """Launch generic MJCF file in passive viewer."""
        import mujoco
        import mujoco.viewer

        try:
            m = mujoco.MjModel.from_xml_path(str(path))
            d = mujoco.MjData(m)

            # This blocks the UI thread if run directly!
            # Should run in separate process.
            # For now, we launch a separate python script that opens the viewer?
            # Or use mujoco.viewer.launch_passive()

            # Better approach: Launch robust viewer script
            viewer_script = (
                REPOS_ROOT
                / "engines"
                / "physics_engines"
                / "mujoco"
                / "python"
                / "passive_viewer.py"
            )

            if viewer_script.exists():
                process = secure_popen(
                    [sys.executable, str(viewer_script), str(path)],
                    creationflags=CREATE_NEW_CONSOLE if os.name == "nt" else 0,
                )
                self.running_processes[path.name] = process
                self.show_toast("Launched Passive Viewer", "success")
            else:
                # Fallback if script missing
                self.show_toast(
                    "Viewer script missing, attempting direct launch...", "warning"
                )
                mujoco.viewer.launch(m, d)

        except Exception as e:
            raise RuntimeError(f"Failed to launch MJCF: {e}") from e

    def _custom_launch_humanoid(self, abs_repo_path: Path) -> None:
        """Launch the humanoid GUI directly."""
        cwd = abs_repo_path.parent
        self._launch_script_process("MuJoCo Humanoid", abs_repo_path, cwd)

    def _custom_launch_comprehensive(self, abs_repo_path: Path) -> None:
        """Launch the comprehensive dashboard directly."""
        cwd = abs_repo_path.parent
        self._launch_script_process("Analysis Dashboard", abs_repo_path, cwd)

    def _custom_launch_drake(self, abs_repo_path: Path) -> None:
        """Launch the Drake GUI directly."""
        # Ensure we point to the python script
        cwd = abs_repo_path.parent
        self._launch_script_process("Drake Engine", abs_repo_path, cwd)

    def _custom_launch_pinocchio(self, abs_repo_path: Path) -> None:
        """Launch the Pinocchio GUI directly."""
        cwd = abs_repo_path.parent
        self._launch_script_process("Pinocchio Engine", abs_repo_path, cwd)

    def _custom_launch_opensim(self, abs_repo_path: Path) -> None:
        """Launch the OpenSim GUI directly."""
        cwd = abs_repo_path.parent
        self._launch_script_process("OpenSim Engine", abs_repo_path, cwd)

    def _custom_launch_myosim(self, abs_repo_path: Path) -> None:
        """Launch the MyoSim GUI directly."""
        cwd = abs_repo_path.parent
        self._launch_script_process("MyoSim Engine", abs_repo_path, cwd)

    def _custom_launch_openpose(self, abs_repo_path: Path) -> None:
        """Launch the OpenPose GUI directly."""
        cwd = abs_repo_path.parent
        self._launch_script_process("OpenPose Analytics", abs_repo_path, cwd)

    def _launch_script_process(self, name: str, script_path: Path, cwd: Path) -> None:
        """Helper to launch python script in loose process."""
        try:
            process = secure_popen(
                [sys.executable, str(script_path)],
                cwd=str(cwd),
                creationflags=CREATE_NEW_CONSOLE if os.name == "nt" else 0,
            )
            self.running_processes[name] = process
            self.show_toast(f"{name} Launched", "success")
            self.lbl_status.setText(f"● {name} Running")
            self.lbl_status.setStyleSheet("color: #30D158;")

        except Exception as e:
            raise RuntimeError(f"Failed to launch {name}: {e}") from e

    def open_layout_manager(self) -> None:
        """Open the layout customization dialog."""
        dialog = LayoutManagerDialog(self.available_models, self.model_order, self)
        if dialog.exec():
            selected = dialog.selected_ids()
            self._apply_model_selection(selected)
            self.show_toast("Layout updated", "success")

    def toggle_layout_mode(self, checked: bool) -> None:
        """Toggle tile editing mode."""
        self.layout_edit_mode = checked
        if checked:
            self.btn_modify_layout.setText("🔓 Edit Mode On")
            self.btn_modify_layout.setStyleSheet("""
                QPushButton {
                    background-color: #007acc;
                    color: white;
                    border: 1px solid #0099ff;
                }
                """)
            self.btn_customize_tiles.setEnabled(True)
            self.show_toast("Drag tiles to reorder. Double-click to launch.", "info")
        else:
            self.btn_modify_layout.setText("🔒 Layout Locked")
            self.btn_modify_layout.setStyleSheet("""
                QPushButton {
                    background-color: #444444;
                    color: #cccccc;
                }
                """)
            self.btn_customize_tiles.setEnabled(False)

        # Update all cards to accept/reject drops
        for card in self.model_cards.values():
            card.setAcceptDrops(checked)

    def _setup_context_help(self) -> None:
        """Setup context help dock."""
        self.context_help = ContextHelpDock(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.context_help)
        # Hidden by default, mostly for advanced context
        self.context_help.hide()

        # Shortcut to toggle?
        # Maybe F1 toggles both?

    def _cleanup_processes(self) -> None:
        """Remove finished processes from tracking."""
        finished = []
        for key, proc in self.running_processes.items():
            if proc.poll() is not None:
                finished.append(key)

        for key in finished:
            del self.running_processes[key]

        if not self.running_processes:
            self.lbl_status.setText("● Ready")
            self.lbl_status.setStyleSheet("color: #aaaaaa;")


def main() -> None:
    """Application entry point."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Show splash
    splash = GolfSplashScreen()
    splash.show()

    # Start async loading
    worker = AsyncStartupWorker(REPOS_ROOT)

    # Create main window but don't show yet
    # We pass the worker reference so it can be cleaned up

    def on_startup_finished(results: StartupResults) -> None:
        window = GolfLauncher(results)
        window.show()
        splash.finish(window)

    def on_startup_progress(msg: str, percent: int) -> None:
        splash.show_message(msg, percent)

    worker.progress_signal.connect(on_startup_progress)
    worker.finished_signal.connect(on_startup_finished)

    worker.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
