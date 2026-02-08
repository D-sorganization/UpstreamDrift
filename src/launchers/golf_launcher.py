#!/usr/bin/env python3
"""
Unified Golf Modeling Suite Launcher (PyQt6)
Features:
- Modern UI with rounded corners.
- Modular Docker Environment Management.
- Integrated Help and Documentation.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Add current directory to path so we can import ui_components if needed locally
sys.path.append(str(Path(__file__).parent))

from src.launchers.docker_manager import DockerLauncher
from src.launchers.launcher_layout_manager import (
    LayoutManager,
    compute_centered_geometry,
)
from src.launchers.launcher_model_handlers import ModelHandlerRegistry
from src.launchers.launcher_process_manager import (
    ProcessManager,
    start_vcxsrv,
)
from src.launchers.ui_components import (
    ASSETS_DIR,
    AsyncStartupWorker,
    ContextHelpDock,
    DockerCheckThread,
    DraggableModelCard,
    EnvironmentDialog,
    GolfSplashScreen,
    LayoutManagerDialog,
    StartupResults,
)
from src.launchers.ui_components import HelpDialog as LegacyHelpDialog

# Import new help system (graceful degradation if not available)
try:
    from src.shared.python.help_system import (
        HelpButton,
        HelpDialog,
        TooltipManager,
    )

    HELP_SYSTEM_AVAILABLE = True
except ImportError:
    HELP_SYSTEM_AVAILABLE = False
    HelpDialog = LegacyHelpDialog  # Fallback to legacy
from src.shared.python.logging_config import configure_gui_logging, get_logger
from src.shared.python.subprocess_utils import kill_process_tree

if TYPE_CHECKING:
    from src.shared.python.ui import ToastManager

from PyQt6.QtCore import QEventLoop, Qt, QTimer, QUrl
from PyQt6.QtGui import (
    QAction,
    QCloseEvent,
    QDesktopServices,
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

if sys.platform == "win32":
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
REPOS_ROOT = Path(__file__).parent.parent.parent.resolve()

# VcXsrv paths for Windows X11 support
# VcXsrv functions moved to launcher_process_manager.py
# Imports: is_vcxsrv_running, start_vcxsrv

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
        self.docker_checker: DockerCheckThread | None = None
        self.selected_model: str | None = None
        self.model_cards: dict[str, Any] = {}
        self.model_order: list[str] = []  # Track model order for drag-and-drop
        self.layout_edit_mode = False  # Track if layout editing is enabled

        # Initialize process and model managers (extracted from god class)
        self.process_manager = ProcessManager(REPOS_ROOT)
        self.model_handler_registry = ModelHandlerRegistry()
        self.docker_launcher = DockerLauncher(REPOS_ROOT)
        # Keep backwards-compatible reference
        self.running_processes = self.process_manager.running_processes
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
                self.registry = MR(REPOS_ROOT / "src/config/models.yaml")
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

        # Initialize layout manager (extracted from god class)
        self.layout_manager = LayoutManager(
            config_file=LAYOUT_CONFIG_FILE,
            available_models=self.available_models,
            get_model_func=self._get_model,
            create_card_func=lambda model: DraggableModelCard(model, self),
        )
        # Keep backward-compatible references
        self.model_cards = self.layout_manager.model_cards
        self.model_order = self.layout_manager.model_order

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

    def _get_subprocess_env(self) -> dict[str, str]:
        """Get environment dict with PYTHONPATH set for subprocess launches."""
        env = os.environ.copy()
        pythonpath = str(REPOS_ROOT)
        if "PYTHONPATH" in env:
            pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
        env["PYTHONPATH"] = pythonpath

        # Fix for MuJoCo DLL loading issue on Windows with Python 3.13
        # Setting empty MUJOCO_PLUGIN_PATH disables bundled plugin loading
        # which can fail with "DLL initialization routine failed" errors
        if "MUJOCO_PLUGIN_PATH" not in env:
            env["MUJOCO_PLUGIN_PATH"] = ""

        return env

    def _check_module_dependencies(self, model_type: str) -> tuple[bool, str]:
        """Check if required dependencies for a module type are available.

        Args:
            model_type: The type of model to check dependencies for.

        Returns:
            Tuple of (success, error_message). If success is True, error_message is empty.
        """
        # Map model types to their required imports
        dependency_checks = {
            "custom_humanoid": ("mujoco", "MuJoCo"),
            "custom_dashboard": ("mujoco", "MuJoCo"),
            "mjcf": ("mujoco", "MuJoCo"),
            "drake": ("pydrake", "Drake (pydrake)"),
            "pinocchio": ("pinocchio", "Pinocchio"),
            "opensim": ("opensim", "OpenSim"),
            "myosim": ("myosuite", "MyoSuite"),
        }

        check = dependency_checks.get(model_type)
        if not check:
            return True, ""  # No specific dependency check needed

        module_name, display_name = check

        # Run import check in subprocess to avoid polluting our process
        # Uses the same environment as launch (includes MUJOCO_PLUGIN_PATH fix)
        import_check_code = f"""
import sys
import os
# Ensure project root is in path for src imports
sys.path.insert(0, os.getcwd())
try:
    import {module_name}
    print("OK")
except ImportError as e:
    print(f"ImportError: {{e}}")
except OSError as e:
    print(f"OSError: {{e}}")
except Exception as e:
    print(f"Error: {{type(e).__name__}}: {{e}}")
"""
        try:
            result = subprocess.run(
                [sys.executable, "-c", import_check_code],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(REPOS_ROOT),
                env=self._get_subprocess_env(),
            )
            output = result.stdout.strip()
            if output == "OK":
                return True, ""
            else:
                return False, f"{display_name} dependency check failed:\n{output}"
        except subprocess.TimeoutExpired:
            return False, f"{display_name} dependency check timed out"
        except Exception as e:
            return False, f"Failed to check {display_name} dependencies: {e}"

    def _show_dependency_error(self, model_name: str, error_msg: str) -> None:
        """Show a dialog with dependency error information and suggestions."""
        detailed_msg = f"Cannot launch {model_name}.\n\n{error_msg}\n\n"

        # Add helpful suggestions based on error type
        if "DLL" in error_msg or "OSError" in error_msg:
            detailed_msg += (
                "Suggestions:\n"
                "• Try reinstalling the package: pip install --force-reinstall mujoco\n"
                "• Ensure Visual C++ Redistributable is installed\n"
                "• Check Python version compatibility (some packages may not support Python 3.13 yet)"
            )
        elif "ImportError" in error_msg or "ModuleNotFoundError" in error_msg:
            detailed_msg += (
                "Suggestions:\n"
                "• Install the missing package using pip\n"
                "• Check that you're using the correct Python environment"
            )

        QMessageBox.warning(self, "Dependency Error", detailed_msg)

    def _setup_keyboard_shortcuts(self) -> None:
        """Set up global keyboard shortcuts."""
        # F1 for help dialog (User Manual)
        shortcut_f1 = QShortcut(QKeySequence("F1"), self)
        shortcut_f1.activated.connect(self._show_help_dialog)

        # Ctrl+? for shortcuts overlay
        shortcut_help = QShortcut(QKeySequence("Ctrl+?"), self)
        shortcut_help.activated.connect(self._show_shortcuts_overlay)

        # Ctrl+, for preferences
        shortcut_prefs = QShortcut(QKeySequence("Ctrl+,"), self)
        shortcut_prefs.activated.connect(self._show_preferences)

        # Ctrl+Q to quit
        shortcut_quit = QShortcut(QKeySequence("Ctrl+Q"), self)
        shortcut_quit.activated.connect(self.close)

    def _show_help_dialog(self, topic: str | None = None) -> None:
        """Show the help dialog.

        Args:
            topic: Optional help topic to display initially.
        """
        if HELP_SYSTEM_AVAILABLE:
            dialog = HelpDialog(self, initial_topic=topic)
            dialog.exec()
        else:
            # Fallback to legacy help dialog
            dialog = LegacyHelpDialog(self)
            dialog.exec()

    def _show_about_dialog(self) -> None:
        """Show the About dialog."""
        QMessageBox.about(
            self,
            "About UpstreamDrift",
            "<h2>UpstreamDrift</h2>"
            "<h3>Golf Modeling Suite</h3>"
            "<p><b>Version 2.1</b></p>"
            "<p>Biomechanical Golf Swing Analysis Platform</p>"
            "<hr>"
            "<p>A unified platform for biomechanical golf swing analysis "
            "integrating multiple physics engines including MuJoCo, Drake, "
            "Pinocchio, OpenSim, and MyoSuite.</p>"
            "<p>Copyright 2024-2026 UpstreamDrift Contributors</p>"
            '<p><a href="https://github.com/dieterolson/UpstreamDrift">GitHub Repository</a></p>',
        )

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
        logger.debug("Building available models from registry...")

        if self.registry:
            all_models = self.registry.get_all_models()
            logger.info(f"Registry returned {len(all_models)} models")

            for model in all_models:
                self.available_models[model.id] = model
                logger.debug(f"  Added model: {model.id} ({model.name})")
                if model.type in ("special_app", "utility", "matlab_app"):
                    self.special_app_lookup[model.id] = model

            logger.info(
                f"Built available_models with {len(self.available_models)} entries"
            )
        else:
            logger.warning("No registry available - no models will be loaded")

    def _initialize_model_order(self) -> None:
        """Set a sensible default grid ordering."""
        logger.debug("Initializing model order...")
        self.layout_manager.initialize_model_order()
        # Keep backward-compatible reference in sync
        self.model_order = self.layout_manager.model_order

    def _save_layout(self) -> None:
        """Save the current model layout to configuration file."""
        window_state = {
            "selected_model": self.selected_model,
            "geometry": {
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
                "docker_mode": (
                    self.chk_docker.isChecked()
                    if hasattr(self, "chk_docker")
                    else False
                ),
            },
        }
        self.layout_manager.save_layout(window_state)

    def _sync_model_cards(self) -> None:
        """Ensure widgets match the current model order."""
        self.layout_manager.sync_model_cards()

    def _apply_model_selection(self, selected_ids: list[str]) -> None:
        """Apply a new set of selected models from the layout dialog."""
        self.layout_manager.apply_model_selection(selected_ids)
        self.model_order = self.layout_manager.model_order
        self._sync_model_cards()
        self._rebuild_grid()
        self._save_layout()

        if self.selected_model not in self.model_order:
            self.selected_model = self.model_order[0] if self.model_order else None

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
        layout_data = self.layout_manager.load_layout()

        if layout_data is None:
            self._rebuild_grid()
            return

        # Keep backward-compatible reference in sync
        self.model_order = self.layout_manager.model_order
        self._sync_model_cards()

        # Restore window geometry
        geo = layout_data.get("window_geometry", {})
        if geo:
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
        if hasattr(self, "chk_docker"):
            # Only restore Docker mode if Docker is available
            saved_docker = options.get("docker_mode", False)
            if saved_docker and self.docker_available:
                self.chk_docker.setChecked(True)

        # Restore selected model
        saved_selection = layout_data.get("selected_model")
        if saved_selection and saved_selection in self.model_cards:
            self.select_model(saved_selection)

        self._rebuild_grid()
        logger.info("Layout loaded successfully")

    def _center_window(self) -> None:
        """Center the window on the primary screen."""
        screen = QApplication.primaryScreen()
        if screen:
            screen_geo = screen.availableGeometry()
            # Extract values, handling Mock objects from tests
            screen_x = self._safe_int(screen_geo.x(), 0)
            screen_y = self._safe_int(screen_geo.y(), 0)
            screen_width = self._safe_int(screen_geo.width(), 1920)
            screen_height = self._safe_int(screen_geo.height(), 1080)
            w = max(self._safe_int(self.width(), 1280), 100)
            h = max(self._safe_int(self.height(), 800), 100)

            x, y, w, h = compute_centered_geometry(
                screen_width, screen_height, w, h, screen_x, screen_y
            )
            self.setGeometry(x, y, w, h)

    def _safe_int(self, value: Any, default: int) -> int:
        """Safely convert a value to int, handling Mock objects from tests."""
        if hasattr(value, "return_value"):  # Handle MagicMock
            return default
        return int(value) if isinstance(value, int | float) else default

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
        if hasattr(self, "cleanup_timer") and self.cleanup_timer is not None:
            self.cleanup_timer.stop()
            self.cleanup_timer.deleteLater()
            self.cleanup_timer = None

        # Clean up docker checker thread
        if hasattr(self, "docker_checker") and self.docker_checker is not None:
            try:
                self.docker_checker.result.disconnect(self.on_docker_check_complete)
            except (TypeError, RuntimeError):
                pass
            if self.docker_checker.isRunning():
                self.docker_checker.wait(1000)
            self.docker_checker = None

        # Terminate running processes using kill_process_tree for proper cleanup
        for key, process in list(self.running_processes.items()):
            if process.poll() is None:
                logger.info(f"Terminating child process: {key}")
                try:
                    # Use kill_process_tree to ensure terminal and all children close
                    if not kill_process_tree(process.pid):
                        # Fallback to direct termination
                        process.terminate()
                except Exception as e:
                    logger.error(f"Failed to terminate {key}: {e}")

        super().closeEvent(event)

    def _init_overlay(self) -> None:
        """Initialize the screen overlay."""
        try:
            from src.shared.python.ui.overlay import OverlayWidget

            self.overlay = OverlayWidget(self)
            self.overlay.hide()
        except ImportError:
            logger.warning("OverlayWidget could not be imported.")

    def _toggle_overlay(self) -> None:
        """Toggle the screen overlay."""
        if hasattr(self, "overlay"):
            self.overlay.toggle()

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

        # --- Launcher Grid ---
        self._setup_grid_area(main_layout)

        # --- Bottom Bar ---
        bottom_bar = self._setup_bottom_bar()
        main_layout.addLayout(bottom_bar)

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

        # Tools Menu
        tools_menu = menubar.addMenu("&Tools")

        action_env = QAction("&Environment Manager...", self)
        action_env.triggered.connect(self.open_environment_manager)
        tools_menu.addAction(action_env)

        action_diag = QAction("&Diagnostics...", self)
        action_diag.triggered.connect(self.open_diagnostics)
        tools_menu.addAction(action_diag)

        # Help Menu
        help_menu = menubar.addMenu("&Help")

        action_manual = QAction("&User Manual", self)
        action_manual.setShortcut("F1")
        action_manual.triggered.connect(lambda: self._show_help_dialog())
        help_menu.addAction(action_manual)

        # Add topic-specific help items
        help_menu.addSeparator()

        action_help_engines = QAction("Engine &Selection Guide", self)
        action_help_engines.triggered.connect(
            lambda: self._show_help_dialog("engine_selection")
        )
        help_menu.addAction(action_help_engines)

        action_help_sim = QAction("Simulation &Controls", self)
        action_help_sim.triggered.connect(
            lambda: self._show_help_dialog("simulation_controls")
        )
        help_menu.addAction(action_help_sim)

        action_help_mocap = QAction("&Motion Capture", self)
        action_help_mocap.triggered.connect(
            lambda: self._show_help_dialog("motion_capture")
        )
        help_menu.addAction(action_help_mocap)

        action_help_viz = QAction("&Visualization", self)
        action_help_viz.triggered.connect(
            lambda: self._show_help_dialog("visualization")
        )
        help_menu.addAction(action_help_viz)

        action_help_analysis = QAction("&Analysis Tools", self)
        action_help_analysis.triggered.connect(
            lambda: self._show_help_dialog("analysis_tools")
        )
        help_menu.addAction(action_help_analysis)

        help_menu.addSeparator()

        action_shortcuts = QAction("&Keyboard Shortcuts...", self)
        action_shortcuts.setShortcut("Ctrl+?")
        action_shortcuts.triggered.connect(self._show_shortcuts_overlay)
        help_menu.addAction(action_shortcuts)

        help_menu.addSeparator()

        action_about = QAction("&About UpstreamDrift", self)
        action_about.triggered.connect(self._show_about_dialog)
        help_menu.addAction(action_about)

    def _toggle_layout_mode_from_menu(self, checked: bool) -> None:
        """Toggle layout edit mode from menu action.

        Args:
            checked: Whether the menu item is checked.
        """
        if hasattr(self, "btn_modify_layout"):
            self.btn_modify_layout.setChecked(checked)
            self.toggle_layout_mode(checked)

    def _toggle_context_help(self, checked: bool) -> None:
        """Toggle the context help panel visibility.

        Args:
            checked: Whether to show the panel.
        """
        if hasattr(self, "context_help"):
            if checked:
                self.context_help.show()
            else:
                self.context_help.hide()

    def _setup_top_bar(self) -> QHBoxLayout:
        """Set up the top tool bar."""
        top_bar = QHBoxLayout()

        # Status Indicator
        self.lbl_status = QLabel("Checking Docker...")
        self.lbl_status.setStyleSheet("color: #aaaaaa; font-weight: bold;")
        top_bar.addWidget(self.lbl_status)

        # Configuration options
        self.chk_live = QCheckBox("Live Viz")
        self.chk_live.setChecked(True)
        self.chk_live.setToolTip("Enable real-time 3D visualization during simulation")
        top_bar.addWidget(self.chk_live)

        self.chk_gpu = QCheckBox("GPU")
        self.chk_gpu.setChecked(False)
        self.chk_gpu.setToolTip(
            "Use GPU for physics computation (requires supported hardware)"
        )
        top_bar.addWidget(self.chk_gpu)

        # Overlay Toggle
        overlay_btn = QPushButton("Overlay")
        overlay_btn.setCheckable(True)
        overlay_btn.clicked.connect(self._toggle_overlay)
        overlay_btn.setStyleSheet("""
            QPushButton {
                background-color: #444; color: white; border: none;
                padding: 5px 10px; border-radius: 4px;
            }
            QPushButton:checked {
                background-color: #007ACC;
            }
            QPushButton:hover { background-color: #555; }
        """)
        top_bar.addWidget(overlay_btn)

        # Docker mode toggle
        self.chk_docker = QCheckBox("Docker")
        self.chk_docker.setChecked(False)
        self.chk_docker.setToolTip(
            "Run physics engines in Docker containers (requires Docker Desktop)\n"
            "Use this for engines not installed locally (Drake, Pinocchio, etc.)"
        )
        self.chk_docker.stateChanged.connect(self._on_docker_mode_changed)
        top_bar.addWidget(self.chk_docker)

        # WSL mode toggle - for full Pinocchio/Drake/Crocoddyl support
        self.chk_wsl = QCheckBox("WSL")
        self.chk_wsl.setChecked(False)
        self.chk_wsl.setToolTip(
            "Run in WSL2 Ubuntu environment (full Pinocchio/Drake/Crocoddyl support)\n"
            "Recommended for advanced robotics features not available on Windows"
        )
        self.chk_wsl.stateChanged.connect(self._on_wsl_mode_changed)
        top_bar.addWidget(self.chk_wsl)

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
        self.search_input.setFixedWidth(200)
        self.search_input.setToolTip("Filter models by name or description (Ctrl+F)")
        self.search_input.setAccessibleName("Search models")
        self.search_input.setClearButtonEnabled(True)  # Add clear button
        self.search_input.textChanged.connect(self.update_search_filter)
        top_bar.addWidget(self.search_input)

        # Modify Layout toggle button
        self.btn_modify_layout = QPushButton("Layout: Locked")
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

        self.btn_customize_tiles = QPushButton("Edit Tiles")
        self.btn_customize_tiles.setEnabled(False)
        self.btn_customize_tiles.setToolTip("Add or remove launcher tiles in edit mode")
        self.btn_customize_tiles.clicked.connect(self.open_layout_manager)
        self.btn_customize_tiles.setCursor(Qt.CursorShape.PointingHandCursor)
        top_bar.addWidget(self.btn_customize_tiles)

        btn_env = QPushButton("Environment")
        btn_env.setToolTip("Manage Docker environment and dependencies")
        btn_env.clicked.connect(self.open_environment_manager)
        top_bar.addWidget(btn_env)

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

        # Add help button for engine selection (next to grid)
        if HELP_SYSTEM_AVAILABLE:
            btn_engine_help = HelpButton(
                "engine_selection", "Help with engine selection", self
            )
            top_bar.addWidget(btn_engine_help)

        btn_diagnostics = QPushButton("Diagnostics")
        btn_diagnostics.setToolTip("Run diagnostics to troubleshoot launcher issues")
        btn_diagnostics.setStyleSheet("""
            QPushButton {
                background-color: #6f42c1;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #7c4dff;
            }
        """)
        btn_diagnostics.clicked.connect(self.open_diagnostics)
        top_bar.addWidget(btn_diagnostics)

        btn_bug = QPushButton("Report Bug")
        btn_bug.setStyleSheet("""
            QPushButton {
                background-color: #d32f2f;
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #b71c1c;
            }
        """)
        btn_bug.setToolTip("Report a bug via email")
        btn_bug.clicked.connect(self._report_bug)
        top_bar.addWidget(btn_bug)

        # AI Assistant Button (if available)
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

        # Register enhanced tooltips for help system
        if HELP_SYSTEM_AVAILABLE:
            TooltipManager.register_tooltip(
                self.chk_live,
                "Live Visualization",
                "Enable real-time 3D visualization during simulation. "
                "Disable for faster computation when visuals aren't needed.",
                "visualization",
            )
            TooltipManager.register_tooltip(
                self.chk_gpu,
                "GPU Acceleration",
                "Use GPU for physics computation when available. "
                "Provides faster simulation for compatible engines.",
                "engine_selection",
            )
            TooltipManager.register_tooltip(
                self.chk_docker,
                "Docker Mode",
                "Run physics engines in Docker containers. "
                "Useful for engines not installed locally (Drake, Pinocchio).",
                "engine_selection",
            )
            TooltipManager.register_tooltip(
                self.chk_wsl,
                "WSL Mode",
                "Run in WSL2 Ubuntu environment for full Linux engine support. "
                "Recommended for advanced features not available on Windows.",
                "engine_selection",
            )

        return top_bar

    def _setup_grid_area(self, layout: QVBoxLayout) -> None:
        """Set up the scrollable grid area."""
        # Scroll Area for Grid
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setStyleSheet("QScrollArea { background: transparent; }")

        # Container for Grid
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

        # Launch Button
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

    def _on_docker_mode_changed(self, state: int) -> None:
        """Handle Docker mode toggle change.

        Args:
            state: Qt checkbox state (0=unchecked, 2=checked)
        """
        use_docker = state == 2
        if use_docker:
            # Disable WSL mode if Docker is enabled (mutually exclusive)
            if hasattr(self, "chk_wsl") and self.chk_wsl.isChecked():
                self.chk_wsl.setChecked(False)

            if not self.docker_available:
                QMessageBox.warning(
                    self,
                    "Docker Not Available",
                    "Docker Desktop is not running or not installed.\n\n"
                    "Please start Docker Desktop and try again.\n\n"
                    "The launcher will continue in local mode.",
                )
                self.chk_docker.setChecked(False)
                return

        if use_docker:
            logger.info("Docker mode enabled")
            # Only show toast if UI is fully initialized
            if hasattr(self, "toast_manager") and self.toast_manager:
                self.show_toast(
                    "Docker mode enabled - engines will run in containers", "info"
                )
        else:
            logger.info("Docker mode disabled")
            if hasattr(self, "toast_manager") and self.toast_manager:
                self.show_toast("Local mode - engines will run on host system", "info")

        # Update UI status
        self.update_execution_status()

        # Update launch button text if a model is selected
        if hasattr(self, "btn_launch"):
            self.update_launch_button()

    def _on_wsl_mode_changed(self, state: int) -> None:
        """Handle WSL mode toggle change.

        Args:
            state: Qt checkbox state (0=unchecked, 2=checked)
        """
        use_wsl = state == 2

        if use_wsl:
            # Disable Docker mode if WSL is enabled (mutually exclusive)
            if hasattr(self, "chk_docker") and self.chk_docker.isChecked():
                self.chk_docker.setChecked(False)

            # Check if WSL is available
            try:
                # wsl.exe outputs UTF-16LE on Windows
                result = subprocess.run(
                    ["wsl", "--list", "--quiet"],
                    capture_output=True,
                    timeout=5,
                    creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
                )

                # Try decoding as UTF-16LE first (standard for wsl.exe), then fallback
                try:
                    output = result.stdout.decode("utf-16-le")
                except UnicodeError:
                    output = result.stdout.decode("utf-8", errors="ignore")

                if result.returncode != 0 or "Ubuntu" not in output:
                    raise RuntimeError("Ubuntu not found in WSL")
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "WSL Not Available",
                    f"WSL2 with Ubuntu is not available.\n\n"
                    f"Error: {e}\n\n"
                    "Please install WSL2 and Ubuntu:\n"
                    "  wsl --install -d Ubuntu-22.04",
                )
                self.chk_wsl.setChecked(False)
                return

            logger.info("WSL mode enabled")
            if hasattr(self, "toast_manager") and self.toast_manager:
                self.show_toast(
                    "WSL mode - full Pinocchio/Drake/Crocoddyl support", "info"
                )
        else:
            logger.info("WSL mode disabled")
            if hasattr(self, "toast_manager") and self.toast_manager:
                self.show_toast("Local Windows mode", "info")

        # Update UI status
        self.update_execution_status()

        # Update launch button text if a model is selected
        if hasattr(self, "btn_launch"):
            self.update_launch_button()

    def _report_bug(self) -> None:
        """Open default mail client to report a bug."""
        subject = "Bug Report: Golf Modeling Suite"
        body = "Please describe the issue you encountered:\n\n"

        # Safely encode
        from urllib.parse import quote

        # Replace with actual support email if known, otherwise placeholder
        email = "support@golfmodelingsuite.com"
        mailto_url = f"mailto:{email}?subject={quote(subject)}&body={quote(body)}"

        QDesktopServices.openUrl(QUrl(mailto_url))

    def update_execution_status(self) -> None:
        """Update the execution mode label based on current settings."""
        if not hasattr(self, "lbl_execution_mode"):
            return

        if hasattr(self, "chk_wsl") and self.chk_wsl.isChecked():
            self.lbl_execution_mode.setText("Mode: WSL (Ubuntu)")
            self.lbl_execution_mode.setStyleSheet(
                "color: #30D158; font-weight: bold; margin-left: 10px;"
            )
        elif hasattr(self, "chk_docker") and self.chk_docker.isChecked():
            self.lbl_execution_mode.setText("Mode: Docker Container")
            self.lbl_execution_mode.setStyleSheet(
                "color: #30D158; font-weight: bold; margin-left: 10px;"
            )
        else:
            self.lbl_execution_mode.setText("Mode: Local (Windows)")
            self.lbl_execution_mode.setStyleSheet(
                "color: #FFD60A; font-weight: bold; margin-left: 10px;"
            )

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
        if self.layout_manager.swap_models(source_id, target_id):
            self.model_order = self.layout_manager.model_order
            self._rebuild_grid()
            self._save_layout()

    def update_search_filter(self, text: str) -> None:
        """Update the search filter and rebuild grid."""
        self.layout_manager.update_search_filter(text)
        self._rebuild_grid()

    def _rebuild_grid(self) -> None:
        """Rebuild the grid layout based on current model order."""
        self.layout_manager.rebuild_grid(self.grid_layout)

    def create_model_card(self, model: Any) -> QFrame:
        """Creates a clickable card widget."""
        return QFrame()

    def launch_model_direct(self, model_id: str) -> None:
        """Selects and immediately launches the model (for double-click)."""
        self.select_model(model_id)
        # Process events to ensure UI updates before launch
        # Use ExcludeUserInputEvents to prevent re-entrancy from user clicks
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
        self.launch_simulation()

    def _launch_urdf_generator(self) -> None:
        """Launch the URDF generator / Model Explorer application."""
        from src.shared.python.constants import URDF_GENERATOR_SCRIPT

        script_path = REPOS_ROOT / URDF_GENERATOR_SCRIPT

        # Check if already running
        if "urdf_generator" in self.running_processes:
            proc = self.running_processes["urdf_generator"]
            if proc.poll() is None:
                self.show_toast("URDF Generator is already running.", "warning")
                # Bring to front logic if possible?
                return

        self.lbl_status.setText("> Launching URDF Generator...")
        self.lbl_status.setStyleSheet("color: #FFD60A;")
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

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
            self.lbl_status.setText("> URDF Generator Running")
            self.lbl_status.setStyleSheet("color: #30D158;")

        except Exception as e:
            logger.error(f"Failed to launch URDF Generator: {e}")
            self.show_toast(f"Launch failed: {e}", "error")
            self.lbl_status.setText("! Launch Error")
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

    def _launch_shot_tracer(self) -> None:
        """Launch the Shot Tracer ball flight visualization."""
        shot_tracer_script = REPOS_ROOT / "src" / "launchers" / "shot_tracer.py"

        if not shot_tracer_script.exists():
            self.show_toast("Shot Tracer script not found.", "error")
            return

        if "shot_tracer" in self.running_processes:
            if self.running_processes["shot_tracer"].poll() is None:
                self.show_toast("Shot Tracer is already running.", "warning")
                return

        try:
            logger.info("Launching Shot Tracer: %s", shot_tracer_script)
            process = secure_popen(
                [sys.executable, str(shot_tracer_script)],
                cwd=str(REPOS_ROOT),  # Run from project root for imports
                env=self._get_subprocess_env(),
                creationflags=CREATE_NEW_CONSOLE if os.name == "nt" else 0,
            )

            self.running_processes["shot_tracer"] = process
            self.show_toast("Shot Tracer launched.", "success")

        except Exception as e:
            logger.error(f"Failed to launch Shot Tracer: {e}")
            self.show_toast(f"Launch failed: {e}", "error")

    def _launch_matlab_app(self, app: Any) -> None:
        """Launch a MATLAB-based application with proper desktop GUI.

        This method launches MATLAB with its full desktop interface rather than
        command-line mode, making it easier for users to interact with and close.
        """
        # This requires MATLAB to be installed and in PATH
        app_path = getattr(app, "path", None)
        if not app_path:
            self.show_toast("Invalid MATLAB configuration.", "error")
            return

        self.show_toast(f"Launching MATLAB: {app.name}...", "info")

        try:
            abs_path = REPOS_ROOT / app_path
            path_str = str(abs_path).replace("\\", "/")  # MATLAB uses forward slashes

            # Check if using batch script wrapper
            if str(app_path).endswith(".bat") or str(app_path).endswith(".sh"):
                cmd = [str(abs_path)]
                process = secure_popen(
                    cmd,
                    cwd=str(abs_path.parent),
                    creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            else:
                # Determine the appropriate MATLAB command based on file type
                if str(app_path).endswith(".slx"):
                    # Simulink model - use open_system
                    matlab_cmd = f"open_system('{path_str}')"
                elif str(app_path).endswith(".m"):
                    # MATLAB script - use run with proper path
                    matlab_cmd = f"cd('{str(abs_path.parent).replace(chr(92), '/')}'); run('{abs_path.name}')"
                else:
                    # Generic file - try to open
                    matlab_cmd = f"open('{path_str}')"

                # Launch MATLAB with desktop (no -nodesktop flag) so user can close it normally
                # Use -nosplash to speed up startup, but keep the desktop GUI
                cmd = ["matlab", "-nosplash", "-r", matlab_cmd]

                # Use CREATE_NO_WINDOW to hide the launcher's console window
                # MATLAB will open its own proper GUI window
                process = secure_popen(
                    cmd,
                    cwd=str(abs_path.parent),
                    creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
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
            self.btn_launch.setText("Select a Model")
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
                self.btn_launch.setText("! Docker Required")
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

        self.btn_launch.setText(f"Launch {name} >")
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
            QMenuBar {
                background-color: #252526;
                color: #CCCCCC;
                border-bottom: 1px solid #3E3E42;
                padding: 2px;
            }
            QMenuBar::item {
                padding: 6px 12px;
                background: transparent;
            }
            QMenuBar::item:selected {
                background-color: #094771;
            }
            QMenu {
                background-color: #252526;
                color: #CCCCCC;
                border: 1px solid #3E3E42;
            }
            QMenu::item {
                padding: 8px 24px;
            }
            QMenu::item:selected {
                background-color: #094771;
            }
            QMenu::separator {
                height: 1px;
                background: #3E3E42;
                margin: 4px 8px;
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
        # Clean up any existing docker checker thread
        if hasattr(self, "docker_checker") and self.docker_checker is not None:
            if self.docker_checker.isRunning():
                self.docker_checker.wait(1000)  # Wait up to 1 second
            try:
                self.docker_checker.result.disconnect(self.on_docker_check_complete)
            except (TypeError, RuntimeError):
                pass

        self.docker_checker = DockerCheckThread()
        self.docker_checker.result.connect(self.on_docker_check_complete)
        self.docker_checker.start()

    def on_docker_check_complete(self, available: bool) -> None:
        """Handle docker check result."""
        self._apply_docker_status(available)

    def open_help(self) -> None:
        """Open the help dialog.

        Note: This method is kept for backward compatibility.
        Use _show_help_dialog() for new code.
        """
        self._show_help_dialog()

    def open_diagnostics(self) -> None:
        """Open the diagnostics dialog to troubleshoot launcher issues."""
        try:
            from src.launchers.launcher_diagnostics import LauncherDiagnostics

            diag = LauncherDiagnostics()
            results = diag.run_all_checks()

            # Add runtime state information
            results["runtime_state"] = {
                "available_models_count": len(self.available_models),
                "available_model_ids": list(self.available_models.keys()),
                "model_order_count": len(self.model_order),
                "model_order": self.model_order,
                "model_cards_count": len(self.model_cards),
                "selected_model": self.selected_model,
                "docker_available": self.docker_available,
                "registry_loaded": self.registry is not None,
            }

            # Create dialog
            dialog = QMessageBox(self)
            dialog.setWindowTitle("Launcher Diagnostics")
            dialog.setIcon(QMessageBox.Icon.Information)

            summary = results["summary"]
            status_emoji = "✅" if summary["status"] == "healthy" else "⚠️"

            text = f"""
{status_emoji} Status: {summary["status"].upper()}

Checks: {summary["passed"]} passed, {summary["failed"]} failed, {summary["warnings"]} warnings

Runtime State:
• Available models: {results["runtime_state"]["available_models_count"]}
• Model order (tiles): {results["runtime_state"]["model_order_count"]}
• Model cards: {results["runtime_state"]["model_cards_count"]}
• Registry loaded: {results["runtime_state"]["registry_loaded"]}

Expected tiles: {summary["expected_tiles"]}
"""

            # Add check details
            for check in results["checks"]:
                if check["status"] == "fail":
                    text += f"\n❌ {check['name']}: {check['message']}"
                elif check["status"] == "warning":
                    text += f"\n⚠️ {check['name']}: {check['message']}"

            # Add recommendations
            text += "\n\nRecommendations:\n"
            for rec in results["recommendations"][:5]:
                text += f"• {rec}\n"

            dialog.setText(text)

            # Add reset button if layout issue detected
            reset_btn = dialog.addButton(
                "Reset Layout", QMessageBox.ButtonRole.ActionRole
            )
            dialog.addButton(QMessageBox.StandardButton.Ok)

            dialog.exec()

            # Handle reset button
            if dialog.clickedButton() == reset_btn:
                self._reset_layout_to_defaults()

        except ImportError as e:
            QMessageBox.warning(
                self,
                "Diagnostics Unavailable",
                f"Could not load diagnostics module: {e}",
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Diagnostics Error",
                f"Error running diagnostics: {e}",
            )

    def _reset_layout_to_defaults(self) -> None:
        """Reset layout configuration to show all 8 default tiles."""
        from pathlib import Path

        config_file = Path.home() / ".golf_modeling_suite" / "launcher_layout.json"

        try:
            if config_file.exists():
                # Backup existing config
                backup_path = config_file.with_suffix(".json.bak")
                config_file.rename(backup_path)
                logger.info(f"Backed up existing config to {backup_path}")

            # Re-initialize model order with all defaults
            self._initialize_model_order()
            self._sync_model_cards()
            self._rebuild_grid()

            self.show_toast("Layout reset to defaults with all 8 tiles", "success")
            logger.info("Layout reset to defaults")

        except Exception as e:
            logger.error(f"Failed to reset layout: {e}")
            self.show_toast(f"Failed to reset layout: {e}", "error")

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
        if "urdf_generator" in model_id or "model_explorer" in model_id:
            self._launch_urdf_generator()
            return
        elif "c3d_viewer" in model_id:
            self._launch_c3d_viewer()
            return
        elif "shot_tracer" in model_id:
            self._launch_shot_tracer()
            return

        model = self._get_model(model_id)
        if not model:
            self.show_toast("Model configuration not found.", "error")
            return

        if model.type == "matlab_app":
            self._launch_matlab_app(model)
            return

        # Handle Standard Physics Models
        use_docker = hasattr(self, "chk_docker") and self.chk_docker.isChecked()

        # Check if Docker mode is enabled
        if use_docker and self.docker_available:
            self.lbl_status.setText(f"> Launching {model.name} in Docker...")
            self.lbl_status.setStyleSheet("color: #64b5f6;")
            QApplication.processEvents(
                QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
            )

            try:
                repo_path = getattr(model, "path", None)
                if repo_path:
                    self._launch_docker_container(model, REPOS_ROOT / repo_path)
                else:
                    self.show_toast("Model path missing for Docker launch.", "error")
                return
            except Exception as e:
                logger.error(f"Docker launch failed: {e}")
                self.show_toast(f"Docker Launch Failed: {e}", "error")
                self.lbl_status.setText("> Ready")
                self.lbl_status.setStyleSheet("color: #aaaaaa;")
                return

        # Check if WSL mode is enabled
        use_wsl = hasattr(self, "chk_wsl") and self.chk_wsl.isChecked()

        # Local mode - check dependencies (only if not using WSL)
        if not use_wsl:
            self.lbl_status.setText(f"> Checking {model.name} dependencies...")
            self.lbl_status.setStyleSheet("color: #FFD60A;")
            QApplication.processEvents(
                QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
            )

            # Check dependencies before launching
            deps_ok, deps_error = self._check_module_dependencies(model.type)
            if not deps_ok:
                # Offer Docker as alternative if available
                if self.docker_available:
                    response = QMessageBox.question(
                        self,
                        "Local Dependencies Missing",
                        f"{deps_error}\n\n"
                        "Would you like to try launching in Docker mode instead?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    )
                    if response == QMessageBox.StandardButton.Yes:
                        self.chk_docker.setChecked(True)
                        self.launch_simulation()  # Retry with Docker
                        return
                self._show_dependency_error(model.name, deps_error)
                self.lbl_status.setText("● Dependency Error")
                self.lbl_status.setStyleSheet("color: #FF375F;")
                return

        self.lbl_status.setText(f"> Launching {model.name}...")
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

        try:
            # Determine launch strategy
            repo_path = getattr(model, "path", None)

            # If path provided, use it
            if repo_path:
                abs_repo_path = REPOS_ROOT / repo_path

                # Try to use the handler registry first (cleaner, extensible approach)
                handler = self.model_handler_registry.get_handler(model.type)
                if handler:
                    success = handler.launch(model, REPOS_ROOT, self.process_manager)
                    if success:
                        self.show_toast(f"{model.name} Launched", "success")
                        self.lbl_status.setText(f"● {model.name} Running")
                        self.lbl_status.setStyleSheet("color: #30D158;")
                    else:
                        self.show_toast(f"Failed to launch {model.name}", "error")
                        self.lbl_status.setText("● Launch Error")
                        self.lbl_status.setStyleSheet("color: #FF375F;")
                # Fallback for MJCF and unknown types
                elif model.type == "mjcf" or str(repo_path).endswith(".xml"):
                    self._launch_generic_mjcf(abs_repo_path)
                else:
                    self.show_toast(f"Unknown launch type: {model.type}", "warning")
            else:
                self.show_toast("Model path missing.", "error")

        except Exception as e:
            logger.error(f"Launch failed: {e}")
            self.show_toast(f"Launch Failed: {e}", "error")
            self.lbl_status.setText("> Ready")
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

    def _launch_docker_container(self, model: Any, repo_path: Path) -> None:
        """Launch the model in a Docker container.

        Delegates to DockerLauncher for container orchestration while
        handling UI feedback (prompts, status updates, error dialogs).
        """
        try:
            # Auto-start VcXsrv on Windows for GUI support
            if os.name == "nt":
                if not start_vcxsrv():
                    response = QMessageBox.question(
                        self,
                        "X Server Not Available",
                        "VcXsrv X server is not running and could not be started.\n\n"
                        "Docker GUI apps require an X server.\n\n"
                        "Install VcXsrv from: https://vcxsrv.com\n\n"
                        "Continue anyway?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    )
                    if response != QMessageBox.StandardButton.Yes:
                        return

            # Check if Docker image exists
            if not self.docker_launcher.check_image_exists():
                QMessageBox.warning(
                    self,
                    "Docker Image Not Found",
                    f"The Docker image '{self.docker_launcher.image_name}' is not available.\n\n"
                    "Build it first using:\n"
                    "  docker build -t robotics_env .\n\n"
                    "Or use the Environment dialog to build.",
                )
                return

            # Launch container via DockerLauncher
            use_gpu = hasattr(self, "chk_gpu") and self.chk_gpu.isChecked()
            process = self.docker_launcher.launch_container(
                model_type=model.type,
                model_name=model.name,
                repo_path=repo_path,
                use_gpu=use_gpu,
            )

            if process:
                self.running_processes[model.name] = process
                self.show_toast(f"{model.name} Launched (Docker)", "success")
                self.lbl_status.setText(f"● {model.name} Running (Docker)")
                self.lbl_status.setStyleSheet("color: #30D158;")
            else:
                self.lbl_status.setText("● Docker Error")
                self.lbl_status.setStyleSheet("color: #FF375F;")
                QMessageBox.critical(
                    self,
                    "Docker Launch Error",
                    f"Failed to launch {model.name} in Docker",
                )

        except Exception as e:
            logger.error(f"Failed to launch Docker container: {e}")
            QMessageBox.critical(
                self,
                "Docker Launch Error",
                f"Failed to launch {model.name} in Docker:\n\n{e}",
            )
            self.lbl_status.setText("● Docker Error")
            self.lbl_status.setStyleSheet("color: #FF375F;")

    def _launch_script_process(self, name: str, script_path: Path, cwd: Path) -> None:
        """Helper to launch python script with error visibility.

        On Windows, uses cmd /k to keep the terminal open if the script crashes,
        allowing users to see the error message.

        If WSL mode is enabled, launches the script in WSL2 Ubuntu environment.

        Delegates to ProcessManager for the actual subprocess handling.
        """
        # Check if WSL mode is enabled
        use_wsl = hasattr(self, "chk_wsl") and self.chk_wsl.isChecked()

        if use_wsl:
            success = self.process_manager.launch_in_wsl(str(script_path))
            if success:
                self.lbl_status.setText(f"● {name} Running (WSL)")
                self.lbl_status.setStyleSheet("color: #30D158;")
                self.show_toast(f"{name} Launched in WSL", "success")
            else:
                QMessageBox.critical(
                    self, "Launch Error", f"Failed to launch {name} in WSL"
                )
            return

        # Delegate to ProcessManager with keep_terminal_open=True for error visibility
        process = self.process_manager.launch_script(
            name, script_path, cwd, keep_terminal_open=True
        )

        if process:
            self.show_toast(f"{name} Launched", "success")
            self.lbl_status.setText(f"● {name} Running")
            self.lbl_status.setStyleSheet("color: #30D158;")
        else:
            QMessageBox.critical(self, "Launch Error", f"Failed to launch {name}")

    def _launch_module_process(self, name: str, module_name: str, cwd: Path) -> None:
        """Helper to launch python module with error visibility.

        Similar to _launch_script_process but uses -m to run a module.
        If WSL mode is enabled, launches in WSL2 Ubuntu environment.

        Delegates to ProcessManager for the actual subprocess handling.
        """
        # Check if WSL mode is enabled
        use_wsl = hasattr(self, "chk_wsl") and self.chk_wsl.isChecked()

        if use_wsl:
            # For WSL, we run the module using python -m
            success = self.process_manager.launch_module_in_wsl(module_name, cwd)
            if success:
                self.lbl_status.setText(f"● {name} Running (WSL)")
                self.lbl_status.setStyleSheet("color: #30D158;")
                self.show_toast(f"{name} Launched in WSL", "success")
            else:
                QMessageBox.critical(
                    self, "Launch Error", f"Failed to launch {name} in WSL"
                )
            return

        # Delegate to ProcessManager with keep_terminal_open=True for error visibility
        process = self.process_manager.launch_module(
            name, module_name, cwd, keep_terminal_open=True
        )

        if process:
            self.show_toast(f"{name} Launched", "success")
            self.lbl_status.setText(f"● {name} Running")
            self.lbl_status.setStyleSheet("color: #30D158;")
        else:
            QMessageBox.critical(self, "Launch Error", f"Failed to launch {name}")

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
        self.layout_manager.set_edit_mode(checked)
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

    # Apply plot theme for matplotlib visualizations
    try:
        from shared.python.plot_theme import apply_plot_theme

        apply_plot_theme(settings_app="GolfModelingSuite")
    except ImportError:
        logger.debug("Plot theme module not available")

    # Show splash
    splash = GolfSplashScreen()
    splash.show()

    # Start async loading
    worker = AsyncStartupWorker(REPOS_ROOT)

    # Keep a reference to prevent garbage collection
    main_window = None

    def on_startup_finished(results: StartupResults) -> None:
        nonlocal main_window
        main_window = GolfLauncher(results)
        main_window.show()
        splash.finish(main_window)
        # Clean up worker after startup is complete
        worker.wait(1000)  # Wait for worker to finish

    def on_startup_progress(msg: str, percent: int) -> None:
        splash.show_message(msg, percent)

    worker.progress_signal.connect(on_startup_progress)
    worker.finished_signal.connect(on_startup_finished)

    worker.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
