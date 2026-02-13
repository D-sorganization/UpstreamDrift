#!/usr/bin/env python3
"""UpstreamDrift Launcher (PyQt6)

Features:
- Modern UI with rounded corners.
- Modular Docker Environment Management.
- Integrated Help and Documentation.

This module composes focused mixin classes into the GolfLauncher:
- LauncherUISetupMixin: Menu bar, top bar, grid area, bottom bar, search, console
- LauncherThemeMixin: Theme application, theme menus, plot theme
- LauncherSimulationMixin: Simulation launching, dependency checking
- LauncherDialogsMixin: Dialogs, settings, keyboard shortcuts, toast
"""

from __future__ import annotations

import sys
from typing import Any

# Add current directory to path so we can import ui_components if needed locally
from PyQt6.QtCore import QEventLoop, QTimer
from PyQt6.QtGui import QCloseEvent, QIcon
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox

from src.launchers.docker_manager import DockerLauncher
from src.launchers.launcher_constants import (
    CONFIG_DIR,
    DOCKER_STAGES,
    GRID_COLUMNS,
    LAYOUT_CONFIG_FILE,
    REPOS_ROOT,
    _lazy_load_engine_manager,
    _lazy_load_model_registry,
    logger,
)
from src.launchers.launcher_dialogs import LauncherDialogsMixin
from src.launchers.launcher_layout_manager import (
    LayoutManager,
    compute_centered_geometry,
)
from src.launchers.launcher_model_handlers import ModelHandlerRegistry
from src.launchers.launcher_process_manager import ProcessManager
from src.launchers.launcher_simulation import LauncherSimulationMixin
from src.launchers.launcher_theme import LauncherThemeMixin
from src.launchers.launcher_ui_setup import LauncherUISetupMixin
from src.launchers.ui_components import (
    ASSETS_DIR,
    AsyncStartupWorker,
    DockerCheckThread,
    DraggableModelCard,
    GolfSplashScreen,
    StartupResults,
)
from src.shared.python.security.subprocess_utils import kill_process_tree
from src.shared.python.theme.style_constants import Styles

# Backward-compatible re-exports
__all__ = [
    "GolfLauncher",
    "GRID_COLUMNS",
    "CONFIG_DIR",
    "LAYOUT_CONFIG_FILE",
    "DOCKER_STAGES",
    "main",
]


class GolfLauncher(
    LauncherUISetupMixin,
    LauncherThemeMixin,
    LauncherSimulationMixin,
    LauncherDialogsMixin,
    QMainWindow,
):
    """Main application window for the launcher.

    Composes focused mixins for UI setup, theme management,
    simulation launching, and dialog/settings management.
    """

    def __init__(self, startup_results: StartupResults | None = None) -> None:
        """Initialize the main window.

        Args:
            startup_results: Optional pre-loaded startup results from AsyncStartupWorker.
                            If provided, skips redundant loading of registry and engines.
        """
        super().__init__()
        self.setWindowTitle("UpstreamDrift")
        self.resize(1400, 900)
        self.center_window()

        self._startup_time_ms = (
            startup_results.startup_time_ms if startup_results else 0
        )

        self._load_window_icon()
        self._init_state(startup_results)
        self._init_managers()
        self._init_registry(startup_results)
        self._init_engine_manager(startup_results)
        self._build_available_models()
        self._init_layout_manager()
        self._initialize_model_order()

        self.init_ui()
        self._apply_theme_system()

        if startup_results:
            self._apply_docker_status(startup_results.docker_available)
        else:
            self.check_docker()

        self._load_layout()

        self.cleanup_timer = QTimer(self)
        self.cleanup_timer.timeout.connect(self._cleanup_processes)
        self.cleanup_timer.start(10000)

        self.toast_manager = None
        self._init_ui_components()

        if self._startup_time_ms > 0:
            logger.info(f"Application startup completed in {self._startup_time_ms}ms")

    def _load_window_icon(self) -> None:
        icon_candidates = [
            ASSETS_DIR / "golf_logo.ico",
            ASSETS_DIR / "golf_logo.png",
        ]
        for icon_path in icon_candidates:
            if icon_path.exists():
                self.setWindowIcon(QIcon(str(icon_path)))
                logger.info("Loaded icon: %s", icon_path.name)
                return
        logger.warning("No icon files found")

    def _init_state(self, startup_results: StartupResults | None) -> None:
        self.docker_available = (
            startup_results.docker_available if startup_results else False
        )
        self.docker_checker: DockerCheckThread | None = None
        self.selected_model: str | None = None
        self.model_cards: dict[str, Any] = {}
        self.model_order: list[str] = []
        self.layout_edit_mode = False
        self.available_models: dict[str, Any] = {}
        self.special_app_lookup: dict[str, Any] = {}
        self.current_filter_text = ""

    def _init_managers(self) -> None:
        self._setup_process_console()
        self.process_manager = ProcessManager(
            REPOS_ROOT,
            output_callback=self._on_process_output,
        )
        self.model_handler_registry = ModelHandlerRegistry()
        self.docker_launcher = DockerLauncher(REPOS_ROOT)
        self.running_processes = self.process_manager.running_processes

    def _init_registry(self, startup_results: StartupResults | None) -> None:
        if startup_results and startup_results.registry is not None:
            self.registry = startup_results.registry
            logger.info("Using pre-loaded model registry from async startup")
        else:
            try:
                MR = _lazy_load_model_registry()
                self.registry = MR(REPOS_ROOT / "src/config/models.yaml")
            except (ImportError, Exception) as e:
                logger.error(f"Failed to load ModelRegistry: {e}")
                self.registry = None

    def _init_engine_manager(self, startup_results: StartupResults | None) -> None:
        if startup_results and startup_results.engine_manager is not None:
            self.engine_manager = startup_results.engine_manager
            logger.info("Using pre-loaded engine manager from async startup")
        else:
            try:
                EM, _ = _lazy_load_engine_manager()
                self.engine_manager = EM(REPOS_ROOT)
            except (RuntimeError, ValueError, OSError) as e:
                logger.warning(f"Failed to initialize EngineManager: {e}")
                self.engine_manager = None

    def _init_layout_manager(self) -> None:
        self.layout_manager = LayoutManager(
            config_file=LAYOUT_CONFIG_FILE,
            available_models=self.available_models,
            get_model_func=self._get_model,
            create_card_func=lambda model: DraggableModelCard(model, self),
        )
        self.model_cards = self.layout_manager.model_cards
        self.model_order = self.layout_manager.model_order

    # -- Model management methods --

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
        self.model_order = self.layout_manager.model_order

    def _get_model(self, model_id: str) -> Any | None:
        """Retrieve a model or application by ID."""
        if model_id in self.available_models:
            return self.available_models[model_id]

        if self.registry:
            return self.registry.get_model(model_id)

        return None

    # -- Layout management --

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

    def create_model_card(self, model: Any) -> None:
        """Creates a clickable card widget (placeholder)."""

    def launch_model_direct(self, model_id: str) -> None:
        """Selects and immediately launches the model (for double-click)."""
        self.select_model(model_id)
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
        self.launch_simulation()

    # -- Window management --

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

        self.model_order = self.layout_manager.model_order
        self._sync_model_cards()

        # Restore window geometry
        geo = layout_data.get("window_geometry", {})
        if geo:
            x = geo.get("x", 100)
            y = geo.get("y", 100)
            w = geo.get("width", 1280)
            h = geo.get("height", 800)
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
        if hasattr(value, "return_value"):
            return default
        return int(value) if isinstance(value, int | float) else default

    # -- Model selection and UI state --

    def select_model(self, model_id: str) -> None:
        """Select a model and update UI."""
        self.selected_model = model_id

        # Update visual selection state using theme colors
        try:
            from src.shared.python.theme import get_current_colors

            c = get_current_colors()
        except ImportError:
            from src.shared.python.theme import (
                DARK_THEME as c,  # type: ignore[assignment]
            )

        for mid, card in self.model_cards.items():
            if mid == model_id:
                card.setStyleSheet(f"""
                    QFrame#ModelCard {{
                        background-color: {c.bg_highlight};
                        border: 2px solid {c.primary};
                        border-radius: 12px;
                    }}
                    """)
            else:
                card.setStyleSheet(f"""
                    QFrame#ModelCard {{
                        background-color: {c.bg_elevated};
                        border: 1px solid {c.border_default};
                        border-radius: 12px;
                    }}
                    QFrame#ModelCard:hover {{
                        background-color: {c.bg_highlight};
                        border: 1px solid {c.border_strong};
                    }}
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
        try:
            from src.shared.python.theme import get_current_colors

            c = get_current_colors()
        except ImportError:
            from src.shared.python.theme import (
                DARK_THEME as c,  # type: ignore[assignment]
            )

        if not self.selected_model:
            self.btn_launch.setText("Select a Model")
            self.btn_launch.setEnabled(False)
            self.btn_launch.setStyleSheet(f"""
                QPushButton {{
                    background-color: {c.bg_elevated};
                    color: {c.text_quaternary};
                    border-radius: 6px;
                }}
                """)
            return

        name = model_name or self.selected_model
        model = self._get_model(self.selected_model)

        # Check Docker dependency
        if model and getattr(model, "requires_docker", False):
            if not self.docker_available:
                self.btn_launch.setText("! Docker Required")
                self.btn_launch.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {c.bg_elevated};
                        color: {c.error};
                        border: 2px solid {c.error};
                        border-radius: 6px;
                    }}
                    """)
                self.btn_launch.setEnabled(False)
                return

        self.btn_launch.setText(f"Launch {name} >")
        self.btn_launch.setEnabled(True)
        self.btn_launch.setStyleSheet(f"""
            QPushButton {{
                background-color: {c.success};
                color: white;
                border-radius: 6px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {c.success_hover};
            }}
            """)

    def _get_engine_type(self, model_type: str) -> Any:
        """Map model type to EngineType."""
        _, EngineType = _lazy_load_engine_manager()

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
        return EngineType.MUJOCO

    # -- Docker --

    def _apply_docker_status(self, available: bool) -> None:
        """Apply Docker availability status to UI."""
        self.docker_available = available
        if available:
            self.lbl_status.setText("System Ready")
            self.lbl_status.setStyleSheet(Styles.STATUS_SUCCESS_BOLD)
        else:
            self.lbl_status.setText("Docker Not Found")
            self.lbl_status.setStyleSheet(Styles.STATUS_ERROR_BOLD)
        self.update_launch_button()

    def check_docker(self) -> None:
        """Start the docker check thread."""
        logger.info("Checking Docker status...")
        if hasattr(self, "docker_checker") and self.docker_checker is not None:
            if self.docker_checker.isRunning():
                self.docker_checker.wait(1000)
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

    # -- Menu toggle handlers --

    def _toggle_layout_mode_from_menu(self, checked: bool) -> None:
        """Toggle layout edit mode from menu action."""
        if hasattr(self, "btn_modify_layout"):
            self.btn_modify_layout.setChecked(checked)
            self.toggle_layout_mode(checked)

    def _toggle_context_help(self, checked: bool) -> None:
        """Toggle the context help panel visibility."""
        if hasattr(self, "context_help"):
            if checked:
                self.context_help.show()
            else:
                self.context_help.hide()

    # -- Cleanup --

    def _cleanup_processes(self) -> None:
        """Remove finished processes from tracking."""
        finished = []
        for key, proc in self.running_processes.items():
            if proc.poll() is not None:
                finished.append(key)

        for key in finished:
            del self.running_processes[key]

        if not self.running_processes:
            self.lbl_status.setText("Ready")
            self.lbl_status.setStyleSheet(Styles.STATUS_INACTIVE)

    def closeEvent(self, event: QCloseEvent | None) -> None:
        """Handle window close event to save layout."""
        running_count = sum(
            1 for p in self.running_processes.values() if p.poll() is None
        )

        if running_count > 0:
            word_is = "is" if running_count == 1 else "are"
            word_es = "es" if running_count > 1 else ""
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                f"There {word_is} {running_count} "
                f"running process{word_es}.\n\n"
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

        # Terminate running processes
        for key, process in list(self.running_processes.items()):
            if process.poll() is None:
                logger.info(f"Terminating child process: {key}")
                try:
                    if not kill_process_tree(process.pid):
                        process.terminate()
                except (RuntimeError, ValueError, OSError) as e:
                    logger.error(f"Failed to terminate {key}: {e}")

        super().closeEvent(event)


def main() -> None:
    """Application entry point."""
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "UpstreamDrift.GolfModelingSuite.Launcher.1"
            )
        except ImportError:
            pass

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    try:
        from shared.python.plot_theme import apply_plot_theme

        apply_plot_theme(settings_app="GolfModelingSuite")
    except ImportError:
        logger.debug("Plot theme module not available")

    splash = GolfSplashScreen()
    splash.show()

    worker = AsyncStartupWorker(REPOS_ROOT)

    main_window = None

    def on_startup_finished(results: StartupResults) -> None:
        """Create and display the main window after startup completes."""
        nonlocal main_window
        main_window = GolfLauncher(results)
        main_window.show()
        splash.finish(main_window)
        worker.wait(1000)

    def on_startup_progress(msg: str, percent: int) -> None:
        """Forward startup progress to the splash screen."""
        splash.show_message(msg, percent)

    worker.progress_signal.connect(on_startup_progress)
    worker.finished_signal.connect(on_startup_finished)

    worker.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
