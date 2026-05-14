# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

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

import contextlib
import sys
from typing import Any

from PyQt6.QtCore import QEventLoop, QObject, QRunnable, QThreadPool, QTimer, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QIcon
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox

from src.launchers.docker_manager import DockerLauncher
from src.launchers.embedded_tool_bootstrap import bootstrap_embeddable_tools
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
    SplashScreen,
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


class ProcessCleanupWorkerSignals(QObject):
    finished = pyqtSignal(list)


class ProcessCleanupWorker(QRunnable):
    """Worker thread for process cleanup (issue #2715).

    Runs process polling in a background thread to prevent UI blocking.
    """

    def __init__(self, running_processes: dict, process_lock) -> None:
        super().__init__()
        self.signals = ProcessCleanupWorkerSignals()
        self.running_processes = running_processes
        self.process_lock = process_lock
        self.signals = ProcessCleanupWorkerSignals()

    def run(self) -> None:
        """Poll processes for completion without blocking UI."""
        finished_keys = []
        with self.process_lock:
            for key, proc in list(self.running_processes.items()):
                if proc.poll() is not None:
                    finished_keys.append(key)
        self.signals.finished.emit(finished_keys)


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

    def __init__(
        self, startup_results: StartupResults | None = None, loading: bool = False
    ) -> None:
        """Initialize the main window.

        Args:
            startup_results: Optional pre-loaded startup results from AsyncStartupWorker.
                            If provided, skips redundant loading of registry and engines.
        """
        super().__init__()
        from PyQt6.QtCore import Qt

        self.loading = loading
        self.setWindowTitle("UpstreamDrift")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # Size to 80% of screen, capped at 1400x900
        screen = QApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            w = min(int(avail.width() * 0.80), 1400)
            h = min(int(avail.height() * 0.80), 900)
        else:
            w, h = 1280, 800
        self.resize(w, h)
        self.center_window()

        self._startup_time_ms = (
            startup_results.startup_time_ms if startup_results else 0
        )

        self._load_window_icon()
        self._init_state(startup_results)
        self._init_managers()
        # Skip heavy initialization in loading mode; async worker will provide results
        if not self.loading:
            self._init_registry(startup_results)
            self._init_engine_manager(startup_results)
            self._build_available_models()

        self._init_layout_manager()

        if not self.loading:
            self._initialize_model_order()

        self.init_ui()
        self._apply_theme_system()

        if not self.loading:
            from PyQt6.QtCore import QTimer as _QTimer

            _QTimer.singleShot(100, self._show_onboarding_if_needed)

        if self.loading:
            pass  # Wait for update_startup_results
        elif startup_results:
            self._apply_docker_status(startup_results.docker_available)
        else:
            self.check_docker()

        self._load_layout()

        # Setup process cleanup timer (issue #2715: moved to thread pool to prevent UI blocking)
        self.cleanup_timer = QTimer(self)
        self.cleanup_timer.timeout.connect(self._schedule_cleanup)
        self.cleanup_timer.start(10000)

        self.toast_manager = None
        self._init_ui_components()

        if self._startup_time_ms > 0:
            logger.info(f"Application startup completed in {self._startup_time_ms}ms")

    def _show_onboarding_if_needed(self) -> None:
        """Show first-run onboarding dialog if this is a new user."""
        try:
            from src.launchers.onboarding_dialog import show_onboarding_if_needed

            show_onboarding_if_needed(self)
        except ImportError as e:
            logger.debug(f"Onboarding dialog not available: {e}")

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
        # Initialize registry and engine_manager to None to preserve invariant
        # that these attributes always exist (required by Settings dialog etc.)
        # They will be populated by _init_registry() or update_startup_results()
        self.registry: Any = None
        self.engine_manager: Any = None

    def _init_managers(self) -> None:
        self._setup_process_console()
        self.process_manager = ProcessManager(
            REPOS_ROOT,
            output_callback=self._on_process_output,
        )
        self.model_handler_registry = ModelHandlerRegistry()
        self.docker_launcher = DockerLauncher(REPOS_ROOT)
        self.running_processes = self.process_manager.running_processes

        # Bootstrap embeddable tools registry (fixes #5049)
        # This ensures EMBEDDABLE_TOOL_REGISTRY is populated before any
        # context menus or embedded host widgets are created
        bootstrap_embeddable_tools()

    def _init_registry(self, startup_results: StartupResults | None) -> None:
        if startup_results and startup_results.registry is not None:
            self.registry = startup_results.registry
            logger.info("Using pre-loaded model registry from async startup")
        else:
            try:
                MR = _lazy_load_model_registry()
                self.registry = MR(REPOS_ROOT / "src/config/models.yaml")
            except ImportError as e:
                # Lazy import may fail when optional dependencies are missing.
                logger.error("Failed to load ModelRegistry: %s", e)
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

    def _create_category_header(self, title: str) -> Any:
        from PyQt6.QtWidgets import QLabel

        from src.shared.python.theme.typography import Weights, get_display_font

        try:
            from src.shared.python.theme import (
                get_current_colors,  # type: ignore[attr-defined]
            )

            c = get_current_colors()  # type: ignore[attr-defined]
        except ImportError:
            from src.shared.python.theme import (
                DARK_THEME as c,  # type: ignore[assignment]
            )

        lbl = QLabel(title)
        lbl.setFont(get_display_font(size=14, weight=Weights.BOLD))
        lbl.setStyleSheet(f"""
            QLabel {{
                color: {c.text_primary};
                padding-top: 20px;
                padding-bottom: 5px;
                border-bottom: 1px solid {c.border_default};
                margin-bottom: 10px;
            }}
        """)
        return lbl

    def _init_layout_manager(self) -> None:
        self.layout_manager = LayoutManager(
            config_file=LAYOUT_CONFIG_FILE,
            available_models=self.available_models,
            get_model_func=self._get_model,
            create_card_func=lambda model, **kwargs: DraggableModelCard(
                model, self, **kwargs
            ),
            create_header_func=self._create_category_header,
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
        if model_id is None:
            raise ValueError("model_id must be provided")
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
        if selected_ids is None:
            raise ValueError("selected_ids must be provided")
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
        if text is None:
            raise ValueError("text must be provided")
        self.layout_manager.update_search_filter(text)
        self._rebuild_grid()

    def _rebuild_grid(self) -> None:
        """Rebuild the grid layout based on current model order."""
        if getattr(self, "loading", False):
            while self.grid_layout.count():
                item = self.grid_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            try:
                from src.launchers.model_card import SkeletonCard
            except ImportError:
                SkeletonCard = None

            if SkeletonCard:
                for i in range(8):
                    self.grid_layout.addWidget(
                        SkeletonCard(self), i // GRID_COLUMNS, i % GRID_COLUMNS
                    )
            return

        self.layout_manager.rebuild_grid(self.grid_layout)

    def update_startup_results(self, results: StartupResults) -> None:
        """Transition from loading skeleton to full application."""
        self.loading = False
        self._startup_time_ms = results.startup_time_ms
        self._init_registry(results)
        self._init_engine_manager(results)
        self._build_available_models()
        self.layout_manager.available_models = self.available_models
        self._initialize_model_order()
        self._apply_docker_status(results.docker_available)
        self._load_layout()

        from PyQt6.QtCore import QTimer as _QTimer

        _QTimer.singleShot(100, self._show_onboarding_if_needed)

    def create_model_card(self, model: Any) -> None:
        """Creates a clickable card widget (placeholder)."""

    def launch_model_direct(self, model_id: str) -> None:
        """Selects and immediately launches the model (for double-click)."""
        if model_id is None:
            raise ValueError("model_id must be provided")
        self.select_model(model_id)
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
        self.launch_simulation()

    # -- Window management --

    def center_window(self) -> None:
        """Center the window on the primary screen.

        Delegates to _center_window() which is the single source of truth
        using compute_centered_geometry() from launcher_layout_manager.py.
        """
        self._center_window()

    def _center_window(self) -> None:
        """Center the window on the primary screen using compute_centered_geometry().

        This is the single source of truth for window centering. Uses
        compute_centered_geometry() from launcher_layout_manager.py for
        consistent geometry calculations.
        """
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

    def _load_layout(self) -> None:
        """Load the saved model layout from configuration file."""
        layout_data = self.layout_manager.load_layout()

        if layout_data is None:
            self._rebuild_grid()
            return

        self.model_order = self.layout_manager.model_order
        self._sync_model_cards()

        # Restore window geometry, clamped to screen bounds
        geo = layout_data.get("window_geometry", {})
        if geo:
            x = geo.get("x", 100)
            y = geo.get("y", 100)
            w = geo.get("width", 1280)
            h = geo.get("height", 800)
            # Clamp to screen size
            screen = QApplication.primaryScreen()
            if screen:
                avail = screen.availableGeometry()
                w = min(w, avail.width() - 40)
                h = min(h, avail.height() - 40)
                x = max(avail.x(), min(x, avail.x() + avail.width() - w))
                y = max(avail.y() + 30, min(y, avail.y() + avail.height() - h))
            elif y < 30:
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
            # If "docker_mode" is not in options, default to self.docker_available
            saved_docker = options.get("docker_mode", self.docker_available)
            if saved_docker and self.docker_available:
                self.chk_docker.setChecked(True)
            else:
                self.chk_docker.setChecked(False)

        # Restore selected model
        saved_selection = layout_data.get("selected_model")
        if saved_selection and saved_selection in self.model_cards:
            self.select_model(saved_selection)

        self._rebuild_grid()
        logger.info("Layout loaded successfully")

    def _safe_int(self, value: Any, default: int) -> int:
        """Safely convert a value to int, handling Mock objects from tests."""
        if default is None:
            raise ValueError("default must be provided")
        if hasattr(value, "return_value"):
            return default
        return int(value) if isinstance(value, int | float) else default

    # -- Model selection and UI state --

    def select_model(self, model_id: str) -> None:
        """Select a model and update UI."""
        if model_id is None:
            raise ValueError("model_id must be provided")
        self.selected_model = model_id

        # Update visual selection state using theme colors
        try:
            from src.shared.python.theme import (
                get_current_colors,  # type: ignore[attr-defined]
            )

            c = get_current_colors()  # type: ignore[attr-defined]
        except ImportError:
            from src.shared.python.theme import (
                DARK_THEME as c,  # type: ignore[assignment]
            )

        for mid, card in self.model_cards.items():
            if hasattr(card, "set_selected"):
                card.set_selected(mid == model_id)
            else:
                # Fallback for older cards
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
            from src.shared.python.theme import (
                get_current_colors,  # type: ignore[attr-defined]
            )

            c = get_current_colors()  # type: ignore[attr-defined]
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
        if (
            model
            and getattr(model, "requires_docker", False)
            and not self.docker_available
        ):
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
        if model_type is None:
            raise ValueError("model_type must be provided")
        _, EngineType = _lazy_load_engine_manager()

        if "mujoco" in model_type:
            return EngineType.MUJOCO
        if "drake" in model_type:
            return EngineType.DRAKE
        if "pinocchio" in model_type:
            return EngineType.PINOCCHIO
        if "opensim" in model_type:
            return EngineType.OPENSIM
        if "myosim" in model_type:
            return EngineType.MYOSIM
        return EngineType.MUJOCO

    # -- Docker --

    def _apply_docker_status(self, available: bool) -> None:
        """Apply Docker availability status to UI."""
        if available is None:
            raise ValueError("available must be provided")
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
            with contextlib.suppress(TypeError, RuntimeError):
                self.docker_checker.result.disconnect(self.on_docker_check_complete)

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

    def _schedule_cleanup(self) -> None:
        """Schedule process cleanup in a worker thread (issue #2715).

        Prevents UI blocking when checking process status.
        """
        worker = ProcessCleanupWorker(
            self.running_processes, self.process_manager._process_lock
        )
        worker.signals.finished.connect(self._on_cleanup_finished)
        QThreadPool.globalInstance().start(worker)

    def _on_cleanup_finished(self, finished_keys: list[str]) -> None:
        """Handle cleanup completion from worker thread."""
        with self.process_manager._process_lock:
            for key in finished_keys:
                if key in self.running_processes:
                    del self.running_processes[key]

        if not self.running_processes and hasattr(self, "lbl_status"):
            self.lbl_status.setText("Ready")
            self.lbl_status.setStyleSheet(Styles.STATUS_INACTIVE)

    def _cleanup_processes(self) -> None:
        """Legacy cleanup method — synchronous for callers that need immediate results."""
        finished_keys = [
            key
            for key, proc in list(self.running_processes.items())
            if proc.poll() is not None
        ]
        self._on_cleanup_finished(finished_keys)

    def closeEvent(self, event: QCloseEvent | None) -> None:  # noqa: C901
        """Handle window close event to save layout."""
        running_count = sum(
            1 for p in self.running_processes.values() if p.poll() is None
        )

        if running_count > 0:
            word_is = "is" if running_count == 1 else "are"
            word_es = "es" if running_count > 1 else ""

            from src.launchers.launcher_dialogs import ThemedModalDialog
            from PyQt6.QtWidgets import QWidget, QDialog

            overlay = QWidget(self)
            overlay.setStyleSheet("background-color: rgba(0, 0, 0, 150);")
            overlay.setGeometry(self.rect())
            overlay.show()

            dialog = ThemedModalDialog(
                self,
                "Confirm Exit",
                f"There {word_is} {running_count} running process{word_es}.\n\nClosing will terminate all running simulations.\nAre you sure you want to exit?",
            )

            reply = dialog.exec()
            overlay.hide()
            overlay.deleteLater()

            if reply == QDialog.DialogCode.Rejected:
                if event:
                    event.ignore()
                return

        self._save_layout()

        # Stop cleanup timer
        if hasattr(self, "cleanup_timer") and self.cleanup_timer is not None:
            self.cleanup_timer.stop()
            self.cleanup_timer.deleteLater()
            self.cleanup_timer = None  # type: ignore[assignment]

        # Clean up docker checker thread
        if hasattr(self, "docker_checker") and self.docker_checker is not None:
            with contextlib.suppress(TypeError, RuntimeError):
                self.docker_checker.result.disconnect(self.on_docker_check_complete)
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


def _install_global_ui_zoom(app: QApplication) -> None:
    from src.launchers.app_zoom import install_global_ui_zoom

    install_global_ui_zoom(app)


def main() -> None:
    """Application entry point."""
    import traceback

    def excepthook(exc_type, exc_value, exc_tb):
        with open("crash_traceback.txt", "w") as f:
            traceback.print_exception(exc_type, exc_value, exc_tb, file=f)
        QApplication.quit()

    sys.excepthook = excepthook

    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "UpstreamDrift.Launcher.1"
            )
        except ImportError:
            logger.debug(
                "ctypes not available; skipping Windows AppUserModelID assignment"
            )

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    _install_global_ui_zoom(app)

    # Set global application icon
    icon_path = ASSETS_DIR / "golf_logo.png"
    if not icon_path.exists():
        icon_path = ASSETS_DIR / "golf_logo.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    qss_path = ASSETS_DIR / "theme" / "dark_modern.qss"
    if qss_path.exists():
        try:
            with open(qss_path) as f:
                app.setStyleSheet(f.read())
        except Exception as e:
            logger.warning(f"Could not load QSS: {e}")

    try:
        from shared.python.plot_theme import apply_plot_theme

        apply_plot_theme(settings_app="UpstreamDrift")
    except ImportError:
        logger.debug("Plot theme module not available")

    try:
        from shared.python.theme.zoom import install_application_zoom

        install_application_zoom(app)
    except ImportError as e:
        logger.debug(f"Zoom support not available: {e}")

    splash = SplashScreen()
    splash.show()
    worker = AsyncStartupWorker(REPOS_ROOT)

    main_window = GolfLauncher(loading=True)
    main_window.show()

    def on_startup_finished(results: StartupResults) -> None:
        """Create and display the main window after startup completes."""
        nonlocal main_window
        try:
            main_window.update_startup_results(results)
            splash.finish(main_window)
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            logger.error(f"Failed to update GolfLauncher: {e}")
            QApplication.quit()
        worker.wait(1000)

    def on_startup_progress(msg: str, percent: int) -> None:
        """Forward startup progress."""
        logger.info(f"Startup progress: {percent}% - {msg}")

    def on_startup_error(error_msg: str) -> None:
        """Handle startup failure."""
        logger.error(f"Startup failed: {error_msg}")

        QMessageBox.critical(
            None, "Startup Error", f"Failed to initialize UpstreamDrift:\n\n{error_msg}"
        )
        QApplication.quit()

    worker.progress_signal.connect(on_startup_progress)
    worker.finished_signal.connect(on_startup_finished)
    worker.error_signal.connect(on_startup_error)

    worker.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
