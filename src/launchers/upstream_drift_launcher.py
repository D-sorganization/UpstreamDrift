# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

#!/usr/bin/env python3
"""UpstreamDrift Launcher (PyQt6)

Features:
- Modern UI with rounded corners.
- Modular Docker Environment Management.
- Integrated Help and Documentation.

This module composes focused mixin classes into the UpstreamDriftLauncher:
- LauncherUISetupMixin: Menu bar, top bar, grid area, bottom bar, search, console
- LauncherThemeMixin: Theme application, theme menus, plot theme
- LauncherSimulationMixin: Simulation launching, dependency checking
- LauncherDialogsMixin: Dialogs, settings, keyboard shortcuts, toast
"""

import contextlib
import os
import sys
import time
from typing import Any, cast

from PyQt6.QtCore import QEventLoop, QObject, QRunnable, QThreadPool, QTimer, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QIcon
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt6.QtCore import QEvent, Qt, QRect

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

from src.launchers.launcher_dialogs import DialogsManager
from src.launchers.launcher_layout_manager import (
    LayoutManager,
    compute_centered_geometry,
)
from src.launchers.launcher_model_handlers import ModelHandlerRegistry
from src.launchers.launcher_process_manager import ProcessManager
from src.launchers.launcher_simulation import SimulationManager
from src.launchers.sidekick_readiness import (
    check_sidekick_api_readiness,
    readiness_detail_for_log,
)
from src.launchers.launcher_theme import ThemeManager
from src.launchers.launcher_ui_setup import UISetupManager

from src.launchers.ui_components import (
    ASSETS_DIR,
    AsyncStartupWorker,
    DockerCheckThread,
    DraggableModelCard,
    SplashScreen,
    StartupResults,
)


class FramelessResizeFilter(QObject):
    def __init__(self, window: QMainWindow) -> None:
        super().__init__(window)
        self.window = window
        self._resizing = False
        self._resize_edge = 0
        self._start_pos = None
        self._start_geo: QRect | None = None

    def eventFilter(self, obj: QObject | None, event: QEvent | None) -> bool:
        try:
            if event is None or obj is None:
                return False

            # Check if either the target object or the filter window has been deleted in C++
            if self.window is None:
                return False
            _ = self.window.parent()
            _ = obj.parent()

            typed_event = cast(Any, event)
            if event.type() in (
                QEvent.Type.MouseMove,
                QEvent.Type.MouseButtonPress,
                QEvent.Type.MouseButtonRelease,
                QEvent.Type.HoverMove,
            ):
                if hasattr(event, "globalPosition"):
                    gpos = event.globalPosition().toPoint()
                elif hasattr(event, "globalPos"):
                    gpos = event.globalPos()
                else:
                    return super().eventFilter(obj, event)

                local_pos = self.window.mapFromGlobal(gpos)
                x, y = local_pos.x(), local_pos.y()
                w, h = self.window.width(), self.window.height()
                border = 8

                if not self._resizing:
                    if 0 <= x <= w and 0 <= y <= h:
                        edge = 0
                        if x < border and y < border:
                            edge = 13
                        elif x > w - border and y < border:
                            edge = 14
                        elif x < border and y > h - border:
                            edge = 16
                        elif x > w - border and y > h - border:
                            edge = 17
                        elif x < border:
                            edge = 10
                        elif x > w - border:
                            edge = 11
                        elif y < border:
                            edge = 12
                        elif y > h - border:
                            edge = 15

                        if edge != 0:
                            if (
                                event.type() == QEvent.Type.MouseButtonPress
                                and typed_event.button() == Qt.MouseButton.LeftButton
                            ):
                                self._resizing = True
                                self._resize_edge = edge
                                self._start_pos = gpos
                                self._start_geo = self.window.geometry()
                                return True
                            if event.type() in (
                                QEvent.Type.HoverMove,
                                QEvent.Type.MouseMove,
                            ):
                                if edge in (13, 17):
                                    self.window.setCursor(
                                        Qt.CursorShape.SizeFDiagCursor
                                    )
                                elif edge in (14, 16):
                                    self.window.setCursor(
                                        Qt.CursorShape.SizeBDiagCursor
                                    )
                                elif edge in (10, 11):
                                    self.window.setCursor(Qt.CursorShape.SizeHorCursor)
                                elif edge in (12, 15):
                                    self.window.setCursor(Qt.CursorShape.SizeVerCursor)
                                return True
                        else:
                            if (
                                self.window.cursor().shape()
                                != Qt.CursorShape.ArrowCursor
                            ):
                                self.window.setCursor(Qt.CursorShape.ArrowCursor)
                else:
                    if event.type() == QEvent.Type.MouseMove:
                        if self._start_geo is None:
                            return False
                        delta = gpos - self._start_pos
                        rect = QRect(self._start_geo)
                        if self._resize_edge in (10, 13, 16):
                            rect.setLeft(rect.left() + delta.x())
                        if self._resize_edge in (11, 14, 17):
                            rect.setRight(rect.right() + delta.x())
                        if self._resize_edge in (12, 13, 14):
                            rect.setTop(rect.top() + delta.y())
                        if self._resize_edge in (15, 16, 17):
                            rect.setBottom(rect.bottom() + delta.y())

                        if (
                            rect.width() >= self.window.minimumWidth()
                            and rect.height() >= self.window.minimumHeight()
                        ):
                            self.window.setGeometry(rect)
                        return True
                    if event.type() == QEvent.Type.MouseButtonRelease:
                        self._resizing = False
                        self.window.setCursor(Qt.CursorShape.ArrowCursor)
                        return True
            return super().eventFilter(obj, event)
        except RuntimeError:
            try:
                app = QApplication.instance()
                if app is not None:
                    app.removeEventFilter(self)
            except Exception:  # noqa: BLE001
                pass
            return False


from src.shared.python.security.subprocess_utils import kill_process_tree
from src.shared.python.theme.style_constants import Styles

# Backward-compatible re-exports
__all__ = [
    "UpstreamDriftLauncher",
    "GRID_COLUMNS",
    "CONFIG_DIR",
    "LAYOUT_CONFIG_FILE",
    "DOCKER_STAGES",
    "STARTUP_TIMEOUT_SEC",
    "main",
]


# Async startup is normally well under a second; 30s is a generous ceiling
# that comfortably covers cold-disk Docker probes and slow first-run
# registry imports while still surfacing a true hang (e.g. crashed worker
# thread) before the user concludes the app is broken.  See issue #5490.
STARTUP_TIMEOUT_SEC: int = 30
assert STARTUP_TIMEOUT_SEC > 0, (
    "STARTUP_TIMEOUT_SEC must be > 0 to schedule a recovery timer"
)

SIDEKICK_API_READY_TIMEOUT_SEC: float = 15.0
SIDEKICK_API_READY_RETRY_MS: int = 500


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


class LauncherOrchestrator:
    """Domain logic coordinator for the UpstreamDrift launcher.

    Manages the ModelRegistry, EngineManager, and Docker health status,
    keeping these domain responsibilities separated from the Qt UI.
    """

    def __init__(self) -> None:
        self.registry = None
        self.engine_manager = None
        self.docker_available = False
        self.available_models: dict[str, Any] = {}
        self.special_app_lookup: dict[str, Any] = {}

    def initialize_from_results(self, startup_results: "StartupResults | None") -> None:
        """Initialize domain state from async startup results."""
        self.docker_available = (
            startup_results.docker_available if startup_results else False
        )
        self.init_registry(startup_results)
        self.init_engine_manager(startup_results)
        self.build_available_models()

    def init_registry(self, startup_results: "StartupResults | None") -> None:
        if startup_results and startup_results.registry is not None:
            self.registry = startup_results.registry
            logger.info("Using pre-loaded model registry from async startup")
        else:
            try:
                MR = _lazy_load_model_registry()
                self.registry = MR(REPOS_ROOT / "src/config/models.yaml")
            except ImportError as e:
                logger.error("Failed to load ModelRegistry: %s", e)
                self.registry = None

    def init_engine_manager(self, startup_results: "StartupResults | None") -> None:
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

    def build_available_models(self) -> None:
        """Collect all known models and auxiliary applications."""
        logger.debug("Building available models from registry...")
        self.available_models.clear()
        self.special_app_lookup.clear()

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

    def get_model(self, model_id: str) -> "Any | None":
        """Retrieve a model or application by ID."""
        if model_id is None:
            raise ValueError("model_id must be provided")
        if model_id in self.available_models:
            return self.available_models[model_id]

        if self.registry:
            return self.registry.get_model(model_id)

        return None


class UpstreamDriftLauncher(QMainWindow):
    """Main application window for the launcher.

    Composes focused mixins for UI setup, theme management,
    simulation launching, and dialog/settings management.
    """

    sidekick_sidebar: Any | None
    sidekick_window: Any | None
    _sidekick_popped_out: bool
    _sidekick_needs_initial_sizing: bool

    @property
    def docker_available(self) -> bool:
        return self.orchestrator.docker_available

    @docker_available.setter
    def docker_available(self, value: bool) -> None:
        self.orchestrator.docker_available = value

    @property
    def registry(self) -> Any:
        return self.orchestrator.registry

    @registry.setter
    def registry(self, value: Any) -> None:
        self.orchestrator.registry = value

    @property
    def engine_manager(self) -> Any:
        return self.orchestrator.engine_manager

    @engine_manager.setter
    def engine_manager(self, value: Any) -> None:
        self.orchestrator.engine_manager = value

    @property
    def available_models(self) -> dict:
        return self.orchestrator.available_models

    @available_models.setter
    def available_models(self, value: dict) -> None:
        self.orchestrator.available_models = value

    @property
    def special_app_lookup(self) -> dict:
        return self.orchestrator.special_app_lookup

    @special_app_lookup.setter
    def special_app_lookup(self, value: dict) -> None:
        self.orchestrator.special_app_lookup = value

    def _get_model(self, model_id: str) -> Any:
        return self.orchestrator.get_model(model_id)

    def get_model(self, model_id: str) -> Any:
        return self.orchestrator.get_model(model_id)

    def __getattr__(self, name: str) -> Any:
        # Check managers to forward attributes dynamically (maintaining mixin-compatibility)
        for manager_name in (
            "ui_setup_manager",
            "theme_manager",
            "simulation_manager",
            "dialogs_manager",
        ):
            if manager_name in self.__dict__:
                manager = self.__dict__[manager_name]
                if name in manager.__dict__ or hasattr(type(manager), name):
                    attr = getattr(manager, name)
                    import types

                    if isinstance(attr, types.MethodType):
                        return types.MethodType(attr.__func__, self)
                    return attr
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

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

        self.ui_setup_manager = UISetupManager(self)
        self.theme_manager = ThemeManager(self)
        self.simulation_manager = SimulationManager(self)
        self.dialogs_manager = DialogsManager(self)
        self.loading = loading
        self.orchestrator = LauncherOrchestrator()
        self.setWindowTitle("UpstreamDrift")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowMinimizeButtonHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumSize(800, 600)

        # Enable app-level mouse tracking filter for frameless resizing
        self._resize_filter = FramelessResizeFilter(self)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self._resize_filter)

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
            pass  # Orchestrator already initialized from results in _init_state

        self._init_layout_manager()

        if not self.loading:
            self._initialize_model_order()

        self.ui_setup_manager.init_ui()
        self.theme_manager._apply_theme_system()

        if not self.loading:
            from PyQt6.QtCore import QTimer as _QTimer

            _QTimer.singleShot(100, self._show_onboarding_if_needed)

        if self.loading:
            # Issue #5490: previously this branch was a silent ``pass`` that
            # waited forever for ``update_startup_results``.  If the async
            # startup worker crashed in a sibling thread the user was left
            # staring at an empty skeleton with no diagnostic.  Emit a log
            # line and arm a recovery timeout that surfaces a visible
            # error if the worker never reports back.
            logger.info(
                "Launcher entered async-startup wait state; "
                "arming %ss timeout for update_startup_results",
                STARTUP_TIMEOUT_SEC,
            )
            QTimer.singleShot(
                int(STARTUP_TIMEOUT_SEC * 1000), self._handle_startup_timeout
            )
        elif startup_results:
            self._apply_docker_status(startup_results.docker_available)
        else:
            self.simulation_manager.check_docker()

        self._load_layout()

        # Setup process cleanup timer (issue #2715: moved to thread pool to prevent UI blocking)
        self.cleanup_timer = QTimer(self)
        self.cleanup_timer.timeout.connect(self._schedule_cleanup)
        self.cleanup_timer.start(10000)

        self.toast_manager = None
        self.ui_setup_manager._init_ui_components()

        if self._startup_time_ms > 0:
            logger.info(f"Application startup completed in {self._startup_time_ms}ms")

    def showEvent(self, event: Any) -> None:
        """Force sidekick splitter sizes on first display."""
        super().showEvent(event)
        if self._sidekick_needs_initial_sizing:
            self._sidekick_needs_initial_sizing = False
            self._apply_sidekick_splitter_sizes()

    def _apply_sidekick_splitter_sizes(self) -> None:
        """Set main_layout splitter sizes to give the sidekick 300px."""
        layout = self.main_layout
        sidebar = self.sidekick_sidebar
        if layout is None or sidebar is None:
            return
        sizes = layout.sizes()
        if len(sizes) == 3 and sum(sizes) > 0:
            if sizes[2] == 0:
                total = sum(sizes)
                # Keep current sidebar width if it's > 0, otherwise default to 120
                sizes[0] = max(sizes[0], 120) if sizes[0] > 0 else 120
                sizes[2] = 300
                sizes[1] = max(100, total - sizes[0] - sizes[2])
                layout.setSizes(sizes)
                logger.info("Sidekick splitter sized: %s", sizes)
            sidebar.setVisible(True)

    def _show_onboarding_if_needed(self) -> None:
        """Show first-run onboarding dialog if this is a new user."""
        if self._should_skip_onboarding():
            logger.debug("Skipping onboarding dialog in non-interactive test mode")
            return
        try:
            from src.launchers.onboarding_dialog import show_onboarding_if_needed

            show_onboarding_if_needed(self)
        except ImportError as e:
            logger.debug(f"Onboarding dialog not available: {e}")

    @staticmethod
    def _should_skip_onboarding() -> bool:
        """Return true when modal onboarding would block a non-interactive run."""
        return (
            os.environ.get("UPSTREAMDRIFT_DISABLE_ONBOARDING") == "1"
            or "PYTEST_CURRENT_TEST" in os.environ
        )

    def _get_sidekick_module(self) -> Any | None:
        """Import the Sidekick sidebar module, trying multiple fallback paths."""
        try:
            from src.shared.python.gui_launcher.tools_sidebar_integration import (
                _import_sidebar_module,
            )
        except ImportError as e:
            logger.debug("Sidekick integration shim not importable: %s", e)
            return None

        module = _import_sidebar_module()
        if module is None:
            import importlib
            import sys as _sys

            # Try checking out sibling Tools repository first
            sibling_tools = REPOS_ROOT.parent / "Tools"
            if sibling_tools.is_dir():
                sibling_src = str(sibling_tools / "src")
                sibling_python = str(sibling_tools / "src" / "shared" / "python")
                if sibling_src not in _sys.path:
                    _sys.path.insert(0, sibling_src)
                if sibling_python not in _sys.path:
                    _sys.path.insert(0, sibling_python)

            # Fall back to vendored ud-tools
            vendor_src = str(REPOS_ROOT / "vendor" / "ud-tools" / "src")
            vendor_python = str(
                REPOS_ROOT / "vendor" / "ud-tools" / "src" / "shared" / "python"
            )
            if vendor_src not in _sys.path:
                _sys.path.insert(0, vendor_src)
            if vendor_python not in _sys.path:
                _sys.path.insert(0, vendor_python)
            for _name in (
                "shared.python.sidekick.ui.tools_sidebar",
                "sidekick.ui.tools_sidebar",
            ):
                try:
                    module = importlib.import_module(_name)
                    break
                except ImportError:
                    continue
        return module

    def _create_sidekick_sidebar_widget(self, module: Any) -> Any | None:
        """Invoke the factory to create the sidekick sidebar widget."""
        if module is None:
            logger.warning("Sidekick sidebar not installed: shared module unavailable")
            return None

        factory = getattr(module, "create_tools_sidebar", None)
        if factory is None:
            logger.warning(
                "Sidekick sidebar not installed: create_tools_sidebar() missing "
                "from %s",
                getattr(module, "__name__", "<unknown>"),
            )
            return None

        try:
            return factory(parent=self, project_root=str(REPOS_ROOT))
        except TypeError:
            try:
                return factory(parent=self)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Sidekick factory call failed: %s", exc)
                return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sidekick factory call failed: %s", exc)
            return None

    def _embed_sidekick_sidebar_widget(self, sidebar_widget: Any) -> None:
        """Embed the created sidebar widget into the main layout."""
        if sidebar_widget is None:
            return

        main_layout = self.main_layout
        if main_layout is None or not hasattr(main_layout, "addWidget"):
            logger.warning(
                "Sidekick sidebar not installed: main_layout splitter not "
                "available on host launcher"
            )
            return

        main_layout.addWidget(sidebar_widget)
        if hasattr(main_layout, "setStretchFactor"):
            with contextlib.suppress(RuntimeError, ValueError, TypeError):
                main_layout.setStretchFactor(main_layout.count() - 1, 2)

        self.sidekick_sidebar = sidebar_widget
        self._apply_sidekick_splitter_sizes()

        logger.info("Sidekick sidebar embedded in main splitter")

    def _install_sidekick_sidebar(self) -> None:
        """Embed the Sidekick multitab sidebar as a third splitter pane."""
        logger.info("Initializing _install_sidekick_sidebar")
        module = self._get_sidekick_module()
        widget = self._create_sidekick_sidebar_widget(module)
        self._embed_sidekick_sidebar_widget(widget)

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
        self.docker_checker: DockerCheckThread | None = None
        self.selected_model: str | None = None
        self.model_cards: dict[str, Any] = {}
        self.model_order: list[str] = []
        self.background_api_process: Any | None = None
        self._sidekick_api_wait_started_at: float | None = None
        self.layout_edit_mode = False
        self.current_filter_text = ""
        self._sidekick_needs_initial_sizing = True
        self.sidekick_sidebar = None
        self.sidekick_window = None
        self._sidekick_popped_out = False
        self.orchestrator.initialize_from_results(startup_results)

    def _init_managers(self) -> None:
        self.ui_setup_manager._setup_process_console()
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

        # Start the background API server so the Sidekick Chat UI can connect
        cwd = (
            REPOS_ROOT / "UpstreamDrift"
            if (REPOS_ROOT / "UpstreamDrift").exists()
            else REPOS_ROOT
        )

        if "PYTEST_CURRENT_TEST" not in os.environ:
            self.background_api_process = self.process_manager.launch_module(
                name="background_api_server",
                module_name="src.api.server",
                cwd=cwd,
            )

    def _create_category_header(self, title: str) -> Any:
        from PyQt6.QtWidgets import QLabel

        from src.shared.python.theme.typography import Weights, get_display_font

        try:
            import src.shared.python.theme as _theme

            c = _theme.get_current_colors()  # type: ignore[attr-defined]
        except (ImportError, AttributeError):
            from src.shared.python.theme import DARK_THEME as c  # type: ignore[assignment,no-redef]

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
            available_models=self.orchestrator.available_models,
            get_model_func=self.orchestrator.get_model,
            create_card_func=lambda model, **kwargs: DraggableModelCard(
                model, self, **kwargs
            ),
            create_header_func=self._create_category_header,
        )
        self.model_cards = self.layout_manager.model_cards
        self.model_order = self.layout_manager.model_order

    # -- Model management methods --

    def _initialize_model_order(self) -> None:
        """Set a sensible default grid ordering."""
        logger.debug("Initializing model order...")
        self.layout_manager.initialize_model_order()
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
                "live_visualization": (self.chk_live.isChecked() if True else True),
                "gpu_acceleration": (self.chk_gpu.isChecked() if True else False),
                "docker_mode": (self.chk_docker.isChecked() if True else False),
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
                widget = item.widget() if item is not None else None
                if widget is not None:
                    widget.deleteLater()

            _SkeletonCard: Any = None
            with contextlib.suppress(ImportError):
                from src.launchers.model_card import SkeletonCard as _SkeletonCard  # type: ignore[no-redef]

            if _SkeletonCard is not None:
                for i in range(8):
                    self.grid_layout.addWidget(
                        _SkeletonCard(self), i // GRID_COLUMNS, i % GRID_COLUMNS
                    )
            return

        self.layout_manager.rebuild_grid(self.grid_layout)

    def update_startup_results(self, results: StartupResults) -> None:
        """Transition from loading skeleton to full application."""
        self.loading = False
        self._startup_time_ms = results.startup_time_ms
        self.orchestrator.initialize_from_results(results)
        self.layout_manager.available_models = self.orchestrator.available_models
        self._initialize_model_order()
        self._apply_docker_status(results.docker_available)
        self._load_layout()

        from PyQt6.QtCore import QTimer as _QTimer

        # Defer Sidekick installation slightly so the main UI can render and
        # become 100% responsive immediately, bypassing any startup freeze
        # caused by blocking/synchronous AI network calls on initialization.
        _QTimer.singleShot(300, self._install_sidekick_sidebar_deferred)
        _QTimer.singleShot(100, self._show_onboarding_if_needed)

    def _install_sidekick_sidebar_deferred(self) -> None:
        """Deferred Sidekick installation to prevent startup freeze."""
        if not self._sidekick_api_ready_for_sidebar():
            return
        self._install_sidekick_sidebar()
        self._seed_sidekick_workspace()

    def _sidekick_api_ready_for_sidebar(self) -> bool:
        """Gate Sidekick chat/sidebar installation on API readiness."""
        readiness = check_sidekick_api_readiness()
        if readiness.ready:
            self._sidekick_api_wait_started_at = None
            return True

        now = time.monotonic()
        if self._sidekick_api_wait_started_at is None:
            self._sidekick_api_wait_started_at = now

        elapsed = now - self._sidekick_api_wait_started_at
        process = getattr(self, "background_api_process", None)
        process_running = process is not None and process.poll() is None
        if elapsed < SIDEKICK_API_READY_TIMEOUT_SEC and process_running:
            logger.info(
                "Waiting for Sidekick API readiness before installing sidebar: %s",
                readiness_detail_for_log(readiness),
            )
            QTimer.singleShot(
                SIDEKICK_API_READY_RETRY_MS, self._install_sidekick_sidebar_deferred
            )
            return False

        logger.warning(
            "Sidekick sidebar not installed because API readiness failed: %s",
            readiness_detail_for_log(readiness),
        )
        show_toast = getattr(self, "show_toast", None)
        if callable(show_toast):
            show_toast(
                "Sidekick chat is waiting on the background API. "
                f"Readiness check failed at {readiness.url}.",
                "warning",
            )
        return False

    def _seed_sidekick_workspace(self) -> None:
        """Push launcher state into any active Sidekick workspace registry.

        Called after startup results are applied so the Sidekick workspace
        tab shows live state (engine manager, model registry, current scenario)
        rather than an empty inspector.

        Issue #5616 — seed the registry from the launcher.

        Postcondition: if a sidebar with a WorkspaceRegistry is reachable,
        its registry contains 'engine_manager' and 'model_registry' keys.
        LOD: reaches only one level deep (sidebar.registry).
        """
        sidebar = self.sidekick_sidebar
        if sidebar is None:
            # Try the tools-sidebar integration hook if present
            try:
                from src.shared.python.gui_launcher import tools_sidebar_integration

                get_active_sidebar = getattr(
                    tools_sidebar_integration, "get_active_sidebar", None
                )
                if callable(get_active_sidebar):
                    sidebar = get_active_sidebar()
            except ImportError:
                pass

        if sidebar is None:
            logger.debug("No sidekick sidebar found; workspace seed skipped")
            return

        registry = getattr(sidebar, "registry", None)
        set_variable = getattr(registry, "set_variable", None)
        if not callable(set_variable):
            logger.debug(
                "sidebar.registry has no callable set_variable; workspace seed skipped"
            )
            return

        if self.orchestrator.engine_manager is not None:
            set_variable("engine_manager", self.orchestrator.engine_manager)
        if self.orchestrator.registry is not None:
            set_variable("model_registry", self.orchestrator.registry)
        logger.info("Sidekick workspace seeded with engine_manager and model_registry")

    def _handle_startup_timeout(self) -> None:
        """Recover from a hung async-startup worker (issue #5490).

        Fires ``STARTUP_TIMEOUT_SEC`` after the launcher entered the
        loading-skeleton wait state.  If ``update_startup_results`` already
        completed, this is a no-op — ``self.loading`` will be ``False``
        and we leave the live UI untouched.  Otherwise we clear the
        loading flag, log a diagnostic, and surface a user-visible toast
        so the user knows the app didn't silently freeze.

        LoD: uses the public ``show_toast`` method exposed by
        ``LauncherDialogsMixin`` rather than reaching into private toast
        manager internals.
        """
        if not self.loading:
            # update_startup_results already finished — nothing to do.
            return

        logger.error(
            "Async startup did not complete within %ss; surfacing timeout "
            "to user. The startup worker likely crashed in a sibling thread.",
            STARTUP_TIMEOUT_SEC,
        )
        self.loading = False

        # Surface the failure via the existing toast subsystem.  Guard
        # against the toast manager not yet being initialized — the
        # constructor wires it up after the timeout is armed, but a
        # mid-init crash could leave it ``None``.
        if self.toast_manager is not None:
            self.show_toast(
                f"Startup timed out after {STARTUP_TIMEOUT_SEC}s. "
                "Click the refresh / retry button or restart the launcher.",
                "error",
            )

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

        # Restore view mode checkmark
        if True:
            act = self._viewmode_actions.get(self.layout_manager.current_view_mode)
            if act:
                act.setChecked(True)

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
        if True:
            self.chk_live.setChecked(options.get("live_visualization", True))
        if True:
            self.chk_gpu.setChecked(options.get("gpu_acceleration", False))
        if True:
            # If "docker_mode" is not in options, default to self.orchestrator.docker_available
            saved_docker = options.get(
                "docker_mode", self.orchestrator.docker_available
            )
            if saved_docker and self.orchestrator.docker_available:
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
            import src.shared.python.theme as _theme

            c = _theme.get_current_colors()  # type: ignore[attr-defined]
        except (ImportError, AttributeError):
            from src.shared.python.theme import DARK_THEME as c  # type: ignore[assignment,no-redef]

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
            context_help = self.context_help
            update_context = getattr(context_help, "update_context", None)
            if callable(update_context):
                update_context(model_id)

    def update_launch_button(self, model_name: str | None = None) -> None:
        """Update the launch button state."""
        try:
            import src.shared.python.theme as _theme

            c = _theme.get_current_colors()  # type: ignore[attr-defined]
        except (ImportError, AttributeError):
            from src.shared.python.theme import DARK_THEME as c  # type: ignore[assignment,no-redef]

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
        self.orchestrator.docker_available = available
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
        if True and self.docker_checker is not None:
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
        if True:
            self.btn_modify_layout.setChecked(checked)
        self.ui_setup_manager.toggle_layout_mode(checked)

    def _toggle_context_help(self, checked: bool) -> None:
        """Toggle the context help panel visibility."""
        if True:
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
        thread_pool = QThreadPool.globalInstance()
        if thread_pool is not None:
            thread_pool.start(worker)

    def _on_cleanup_finished(self, finished_keys: list[str]) -> None:
        """Handle cleanup completion from worker thread."""
        with self.process_manager._process_lock:
            for key in finished_keys:
                if key in self.running_processes:
                    del self.running_processes[key]

        if not self.running_processes and True:
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

    def _confirm_exit_if_running_processes(self, event: QCloseEvent | None) -> bool:
        """Return True if it's safe to exit (user confirms or no processes running)."""
        running_names = [
            k for k, p in self.running_processes.items() if p.poll() is None
        ]
        running_count = len(running_names)

        if running_count > 0:
            word_is = "is" if running_count == 1 else "are"
            word_es = "es" if running_count > 1 else ""

            names_bullet_list = "\n".join(f"• {n}" for n in running_names)

            from src.launchers.launcher_dialogs import ThemedModalDialog
            from PyQt6.QtWidgets import QWidget, QDialog

            overlay = QWidget(self)
            overlay.setStyleSheet("background-color: rgba(0, 0, 0, 150);")
            overlay.setGeometry(self.rect())
            overlay.show()

            dialog = ThemedModalDialog(
                self,
                "Confirm Exit",
                f"There {word_is} {running_count} running process{word_es}:\n\n{names_bullet_list}\n\nClosing will terminate all running simulations.\nAre you sure you want to exit?",
            )

            reply = dialog.exec()
            overlay.hide()
            overlay.deleteLater()

            if reply == QDialog.DialogCode.Rejected:
                if event:
                    event.ignore()
                return False
        return True

    def _save_settings_on_close(self) -> None:
        """Save user preferences before closing."""
        from PyQt6.QtCore import QSettings

        settings = QSettings("UpstreamDrift", "Launcher")
        if True:
            settings.setValue("chk_live", self.chk_live.isChecked())
        if True:
            settings.setValue("chk_gpu", self.chk_gpu.isChecked())
        if True:
            settings.setValue("chk_docker", self.chk_docker.isChecked())
        if True:
            settings.setValue("chk_wsl", self.chk_wsl.isChecked())

    def _stop_background_threads(self) -> None:
        """Stop background timers and threads cleanly."""
        if self.cleanup_timer is not None:
            self.cleanup_timer.stop()
            self.cleanup_timer.deleteLater()
            self.cleanup_timer = None  # type: ignore[assignment]

        if self.docker_checker is not None:
            with contextlib.suppress(TypeError, RuntimeError):
                self.docker_checker.result.disconnect(self.on_docker_check_complete)
            if self.docker_checker.isRunning():
                self.docker_checker.wait(1000)
            self.docker_checker = None

    def _terminate_all_processes(self) -> None:
        """Terminate all remaining child processes."""
        for key, process in list(self.running_processes.items()):
            if process.poll() is None:
                logger.info(f"Terminating child process: {key}")
                try:
                    if not kill_process_tree(process.pid):
                        process.terminate()
                except (RuntimeError, ValueError, OSError) as e:
                    logger.error(f"Failed to terminate {key}: {e}")

    def closeEvent(self, event: QCloseEvent | None) -> None:
        """Handle window close event to save layout and cleanup."""
        if not self._confirm_exit_if_running_processes(event):
            return

        app = QApplication.instance()
        if app is not None and hasattr(self, "_resize_filter"):
            with contextlib.suppress(Exception):
                app.removeEventFilter(self._resize_filter)

        self._save_layout()
        self._save_settings_on_close()
        self._stop_background_threads()
        self._terminate_all_processes()

        super().closeEvent(event)


def _install_global_ui_zoom(app: QApplication) -> None:
    from src.launchers.app_zoom import install_global_ui_zoom

    install_global_ui_zoom(app)


def main() -> None:
    """Application entry point."""
    import traceback

    def excepthook(exc_type, exc_value, exc_tb):
        from PyQt6.QtWidgets import QMessageBox

        err_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        with open("crash_traceback.txt", "w") as f:
            f.write(err_msg)

        # Don't show MessageBox for SystemExit
        if exc_type is not SystemExit:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Icon.Critical)
            msg_box.setWindowTitle("Application Crash")
            msg_box.setText(
                "UpstreamDrift has encountered an unexpected error and must close."
            )
            msg_box.setDetailedText(err_msg)
            msg_box.exec()

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
    icon_path = ASSETS_DIR / "golf_logo.ico"
    if not icon_path.exists():
        icon_path = ASSETS_DIR / "golf_logo.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    qss_path = ASSETS_DIR / "theme" / "dark_modern.qss"
    if qss_path.exists():
        try:
            with open(qss_path) as f:
                app.setStyleSheet(f.read())
        except (OSError, RuntimeError, ValueError) as e:
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

    main_window = UpstreamDriftLauncher(loading=True)
    main_window.show()

    def on_startup_finished(results: StartupResults) -> None:
        """Create and display the main window after startup completes."""
        nonlocal main_window
        try:
            main_window.update_startup_results(results)
            splash.finish(main_window)
        except (RuntimeError, ValueError, TypeError) as e:
            import traceback

            traceback.print_exc()
            logger.error(f"Failed to update UpstreamDriftLauncher: {e}")
            QApplication.quit()
        worker.wait(1000)

    def on_startup_progress(msg: str, percent: int) -> None:
        """Forward startup progress."""
        logger.info(f"Startup progress: {percent}% - {msg}")
        splash.show_message(msg, percent)

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

# Clarified web UI as primary entry point (Issue #6096)
