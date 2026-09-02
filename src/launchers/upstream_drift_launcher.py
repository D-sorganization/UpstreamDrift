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
import time
from typing import Any

from PyQt6.QtCore import (
    QEventLoop,
    QThreadPool,
    QTimer,
    Qt,
)
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QApplication, QMainWindow

from src.launchers.docker_manager import DockerLauncher

from src.launchers.embedded_tool_bootstrap import bootstrap_embeddable_tools

from src.launchers.launcher_constants import (
    CONFIG_DIR,
    DOCKER_STAGES,
    GRID_COLUMNS,
    LAYOUT_CONFIG_FILE,
    REPOS_ROOT,
    _lazy_load_engine_manager,
    logger,
)

from src.launchers.launcher_dialogs import DialogsManager
from src.launchers.launcher_layout_manager import (
    LayoutManager,
    compute_centered_geometry,
)
from src.launchers.launcher_layout_persistence import (
    load_layout_state,
    save_layout_state,
)
from src.launchers.launcher_model_handlers import ModelHandlerRegistry
from src.launchers.launcher_orchestrator import LauncherOrchestrator
from src.launchers.launcher_process_cleanup_worker import ProcessCleanupWorker
from src.launchers.launcher_process_manager import ProcessManager
from src.launchers.launcher_simulation import (
    DEPENDENCY_MAP,
    dependency_cache_key,
    dependency_probe_key,
)
from src.launchers.launcher_ui.frameless_window import configure_frameless_window
from src.launchers.launcher_simulation import SimulationManager
from src.launchers.launcher_sidekick_sidebar import (
    SIDEKICK_API_HEALTHCHECK_MS as SIDEKICK_API_HEALTHCHECK_MS,
    SIDEKICK_API_MAX_RESTARTS as SIDEKICK_API_MAX_RESTARTS,
    SIDEKICK_API_READY_RETRY_MS as SIDEKICK_API_READY_RETRY_MS,
    SIDEKICK_API_READY_TIMEOUT_SEC as SIDEKICK_API_READY_TIMEOUT_SEC,
    SIDEKICK_API_RESTART_DELAY_MS as SIDEKICK_API_RESTART_DELAY_MS,
    SidekickSidebarManager,
)
from src.launchers.sidekick_readiness import (
    check_sidekick_api_readiness,
)
from src.launchers.sidekick_runtime import (
    SidekickRuntimeConfig,
    configure_sidekick_runtime,
    reselect_sidekick_runtime_port,
)
from src.launchers.tools_repo_path import resolve_tools_source_root
from src.launchers.launcher_theme import ThemeManager
from src.launchers.launcher_ui_setup import UISetupManager

from src.launchers.ui_components import (
    ASSETS_DIR,
    DockerCheckThread,
    DraggableModelCard,
    SplashScreen,
    StartupResults,
)

from src.shared.python.security.subprocess_utils import kill_process_tree
from src.shared.python.theme.style_constants import Styles
from src.shared.python.ui import (
    apply_window_icon,
)

# Windows taskbar identity. Declaring this (before any window is shown) is what
# makes the taskbar use the app icon instead of the generic python.exe icon —
# the piece earlier favicon fixes missed.
_APP_USER_MODEL_ID = "D-sorganization.UpstreamDrift"

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


class UpstreamDriftLauncher(QMainWindow):
    """Main application window for the launcher.

    Composes focused mixins for UI setup, theme management,
    simulation launching, and dialog/settings management.
    """

    sidekick_sidebar: Any | None
    sidekick_window: Any | None
    _sidekick_popped_out: bool
    _sidekick_needs_initial_sizing: bool
    _sidekick_action_service: Any | None
    _sidekick_action_service_host: Any | None

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
            "sidekick_sidebar_manager",
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
        self,
        startup_results: StartupResults | None = None,
        loading: bool = False,
        splash: SplashScreen | None = None,
    ) -> None:
        """Initialize the main window.

        Args:
            startup_results: Optional pre-loaded startup results from AsyncStartupWorker.
                            If provided, skips redundant loading of registry and engines.
        """
        super().__init__()

        self._init_core_managers()
        self.loading = loading
        self.splash = splash
        self.toast_manager = None
        self.orchestrator = LauncherOrchestrator()
        self._configure_window_frame()

        self._startup_time_ms = (
            startup_results.startup_time_ms if startup_results else 0
        )

        self._load_window_icon()
        self._init_state(startup_results)
        self._init_managers()
        self._init_layout_manager()
        if not self.loading:
            self._initialize_model_order()

        self.ui_setup_manager.init_ui()
        self.theme_manager._apply_theme_system()
        self._run_startup_mode(startup_results)
        self._load_layout()
        self._start_cleanup_timer()
        self.toast_manager = None
        self.ui_setup_manager._init_ui_components()

        if self._startup_time_ms > 0:
            logger.info("Application startup completed in %sms", self._startup_time_ms)

    def _init_core_managers(self) -> None:
        """Initialize launcher manager delegates."""
        self.ui_setup_manager = UISetupManager(self)
        self.theme_manager = ThemeManager(self)
        self.simulation_manager = SimulationManager(self)
        self.dialogs_manager = DialogsManager(self)
        self.sidekick_sidebar_manager = SidekickSidebarManager(self)

    def _configure_window_frame(self) -> None:
        """Configure frameless window chrome and initial geometry."""
        self.setWindowTitle("UpstreamDrift")
        self._resize_filter = configure_frameless_window(self)
        self.setMinimumSize(800, 600)
        self._resize_to_initial_screen()

    def _resize_to_initial_screen(self) -> None:
        """Size the launcher to 80% of the primary screen, capped at 1400x900."""
        screen = QApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            w = min(int(avail.width() * 0.80), 1400)
            h = min(int(avail.height() * 0.80), 900)
        else:
            w, h = 1280, 800
        self.resize(w, h)
        self.center_window()

    def _run_startup_mode(self, startup_results: StartupResults | None) -> None:
        """Schedule onboarding or async-startup handling based on launch mode."""
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

    def _start_cleanup_timer(self) -> None:
        """Start the non-blocking process cleanup timer."""
        self.cleanup_timer = QTimer(self)
        self.cleanup_timer.timeout.connect(self._schedule_cleanup)
        self.cleanup_timer.start(10000)

    def _apply_sidekick_splitter_sizes(self) -> None:
        """Expose the Sidekick splitter sizing hook on the launcher class."""
        return self.sidekick_sidebar_manager._apply_sidekick_splitter_sizes()

    def _install_sidekick_sidebar(self) -> None:
        """Expose the Sidekick sidebar installer on the launcher class."""
        return self.sidekick_sidebar_manager._install_sidekick_sidebar()

    def _show_onboarding_if_needed(self) -> None:
        """Expose the onboarding hook scheduled during launcher startup."""
        return self.sidekick_sidebar_manager._show_onboarding_if_needed()

    def _on_windows_mode_changed(self, state: int) -> None:
        """Expose the Windows-mode checkbox handler on the launcher class."""
        return self.dialogs_manager._on_windows_mode_changed(state)

    def showEvent(self, event: Any) -> None:
        """Force sidekick splitter sizes on first display."""
        super().showEvent(event)
        if self._sidekick_needs_initial_sizing:
            self._sidekick_needs_initial_sizing = False
            self._apply_sidekick_splitter_sizes()

    @property
    def sidekick_action_service(self) -> Any:
        """Launcher-owned Sidekick action service for embedded chat panels."""
        return self._ensure_sidekick_action_service()

    def _ensure_sidekick_action_service(self) -> Any:
        """Create or refresh the launcher Sidekick action service."""
        embedded_host = getattr(self, "embedded_host", None)
        if (
            self._sidekick_action_service is not None
            and self._sidekick_action_service_host is embedded_host
        ):
            return self._sidekick_action_service

        from src.launchers.sidekick_host_port import create_launcher_action_service

        workspace = None
        if embedded_host is not None:
            workspace = getattr(embedded_host, "launcher_context", None)
        if workspace is None:
            sidebar = getattr(self, "sidekick_sidebar", None)
            workspace = getattr(sidebar, "registry", None)

        service = create_launcher_action_service(
            launcher=self,
            embedded_host=embedded_host,
            workspace=workspace,
        )
        self._sidekick_action_service = service
        self._sidekick_action_service_host = embedded_host
        if embedded_host is not None:
            embedded_host.sidekick_action_service = service
        return service

    def _load_window_icon(self) -> None:
        # Sets the AppUserModelID (idempotent), the application icon, and this
        # window's icon. The AppUserModelID is what fixes the Windows taskbar
        # icon; setting only the window icon (as before) was not enough.
        apply_window_icon(
            app=QApplication.instance(),
            window=self,
            icon_candidates=[
                ASSETS_DIR / "golf_logo.ico",
                ASSETS_DIR / "golf_logo.png",
            ],
            app_id=_APP_USER_MODEL_ID,
        )

    def _init_state(self, startup_results: StartupResults | None) -> None:
        self.docker_checker: DockerCheckThread | None = None
        self.selected_model: str | None = None
        self.model_cards: dict[str, Any] = {}
        self.model_order: list[str] = []
        self.background_api_process: Any | None = None
        self._sidekick_runtime_config: SidekickRuntimeConfig | None = None
        self._sidekick_runtime_error = ""
        self._sidekick_api_wait_started_at: float | None = None
        self._sidekick_api_restart_count = 0
        self._sidekick_api_monitoring = True
        self._sidekick_api_was_ready = False
        self.layout_edit_mode = False
        self.current_filter_text = ""
        self._sidekick_needs_initial_sizing = True
        self.sidekick_sidebar = None
        self.sidekick_window = None
        self._sidekick_popped_out = False
        self._sidekick_action_service = None
        self._sidekick_action_service_host = None
        self._popped_out_windows: list[Any] = []
        self._dependency_status_cache: dict[str, tuple[bool, str]] = {}
        self._dependency_probe_workers: dict[str, Any] = {}
        self.orchestrator.initialize_from_results(startup_results)

    def _init_managers(self) -> None:
        self.ui_setup_manager._setup_process_console()
        self.process_manager = ProcessManager(
            REPOS_ROOT,
            output_callback=self.ui_setup_manager._on_process_output,
        )
        self.process_manager.on_process_list_changed = (  # type: ignore[attr-defined]
            self.ui_setup_manager.update_running_processes_ui
        )
        self.model_handler_registry = ModelHandlerRegistry()
        self.docker_launcher = DockerLauncher(REPOS_ROOT)
        self.running_processes = self.process_manager.running_processes

        # Activate approved Upstream-owned extensions before adapters import
        # canonical Tools packages during registry bootstrap.
        self.sidekick_sidebar_manager._install_sidekick_import_paths()

        # Bootstrap embeddable tools registry (fixes #5049)
        # This ensures EMBEDDABLE_TOOL_REGISTRY is populated before any
        # context menus or embedded host widgets are created
        bootstrap_embeddable_tools()

        if "PYTEST_CURRENT_TEST" not in os.environ:
            try:
                self._sidekick_runtime_config = configure_sidekick_runtime(os.environ)
            except (TypeError, ValueError) as exc:
                self._sidekick_runtime_error = str(exc)
                logger.error("Sidekick runtime configuration failed: %s", exc)
            else:
                self.background_api_process = self._launch_sidekick_background_api()

    def _launch_sidekick_background_api(self) -> Any | None:
        """Launch the API child using the already-exported runtime contract."""
        cwd = (
            REPOS_ROOT / "UpstreamDrift"
            if (REPOS_ROOT / "UpstreamDrift").exists()
            else REPOS_ROOT
        )
        tools_source = resolve_tools_source_root(
            REPOS_ROOT,
            os.environ.get("TOOLS_REPO_PATH"),
        )
        return self.process_manager.launch_module(
            name="background_api_server",
            module_name="src.api.server",
            cwd=cwd,
            extra_python_paths=(
                tools_source / "shared" / "python",
                tools_source,
            ),
        )

    def _restart_sidekick_background_api(self) -> Any | None:
        """Refresh the dynamic port contract before replacing the API child."""
        runtime = self._sidekick_runtime_config
        if runtime is None:
            return None
        self._sidekick_runtime_config = reselect_sidekick_runtime_port(
            runtime,
            os.environ,
        )
        return self._launch_sidekick_background_api()

    def _create_category_header(self, title: str) -> Any:
        from PyQt6.QtWidgets import QLabel

        from src.shared.python.theme.typography import Weights, get_display_font

        try:
            import src.shared.python.theme as _theme

            c = _theme.get_current_colors()  # type: ignore[attr-defined]
        except (ImportError, AttributeError):
            from src.shared.python.theme.palette import (  # type: ignore[assignment,no-redef]
                DARK_THEME as c,
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
        save_layout_state(self)

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
                    # Hide before deleteLater() so widgets that stop
                    # per-widget resources in hideEvent (e.g. SkeletonCard's
                    # pulse animation, #8906) actually get a chance to.
                    widget.hide()
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

        # Update Clear Filters button visibility
        has_filter = False
        if hasattr(self, "layout_manager") and self.layout_manager:
            cat = getattr(self.layout_manager, "current_category_filter", "All")
            txt = getattr(self.layout_manager, "current_filter_text", "")
            if cat != "All" or txt:
                has_filter = True
        if (
            hasattr(self.ui_setup_manager, "btn_clear_filters")
            and self.ui_setup_manager.btn_clear_filters
        ):
            self.ui_setup_manager.btn_clear_filters.setVisible(has_filter)

    def update_startup_results(self, results: StartupResults) -> None:
        """Transition from loading skeleton to full application."""
        self._startup_time_ms = results.startup_time_ms
        self.orchestrator.initialize_from_results(results)
        self.layout_manager.available_models = self.orchestrator.available_models
        # Clear the loading flag BEFORE building the grid: _rebuild_grid()
        # renders placeholder SkeletonCards while self.loading is True, so if
        # the flag is still set when _load_layout() rebuilds, the Home view is
        # left showing skeletons until the next rebuild trigger (e.g. a sidebar
        # click). Flipping it first makes the startup rebuild render real cards.
        self.loading = False
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
        """Install local tools immediately and monitor Chat independently."""
        self._install_sidekick_sidebar()
        self._seed_sidekick_workspace()
        self._monitor_sidekick_api_readiness()

    def _monitor_sidekick_api_readiness(self) -> None:
        """Probe API readiness off the GUI thread, then advance the monitor.

        The blocking HTTP probe runs on a ``SidekickReadinessProbeThread``
        (issue #8939); its result is marshalled back to the GUI thread via a
        signal and fed to the sidebar manager's readiness state machine.
        """
        if not getattr(self, "_sidekick_api_monitoring", True):
            return
        from src.launchers.sidekick_readiness_worker import (
            SidekickReadinessProbeThread,
        )

        runtime = self._sidekick_runtime_config
        expected_instance_id = runtime.instance_id if runtime is not None else None
        worker = SidekickReadinessProbeThread(
            probe=check_sidekick_api_readiness,
            expected_instance_id=expected_instance_id,
        )
        worker.readiness_ready.connect(self._on_sidekick_readiness_result)
        worker.finished.connect(worker.deleteLater)
        # Keep a reference so the thread is not garbage-collected mid-probe.
        self._sidekick_readiness_worker = worker
        worker.start()

    def _on_sidekick_readiness_result(self, readiness: Any) -> None:
        """Advance the readiness state machine with an already-probed result."""
        self.sidekick_sidebar_manager._monitor_sidekick_api_readiness(
            readiness_check=lambda *, expected_instance_id=None: readiness,
            schedule_once=QTimer.singleShot,
            monotonic=time.monotonic,
        )

    def _report_sidekick_api_failure(self, readiness: Any) -> None:
        """Delegate terminal readiness reporting to the Sidekick owner."""
        self.sidekick_sidebar_manager._report_sidekick_api_failure(readiness)

    def _seed_sidekick_workspace(self) -> None:
        """Delegate workspace seeding to the Sidekick owner."""
        self.sidekick_sidebar_manager._seed_sidekick_workspace()

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
        splash = getattr(self, "splash", None)
        if splash is not None:
            splash.close()

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
        load_layout_state(self)

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
        colors = self._selection_theme_colors()

        for mid, card in self.model_cards.items():
            self._set_model_card_selected(card, mid == model_id, colors)

        model = self._get_model(model_id)
        if model:
            self.update_launch_button(model.name)
            self._update_selection_context(model_id)
            self._check_selected_model_dependencies(model_id, model)

    def _selection_theme_colors(self) -> Any:
        """Return current colors for model-card selection styling."""
        try:
            import src.shared.python.theme as _theme

            return _theme.get_current_colors()  # type: ignore[attr-defined]
        except (ImportError, AttributeError):
            from src.shared.python.theme.palette import DARK_THEME

            return DARK_THEME

    def _set_model_card_selected(self, card: Any, selected: bool, colors: Any) -> None:
        """Apply selected state to either modern or legacy model cards."""
        if hasattr(card, "set_selected"):
            card.set_selected(selected)
            return
        if selected:
            card.setStyleSheet(f"""
                        QFrame#ModelCard {{
                            background-color: {colors.bg_highlight};
                            border: 2px solid {colors.primary};
                            border-radius: 12px;
                        }}
                        """)
            return
        card.setStyleSheet(f"""
                        QFrame#ModelCard {{
                            background-color: {colors.bg_elevated};
                            border: 1px solid {colors.border_default};
                            border-radius: 12px;
                        }}
                        QFrame#ModelCard:hover {{
                            background-color: {colors.bg_highlight};
                            border: 1px solid {colors.border_strong};
                        }}
                        """)

    def _update_selection_context(self, model_id: str) -> None:
        """Update contextual help for the selected model."""
        context_help = self.context_help
        update_context = getattr(context_help, "update_context", None)
        if callable(update_context):
            update_context(model_id)

    def _check_selected_model_dependencies(self, model_id: str, model: Any) -> None:
        """Run dependency checks for the selected model when needed."""
        use_wsl = hasattr(self, "chk_wsl") and self.chk_wsl.isChecked()
        use_docker = hasattr(self, "chk_docker") and self.chk_docker.isChecked()
        if use_wsl or use_docker:
            return

        cache_key = dependency_cache_key(model)
        key = dependency_probe_key(model)
        if key not in DEPENDENCY_MAP:
            return

        if cache_key not in self._dependency_status_cache:
            self.lbl_status.setText(f"> Checking {model.name} dependencies...")
            self.lbl_status.setStyleSheet(Styles.STATUS_WARNING)
            if hasattr(self, "btn_launch"):
                self.btn_launch.setEnabled(False)
            self._start_dependency_probe(cache_key, model)
            return
        deps_ok, deps_error = self._dependency_status_cache[cache_key]

        if deps_ok:
            self._set_dependency_success_status()
        else:
            self._set_dependency_error_status(model, key, deps_error, DEPENDENCY_MAP)

    def _set_dependency_success_status(self) -> None:
        """Mark selected-model dependencies as satisfied."""
        self.lbl_status.setText("Ready")
        self.lbl_status.setStyleSheet(Styles.STATUS_SUCCESS)
        self.lbl_status.setCursor(Qt.CursorShape.ArrowCursor)
        self.lbl_status.setToolTip("")

    def _set_dependency_error_status(
        self,
        model: Any,
        key: str,
        deps_error: str,
        dependency_map: dict[str, Any],
    ) -> None:
        """Surface dependency failure details for the selected model."""
        dep_info = dependency_map.get(key, {})
        dep_name = dep_info.get("display_name", key)
        install_cmd = dep_info.get("install_cmd", "")
        doc_url = dep_info.get("doc_url", "")

        if "PYTEST_CURRENT_TEST" not in os.environ and not getattr(
            self, "loading", False
        ):
            self.show_dependency_error(
                model.name,
                dep_name,
                install_cmd,
                doc_url,
                deps_error,
            )
        self.lbl_status.setText("! Dependency Error")
        self.lbl_status.setStyleSheet(Styles.STATUS_ERROR)
        self.lbl_status.setCursor(Qt.CursorShape.PointingHandCursor)
        self.lbl_status.setToolTip("Click to view details in Settings -> Configuration")

    def update_launch_button(self, model_name: str | None = None) -> None:
        """Update the launch button state."""
        try:
            import src.shared.python.theme as _theme

            c = _theme.get_current_colors()  # type: ignore[attr-defined]
        except (ImportError, AttributeError):
            from src.shared.python.theme.palette import (  # type: ignore[assignment,no-redef]
                DARK_THEME as c,
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
            and not self.orchestrator.docker_available
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
        success_hover = getattr(c, "success_hover", c.success)
        self.btn_launch.setStyleSheet(f"""
            QPushButton {{
                background-color: {c.success};
                color: white;
                border-radius: 6px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {success_hover};
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
        """Toggle layout edit mode from the ``View > Edit Layout Mode`` action.

        The checkable ``_action_layout_mode`` QAction is the sole owner of this
        toggle's state; ``toggle_layout_mode`` keeps it in sync. A vestigial
        ``btn_modify_layout`` toolbar button used to mirror the state, but it
        has not been created by the UI setup for a long time — dereferencing it
        raised ``AttributeError`` inside a Qt slot, which PyQt turns into a hard
        abort (``0xC0000409``). See issue #8023.
        """
        self.ui_setup_manager.toggle_layout_mode(checked)

    def _toggle_context_help(self, checked: bool) -> None:
        """Toggle the context help panel visibility.

        Guarded because ``context_help`` is only created by
        ``UISetupManager._setup_context_help``; a launcher built without the
        full UI must not abort the process here (issue #8023).
        """
        context_help = getattr(self, "context_help", None)
        if context_help is None:
            logger.warning("Context help panel is not available; ignoring toggle.")
            return
        if checked:
            context_help.show()
        else:
            context_help.hide()

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

        self.ui_setup_manager.update_running_processes_ui()

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
            k
            for k, p in self.running_processes.items()
            if p.poll() is None and k != "background_api_server"
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
        settings.setValue("chk_live", self.chk_live.isChecked())
        settings.setValue("chk_gpu", self.chk_gpu.isChecked())
        settings.setValue("chk_docker", self.chk_docker.isChecked())
        settings.setValue("chk_wsl", self.chk_wsl.isChecked())

    def _stop_background_threads(self) -> None:
        """Stop background timers and threads cleanly."""
        self._sidekick_api_monitoring = False
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

    def _shutdown_sidekick_sidebar(self) -> None:
        """Stop Sidekick-owned runtimes before the host window disappears."""
        sidebar = getattr(self, "sidekick_sidebar", None)
        shutdown = getattr(sidebar, "shutdown", None)
        if callable(shutdown):
            shutdown()

    def closeEvent(self, event: QCloseEvent | None) -> None:
        """Handle window close event to save layout and cleanup."""
        if not self._confirm_exit_if_running_processes(event):
            return

        self._save_layout()
        self._save_settings_on_close()
        self._stop_background_threads()
        self._shutdown_sidekick_sidebar()
        self._terminate_all_processes()

        super().closeEvent(event)


def main() -> None:
    from src.launchers.upstream_drift_launcher_main import main as launcher_main

    launcher_main()


if __name__ == "__main__":
    main()

# Clarified web UI as primary entry point (Issue #6096)
