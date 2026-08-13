# mypy: disable-error-code="attr-defined,arg-type,assignment"

"""Sidekick sidebar installation helpers for the UpstreamDrift launcher."""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
from pathlib import Path
from typing import Any

from src.launchers.launcher_constants import REPOS_ROOT, logger
from src.launchers.launcher_manager_attrs import forward_manager_attribute
from src.launchers.sidekick_readiness import readiness_detail_for_log
from src.launchers.sidekick_extension_overlay import (
    IncompleteParentSidekickRuntimeError,
    ManifestGatedSidekickFinder,
    install_manifest_gated_sidekick_extensions,
    validate_parent_sidekick_runtime,
)
from src.launchers.tools_repo_path import resolve_tools_source_root

_SOURCE_EXTENSION_FINDER: ManifestGatedSidekickFinder | None = None
_SOURCE_EXTENSION_PARENT: Path | None = None

SIDEKICK_API_READY_TIMEOUT_SEC: float = 45.0
SIDEKICK_API_READY_RETRY_MS: int = 500
SIDEKICK_API_RESTART_DELAY_MS: int = 1_000
SIDEKICK_API_HEALTHCHECK_MS: int = 3_000
SIDEKICK_API_MAX_RESTARTS: int = 2


def _activate_source_extensions(tools_source_root: Path) -> None:
    """Install the exact-module overlay for an UpstreamDrift source checkout."""
    global _SOURCE_EXTENSION_FINDER, _SOURCE_EXTENSION_PARENT
    manifest = REPOS_ROOT / "scripts/config/shared_python_ownership_exceptions.yaml"
    local_python = REPOS_ROOT / "src/shared/python"
    if not manifest.is_file() or not local_python.is_dir():
        return

    parent_python = (tools_source_root / "shared/python").resolve()
    if _SOURCE_EXTENSION_FINDER is not None:
        if parent_python != _SOURCE_EXTENSION_PARENT:
            raise RuntimeError(
                "Sidekick parent authority cannot change after extension loading"
            )
        return
    validate_parent_sidekick_runtime(parent_python)
    _SOURCE_EXTENSION_FINDER = install_manifest_gated_sidekick_extensions(
        local_python_root=local_python,
        parent_python_root=parent_python,
        manifest_path=manifest,
    )
    _SOURCE_EXTENSION_PARENT = parent_python


class SidekickSidebarManager:
    """Mixin-style manager for launcher Sidekick sidebar integration."""

    def __init__(self, launcher: Any) -> None:
        self.launcher = launcher

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(self.launcher, name)
        except AttributeError as err:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from err

    def __setattr__(self, name: str, value: Any) -> None:
        forward_manager_attribute(self, name, value)

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

            show_onboarding_if_needed(self.launcher)
        except ImportError as exc:
            logger.debug("Onboarding dialog not available: %s", exc)

    def _monitor_sidekick_api_readiness(
        self,
        *,
        readiness_check: Any,
        schedule_once: Any,
        monotonic: Any,
    ) -> None:
        """Advance the bounded API readiness state machine for the host.

        The launcher supplies the clock, readiness probe, and Qt scheduler so
        this manager stays independently testable and does not acquire a Qt
        import-time dependency.
        """
        launcher = self.launcher
        if not getattr(launcher, "_sidekick_api_monitoring", True):
            return

        runtime = launcher._sidekick_runtime_config
        expected_instance_id = runtime.instance_id if runtime is not None else None
        readiness = readiness_check(expected_instance_id=expected_instance_id)
        if runtime is None:
            launcher._report_sidekick_api_failure(readiness)
            return
        if readiness.ready:
            launcher._sidekick_api_wait_started_at = None
            launcher._sidekick_api_restart_count = 0
            if not getattr(launcher, "_sidekick_api_was_ready", False):
                logger.info("Sidekick API is ready: %s", readiness.url)
            launcher._sidekick_api_was_ready = True
            schedule_once(
                SIDEKICK_API_HEALTHCHECK_MS,
                launcher._monitor_sidekick_api_readiness,
            )
            return

        launcher._sidekick_api_was_ready = False
        now = monotonic()
        if launcher._sidekick_api_wait_started_at is None:
            launcher._sidekick_api_wait_started_at = now

        elapsed = now - launcher._sidekick_api_wait_started_at
        process = getattr(launcher, "background_api_process", None)
        process_running = process is not None and process.poll() is None
        if elapsed < SIDEKICK_API_READY_TIMEOUT_SEC and process_running:
            logger.info(
                "Waiting for Sidekick Chat API readiness: %s",
                readiness_detail_for_log(readiness),
            )
            schedule_once(
                SIDEKICK_API_READY_RETRY_MS,
                launcher._monitor_sidekick_api_readiness,
            )
            return

        if (
            not process_running
            and launcher._sidekick_runtime_config is not None
            and launcher._sidekick_api_restart_count < SIDEKICK_API_MAX_RESTARTS
        ):
            launcher._sidekick_api_restart_count += 1
            logger.warning(
                "Restarting failed Sidekick API child (%s/%s): %s",
                launcher._sidekick_api_restart_count,
                SIDEKICK_API_MAX_RESTARTS,
                readiness_detail_for_log(readiness),
            )
            launcher.background_api_process = (
                launcher._restart_sidekick_background_api()
            )
            launcher._sidekick_api_wait_started_at = now
            schedule_once(
                SIDEKICK_API_RESTART_DELAY_MS,
                launcher._monitor_sidekick_api_readiness,
            )
            return

        launcher._report_sidekick_api_failure(readiness)

    def _report_sidekick_api_failure(self, readiness: Any) -> None:
        """Surface terminal API failure while leaving local tools available."""
        launcher = self.launcher
        logger.warning(
            "Sidekick Chat remains degraded after API startup failed: %s",
            readiness_detail_for_log(readiness),
        )
        show_toast = getattr(launcher, "show_toast", None)
        if callable(show_toast):
            configuration_detail = (
                f" Configuration error: {launcher._sidekick_runtime_error}"
                if launcher._sidekick_runtime_error
                else ""
            )
            show_toast(
                "Sidekick tools are available, but Chat could not connect to "
                f"its local API at {readiness.url}.{configuration_detail}",
                "warning",
            )

    def _seed_sidekick_workspace(self) -> None:
        """Seed the active sidebar registry with launcher-owned model state."""
        launcher = self.launcher
        sidebar = launcher.sidekick_sidebar
        if sidebar is None:
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

        orchestrator = launcher.orchestrator
        if orchestrator.engine_manager is not None:
            set_variable("engine_manager", orchestrator.engine_manager)
        if orchestrator.registry is not None:
            set_variable("model_registry", orchestrator.registry)
        logger.info("Sidekick workspace seeded with engine_manager and model_registry")

    @staticmethod
    def _should_skip_onboarding() -> bool:
        """Return true when modal onboarding would block a non-interactive run."""
        return (
            os.environ.get("UPSTREAMDRIFT_DISABLE_ONBOARDING") == "1"
            or "PYTEST_CURRENT_TEST" in os.environ
            or "pytest" in sys.modules
        )

    def _get_sidekick_module(self) -> Any | None:
        """Import the Sidekick sidebar module, trying multiple fallback paths."""
        try:
            SidekickSidebarManager._install_sidekick_import_paths(self)
        except IncompleteParentSidekickRuntimeError as exc:
            if os.environ.get("TOOLS_REPO_PATH"):
                raise
            logger.warning(
                "Sidekick sidebar disabled because the implicit Tools runtime "
                "is unavailable: %s",
                exc,
            )
            return None
        try:
            return importlib.import_module("sidekick.ui.tools_sidebar.api")
        except ImportError:
            pass

        try:
            from src.shared.python.gui_launcher.tools_sidebar_integration import (
                _import_sidebar_module,
            )
        except ImportError as exc:
            logger.debug("Sidekick integration shim not importable: %s", exc)
            return None

        module = _import_sidebar_module()
        if module is not None:
            return module

        for module_name in (
            "shared.python.sidekick.ui.tools_sidebar",
            "sidekick.ui.tools_sidebar",
        ):
            try:
                return importlib.import_module(module_name)
            except ImportError:
                continue
        return None

    def _install_sidekick_import_paths(self) -> None:
        """Prepend the configured, vendored, or sibling Tools source."""
        source_root = resolve_tools_source_root(
            REPOS_ROOT,
            os.environ.get("TOOLS_REPO_PATH"),
        )
        SidekickSidebarManager._prepend_tools_source_paths(source_root)
        _activate_source_extensions(source_root)

    @staticmethod
    def _prepend_tools_source_paths(source_root: Path) -> None:
        """Install one selected Tools source without changing its authority."""
        SidekickSidebarManager._prepend_sys_path(source_root)
        SidekickSidebarManager._prepend_sys_path(source_root / "shared" / "python")

    @staticmethod
    def _prepend_sys_path(path: Any) -> None:
        """Place a source path first even when bootstrap already added it.

        The direct ``shared/python`` packages must win over legacy alias shims
        under ``src``. Merely skipping an existing entry preserves the wrong
        order when another bootstrapper inserted ``src`` first.
        """
        path_text = str(path)
        while path_text in sys.path:
            sys.path.remove(path_text)
        sys.path.insert(0, path_text)

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
            return factory(parent=self.launcher, project_root=str(REPOS_ROOT))
        except TypeError:
            try:
                return factory(parent=self.launcher)
            except (RuntimeError, ValueError) as exc:
                logger.warning("Sidekick factory call failed: %s", exc)
                return None
        except (RuntimeError, ValueError) as exc:
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
        service = None
        ensure_service = getattr(self, "_ensure_sidekick_action_service", None)
        if callable(ensure_service):
            service = ensure_service()
        setter = getattr(sidebar_widget, "set_action_service", None)
        if callable(setter) and service is not None:
            setter(service)
        SidekickSidebarManager._apply_sidekick_splitter_sizes(self)

        logger.info("Sidekick sidebar embedded in main splitter")

    def _install_sidekick_sidebar(self) -> None:
        """Embed the Sidekick multitab sidebar as a third splitter pane."""
        existing_sidebar = self.sidekick_sidebar
        if existing_sidebar is not None:
            existing_sidebar.setVisible(True)
            return
        logger.info("Initializing _install_sidekick_sidebar")
        module = SidekickSidebarManager._get_sidekick_module(self)
        widget = SidekickSidebarManager._create_sidekick_sidebar_widget(self, module)
        SidekickSidebarManager._embed_sidekick_sidebar_widget(self, widget)
