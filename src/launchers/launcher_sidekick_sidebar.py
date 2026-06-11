# mypy: disable-error-code="attr-defined,arg-type,assignment"

"""Sidekick sidebar installation helpers for the UpstreamDrift launcher."""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
from typing import Any

from src.launchers.launcher_constants import REPOS_ROOT, logger


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
        if name == "launcher" or hasattr(type(self), name) or name in self.__dict__:
            super().__setattr__(name, value)
        elif hasattr(self.launcher, name):
            setattr(self.launcher, name, value)
        else:
            super().__setattr__(name, value)

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
            from src.shared.python.gui_launcher.tools_sidebar_integration import (
                _import_sidebar_module,
            )
        except ImportError as exc:
            logger.debug("Sidekick integration shim not importable: %s", exc)
            return None

        module = _import_sidebar_module()
        if module is not None:
            return module

        self._install_sidekick_import_paths()
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
        """Prepend sibling or vendored Tools paths for Sidekick imports."""
        sibling_tools = REPOS_ROOT.parent / "Tools"
        if sibling_tools.is_dir():
            SidekickSidebarManager._prepend_sys_path(sibling_tools / "src")
            SidekickSidebarManager._prepend_sys_path(
                sibling_tools / "src" / "shared" / "python"
            )
            return

        vendor_root = REPOS_ROOT / "vendor" / "ud-tools" / "src"
        SidekickSidebarManager._prepend_sys_path(vendor_root)
        SidekickSidebarManager._prepend_sys_path(vendor_root / "shared" / "python")

    @staticmethod
    def _prepend_sys_path(path: Any) -> None:
        """Prepend a path string to sys.path if it is not already present."""
        path_text = str(path)
        if path_text not in sys.path:
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
        self._apply_sidekick_splitter_sizes()

        logger.info("Sidekick sidebar embedded in main splitter")

    def _install_sidekick_sidebar(self) -> None:
        """Embed the Sidekick multitab sidebar as a third splitter pane."""
        logger.info("Initializing _install_sidekick_sidebar")
        module = self._get_sidekick_module()
        widget = self._create_sidekick_sidebar_widget(module)
        self._embed_sidekick_sidebar_widget(widget)
