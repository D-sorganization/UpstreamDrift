"""Verify strict lazy-loading of engine dependencies in the launcher UI.

Issue #1956: Engine-specific Python packages (mujoco, pydrake, pinocchio, etc.)
must NOT be imported as a side-effect of importing the dashboard launcher
modules.  Heavy engine packages should only be loaded when the user explicitly
activates the corresponding engine entry point.

These tests snapshot ``sys.modules`` before and after importing each dashboard
module, then assert that none of the physics-engine packages were added.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest

# Heavy engine packages whose module-level presence indicates an eager import
_ENGINE_PACKAGES = [
    "mujoco",
    "pydrake",
    "pinocchio",
    "opensim",
    "myosim",
]

# Dashboard module names under test
_DASHBOARD_MODULES = [
    "src.launchers.mujoco_dashboard",
    "src.launchers.pinocchio_dashboard",
    "src.launchers.drake_dashboard",
]


def _engine_packages_in_modules(module_snapshot: set[str]) -> list[str]:
    """Return any engine packages present in the given module-name set."""
    return [pkg for pkg in _ENGINE_PACKAGES if pkg in module_snapshot]


class TestDashboardModulesDoNotEagerlyImportEngines:
    """Importing dashboard modules must not pull in heavy engine packages."""

    @pytest.mark.parametrize("dashboard_module", _DASHBOARD_MODULES)
    def test_importing_dashboard_does_not_load_engine_packages(
        self, dashboard_module: str
    ) -> None:
        """Importing a dashboard module must not trigger engine package imports."""
        # Remove the module (and any cached import) so we get a fresh import
        sys.modules.pop(dashboard_module, None)

        # Snapshot of engine packages present BEFORE the import
        before = {
            key
            for key in sys.modules
            if any(key == p or key.startswith(p + ".") for p in _ENGINE_PACKAGES)
        }

        # Patch PyQt6 so the import works even in headless environments
        with patch.dict(
            "sys.modules",
            {
                "PyQt6": sys.modules.get("PyQt6", MagicMock()),
                "PyQt6.QtWidgets": sys.modules.get("PyQt6.QtWidgets", MagicMock()),
            },
        ):
            importlib.import_module(dashboard_module)

        # Snapshot AFTER the import
        after = {
            key
            for key in sys.modules
            if any(key == p or key.startswith(p + ".") for p in _ENGINE_PACKAGES)
        }

        newly_imported = after - before
        msg = (
            f"Importing '{dashboard_module}' eagerly loaded engine package(s): "
            f"{newly_imported}. Move those imports inside the main() function."
        )
        assert not newly_imported, msg


class TestDashboardModulesHaveCallableMain:
    """Each dashboard module must expose a callable main() entry point."""

    @pytest.mark.parametrize("dashboard_module", _DASHBOARD_MODULES)
    def test_main_is_callable(self, dashboard_module: str) -> None:
        mod = importlib.import_module(dashboard_module)
        assert callable(
            getattr(mod, "main", None)
        ), f"{dashboard_module} does not expose a callable main()"
