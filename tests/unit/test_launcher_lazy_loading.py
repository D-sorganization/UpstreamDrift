"""Tests verifying strict lazy-loading of engine dependencies in dashboard launchers.

GH1956: Consolidate GUI Interfaces and ensure strict lazy-loading of engine
dependencies. These tests confirm that importing a dashboard launcher module
does NOT trigger its heavy physics engine import at module level.
"""

import sys
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

# Pre-import dashboard launcher to populate sys.modules with heavy dependencies (scipy, numpy, pyqt)
# so that patch.dict("sys.modules") doesn't remove them and trigger C-extension reload errors.
import src.shared.python.dashboard.launcher  # noqa: F401

# Dashboard launcher modules these tests evict from ``sys.modules`` in order to
# force a fresh import.
_DASHBOARD_MODULES = (
    "src.launchers.mujoco_dashboard",
    "src.launchers.pinocchio_dashboard",
    "src.launchers.drake_dashboard",
)


@pytest.fixture(autouse=True)
def _restore_dashboard_modules() -> Generator[None, None, None]:
    """Put the original dashboard modules back after each test.

    These tests deliberately evict launcher modules from ``sys.modules`` so the
    next import runs fresh. Leaving the hole behind is what made issue #9183
    intermittent: a later test that had already bound ``main`` at import time
    kept a reference into the *old* module dict, while ``mock.patch`` re-imported
    the module and patched the *new* one. The patch silently missed, the real
    ``launch_dashboard`` ran, and CI hung in a Qt event loop until it timed out.

    Restoring the original module objects keeps every previously-bound
    reference and its module dict in agreement.
    """
    saved = {
        name: sys.modules[name] for name in _DASHBOARD_MODULES if name in sys.modules
    }
    yield
    for name, module in saved.items():
        sys.modules[name] = module


class TestMuJoCoDashboardLazyLoading:
    """Verify mujoco_dashboard does not import engine at module load time."""

    def test_module_importable_without_mujoco_installed(self) -> None:
        """mujoco_dashboard can be imported even when the mujoco engine module is absent."""
        engine_module = (
            "src.engines.physics_engines.mujoco.python"
            ".mujoco_humanoid_golf.physics_engine"
        )
        # Remove any cached version of the dashboard module to force a fresh import
        sys.modules.pop("src.launchers.mujoco_dashboard", None)

        with patch.dict("sys.modules", {engine_module: None}):
            # If the import were at module level, this would raise ImportError.
            import src.launchers.mujoco_dashboard  # noqa: F401

        # Clean up so other tests are unaffected
        sys.modules.pop("src.launchers.mujoco_dashboard", None)

    def test_engine_not_imported_at_module_level(self) -> None:
        """Importing mujoco_dashboard must not touch the MuJoCo physics engine module."""
        engine_module = (
            "src.engines.physics_engines.mujoco.python"
            ".mujoco_humanoid_golf.physics_engine"
        )
        # Remove cached module so the import is fresh
        sys.modules.pop("src.launchers.mujoco_dashboard", None)
        sys.modules.pop(engine_module, None)

        mock_engine = MagicMock()
        with patch.dict("sys.modules", {engine_module: mock_engine}):
            import src.launchers.mujoco_dashboard  # noqa: F401

            # The mock was placed in sys.modules, but the dashboard should not have
            # accessed any of its attributes at import time.
            assert not mock_engine.MuJoCoPhysicsEngine.called, (
                "MuJoCoPhysicsEngine was accessed at module-import time — "
                "the import must be deferred to main()"
            )

        sys.modules.pop("src.launchers.mujoco_dashboard", None)

    def test_dashboard_module_has_main(self) -> None:
        """mujoco_dashboard exposes a callable main() function."""
        sys.modules.pop("src.launchers.mujoco_dashboard", None)
        import src.launchers.mujoco_dashboard as mod

        assert callable(mod.main)
        sys.modules.pop("src.launchers.mujoco_dashboard", None)


class TestPinocchioDashboardLazyLoading:
    """Verify pinocchio_dashboard does not import engine at module load time."""

    def test_module_importable_without_pinocchio_installed(self) -> None:
        """pinocchio_dashboard can be imported even when the pinocchio engine module is absent."""
        engine_module = (
            "src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine"
        )
        sys.modules.pop("src.launchers.pinocchio_dashboard", None)

        with patch.dict("sys.modules", {engine_module: None}):
            import src.launchers.pinocchio_dashboard  # noqa: F401

        sys.modules.pop("src.launchers.pinocchio_dashboard", None)

    def test_engine_not_imported_at_module_level(self) -> None:
        """Importing pinocchio_dashboard must not touch the Pinocchio physics engine module."""
        engine_module = (
            "src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine"
        )
        sys.modules.pop("src.launchers.pinocchio_dashboard", None)
        sys.modules.pop(engine_module, None)

        mock_engine = MagicMock()
        with patch.dict("sys.modules", {engine_module: mock_engine}):
            import src.launchers.pinocchio_dashboard  # noqa: F401

            assert not mock_engine.PinocchioPhysicsEngine.called, (
                "PinocchioPhysicsEngine was accessed at module-import time — "
                "the import must be deferred to main()"
            )

        sys.modules.pop("src.launchers.pinocchio_dashboard", None)

    def test_dashboard_module_has_main(self) -> None:
        """pinocchio_dashboard exposes a callable main() function."""
        sys.modules.pop("src.launchers.pinocchio_dashboard", None)
        import src.launchers.pinocchio_dashboard as mod

        assert callable(mod.main)
        sys.modules.pop("src.launchers.pinocchio_dashboard", None)


class TestDrakeDashboardLazyLoading:
    """Verify drake_dashboard does not import engine at module load time."""

    def test_module_importable_without_drake_installed(self) -> None:
        """drake_dashboard can be imported even when the drake engine module is absent."""
        engine_module = "src.engines.physics_engines.drake.python.drake_physics_engine"
        sys.modules.pop("src.launchers.drake_dashboard", None)

        with patch.dict("sys.modules", {engine_module: None}):
            import src.launchers.drake_dashboard  # noqa: F401

        sys.modules.pop("src.launchers.drake_dashboard", None)

    def test_engine_not_imported_at_module_level(self) -> None:
        """Importing drake_dashboard must not touch the Drake physics engine module."""
        engine_module = "src.engines.physics_engines.drake.python.drake_physics_engine"
        sys.modules.pop("src.launchers.drake_dashboard", None)
        sys.modules.pop(engine_module, None)

        mock_engine = MagicMock()
        with patch.dict("sys.modules", {engine_module: mock_engine}):
            import src.launchers.drake_dashboard  # noqa: F401

            assert not mock_engine.DrakePhysicsEngine.called, (
                "DrakePhysicsEngine was accessed at module-import time — "
                "the import must be deferred to main()"
            )

        sys.modules.pop("src.launchers.drake_dashboard", None)

    def test_dashboard_module_has_main(self) -> None:
        """drake_dashboard exposes a callable main() function."""
        sys.modules.pop("src.launchers.drake_dashboard", None)
        import src.launchers.drake_dashboard as mod

        assert callable(mod.main)
        sys.modules.pop("src.launchers.drake_dashboard", None)
