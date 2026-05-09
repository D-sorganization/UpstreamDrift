import sys
import unittest
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
from src.shared.python.engine_core.engine_availability import DRAKE_AVAILABLE

pytest.importorskip("pydrake.geometry", reason="pydrake not available")
from pydrake.geometry import SceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import Context, Diagram

# MultibodyPlant uses undocumented Drake APIs; enumerate tested attributes.
_PLANT_SPEC_ATTRS = [
    "Finalize",
    "time_step",
    "GetMyContextFromRoot",
    "SetDefaultPositions",
    "SetDefaultVelocities",
    "GetPositions",
    "GetVelocities",
    "CalcMassMatrixViaInverseDynamics",
    "CalcInverseDynamics",
    "CalcGravityGeneralizedForces",
    "num_velocities",
    "MakeMultibodyForces",
]

# Mock pydrake using patch.dict (auto-cleans) to allow DrakePhysicsEngine import.
_PYDRAKE_KEYS = [
    "pydrake",
    "pydrake.systems",
    "pydrake.systems.analysis",
    "pydrake.systems.framework",
    "pydrake.multibody",
    "pydrake.multibody.parsing",
    "pydrake.multibody.plant",
    "pydrake.geometry",
    "pydrake.math",
    "pydrake.all",
]

_ENGINE_MOD_NAME = "src.engines.physics_engines.drake.python.drake_physics_engine"

# Drake engine parent packages that may be polluted by other tests
_DRAKE_PARENT_PACKAGES = [
    "src.engines",
    "src.engines.physics_engines",
    "src.engines.physics_engines.drake",
    "src.engines.physics_engines.drake.python",
    "src.engines.physics_engines.drake.python.src",
]


@pytest.fixture(autouse=True)
def _fix_drake_pollution() -> Generator[None, None, None]:
    """Fix Drake parent package pollution before each test.

    When test_drake_gui_app or other tests import Drake modules, they may leave
    the parent package src.engines.physics_engines.drake.python in sys.modules
    without the drake_physics_engine submodule. This causes @patch decorators
    to fail when trying to patch src.engines.physics_engines.drake.python.drake_physics_engine.

    Instead of removing the parent package (which conftest might restore), we
    ensure drake_physics_engine is properly registered as an attribute AND in sys.modules.
    """
    if _drake_engine_module is not None:
        # Ensure the engine module is in sys.modules
        sys.modules[_ENGINE_MOD_NAME] = _drake_engine_module

        # Also register it as an attribute of the parent package if it exists
        parent_pkg_name = "src.engines.physics_engines.drake.python"
        if parent_pkg_name in sys.modules:
            parent_pkg = sys.modules[parent_pkg_name]
            parent_pkg.drake_physics_engine = _drake_engine_module

    yield

    # Cleanup
    sys.modules.pop(_ENGINE_MOD_NAME, None)


_pydrake_mocks = {k: MagicMock() for k in _PYDRAKE_KEYS}
_drake_engine_module = None  # will hold module reference for @patch usage

# Clean up any polluted Drake parent packages from previous tests
for pkg in _DRAKE_PARENT_PACKAGES:
    sys.modules.pop(pkg, None)
sys.modules.pop(_ENGINE_MOD_NAME, None)

with patch.dict(sys.modules, _pydrake_mocks):
    with patch(
        "src.shared.python.engine_core.engine_availability.DRAKE_AVAILABLE", True
    ):
        try:
            from src.engines.physics_engines.drake.python import (
                drake_physics_engine as _drake_engine_module,
            )

            DrakePhysicsEngine = _drake_engine_module.DrakePhysicsEngine
        except ImportError:
            DrakePhysicsEngine = None  # type: ignore[assignment,misc]

    # Remove the mock-backed engine module to prevent pollution during
    # integration tests that run before this file's tests.
    sys.modules.pop(_ENGINE_MOD_NAME, None)
    for pkg in _DRAKE_PARENT_PACKAGES:
        sys.modules.pop(pkg, None)


@pytest.fixture(autouse=True)
def _isolate_drake_module_state() -> Generator[None, None, None]:
    """Prevent module-level Drake mocks from leaking across tests."""
    engine_before = sys.modules.get(_ENGINE_MOD_NAME)
    pydrake_before = {key: sys.modules.get(key) for key in _PYDRAKE_KEYS}
    yield

    if engine_before is None:
        sys.modules.pop(_ENGINE_MOD_NAME, None)
    else:
        sys.modules[_ENGINE_MOD_NAME] = engine_before

    for key, value in pydrake_before.items():
        if value is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = value


if __name__ == "__main__":
    unittest.main()
