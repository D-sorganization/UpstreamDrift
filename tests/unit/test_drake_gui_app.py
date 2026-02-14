"""Unit tests for Drake GUI App."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.shared.python.engine_core.engine_availability import skip_if_unavailable

# Drake engine module paths that may get imported and must be cleaned up
_DRAKE_ENGINE_MODULES = [
    "src.engines.physics_engines.drake",
    "src.engines.physics_engines.drake.python",
    "src.engines.physics_engines.drake.python.src",
    "src.engines.physics_engines.drake.python.src.drake_gui_app",
]


@pytest.fixture(autouse=True, scope="function")
def _mock_pydrake():
    """Provide mock pydrake modules only during test execution.

    Also cleanup drake engine modules to prevent pollution of test_drake_wrapper.py.
    When drake_gui_app is imported, it brings in the parent package
    src.engines.physics_engines.drake.python into sys.modules. This causes
    test_drake_wrapper.py to fail when it tries to patch
    src.engines.physics_engines.drake.python.drake_physics_engine, because the
    parent package exists but drake_physics_engine was never imported.
    """
    # Save existing drake modules so we can restore them
    saved_modules = {}
    for module_name in _DRAKE_ENGINE_MODULES:
        if module_name in sys.modules:
            saved_modules[module_name] = sys.modules[module_name]

    # Create fresh mocks for each test session to prevent pollution
    pydrake_mocks = {
        "pydrake": MagicMock(),
        "pydrake.all": MagicMock(),
        "pydrake.multibody": MagicMock(),
        "pydrake.multibody.plant": MagicMock(),
        "pydrake.multibody.tree": MagicMock(),
    }
    with patch.dict("sys.modules", pydrake_mocks):
        yield

    # Clean up drake engine modules to prevent pollution
    for module_name in _DRAKE_ENGINE_MODULES:
        if module_name in sys.modules:
            del sys.modules[module_name]

    # Restore saved modules
    for module_name, module in saved_modules.items():
        sys.modules[module_name] = module


@skip_if_unavailable("pyqt6")
class TestDrakeGUIApp:
    def test_compute_specific_control(self):
        """Test compute_specific_control in DrakeInducedAccelerationAnalyzer."""
        # Import inside test to ensure mocks apply
        from src.engines.physics_engines.drake.python.src.drake_gui_app import (
            DrakeInducedAccelerationAnalyzer,
        )

        # Setup mock plant
        plant = MagicMock()
        context = MagicMock()
        analyzer = DrakeInducedAccelerationAnalyzer(plant)

        # Scenario 1: Plant is None (if initialized with None)
        analyzer_none = DrakeInducedAccelerationAnalyzer(None)
        res = analyzer_none.compute_specific_control(context, np.array([1.0]))
        assert len(res) == 0

        # Scenario 2: Valid plant
        # M * a = tau  =>  a = M_inv * tau
        # Let M = identity, tau = [1, 2] => a = [1, 2]
        M = np.eye(2)
        plant.CalcMassMatrix.return_value = M

        # Solve
        tau = np.array([1.0, 2.0])
        res = analyzer.compute_specific_control(context, tau)

        # Check result
        np.testing.assert_array_almost_equal(res, np.array([1.0, 2.0]))

        # Check calls
        plant.CalcMassMatrix.assert_called_with(context)
