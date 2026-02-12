"""Unit tests for Drake GUI App."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.shared.python.engine_core.engine_availability import skip_if_unavailable

_PYDRAKE_MOCKS = {
    "pydrake": MagicMock(),
    "pydrake.all": MagicMock(),
    "pydrake.multibody": MagicMock(),
    "pydrake.multibody.plant": MagicMock(),
    "pydrake.multibody.tree": MagicMock(),
}


@pytest.fixture(autouse=True)
def _mock_pydrake():
    """Provide mock pydrake modules only during test execution."""
    with patch.dict("sys.modules", _PYDRAKE_MOCKS):
        yield


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
