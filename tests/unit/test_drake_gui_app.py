"""Unit tests for Drake GUI App."""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# Mock modules before importing
sys.modules["pydrake"] = MagicMock()
sys.modules["pydrake.all"] = MagicMock()
sys.modules["pydrake.multibody"] = MagicMock()
sys.modules["pydrake.multibody.plant"] = MagicMock()
sys.modules["pydrake.multibody.tree"] = MagicMock()

# Check for PyQt6 GUI library availability (not just module presence)
try:
    from PyQt6 import QtWidgets  # noqa: F401

    HAS_QT = True
except (ImportError, OSError):
    HAS_QT = False


def teardown_module(module):
    """Clean up sys.modules pollution."""
    for key in list(sys.modules.keys()):
        if key.startswith("pydrake"):
            del sys.modules[key]


@pytest.mark.skipif(not HAS_QT, reason="PyQt6 not available")
class TestDrakeGUIApp:

    def test_compute_specific_control(self):
        """Test compute_specific_control in DrakeInducedAccelerationAnalyzer."""
        # Import inside test to ensure mocks apply
        from engines.physics_engines.drake.python.src.drake_gui_app import (
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
