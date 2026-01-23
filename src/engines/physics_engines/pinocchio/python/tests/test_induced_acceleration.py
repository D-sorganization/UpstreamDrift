"""Unit tests for Pinocchio Induced Acceleration Analyzer."""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# Mock pinocchio before importing
mock_pin = MagicMock()
sys.modules["pinocchio"] = mock_pin

from src.engines.physics_engines.pinocchio.python.pinocchio_golf.induced_acceleration import (  # noqa: E402, E501
    InducedAccelerationAnalyzer,
)


class TestPinocchioInducedAcceleration:
    """Test suite for InducedAccelerationAnalyzer."""

    @pytest.fixture(autouse=True)
    def reset_mocks(self):
        """Reset mocks before each test."""
        mock_pin.reset_mock()
        mock_pin.aba.side_effect = None
        mock_pin.aba.return_value = MagicMock()  # Default return
        yield

    @pytest.fixture
    def mock_model(self):
        """Mock Pinocchio Model."""
        model = MagicMock()
        model.nq = 2
        model.nv = 2
        model.createData.return_value = MagicMock()
        return model

    @pytest.fixture
    def mock_data(self):
        """Mock Pinocchio Data."""
        return MagicMock()

    @pytest.fixture
    def analyzer(self, mock_model, mock_data):
        """Create analyzer instance."""
        return InducedAccelerationAnalyzer(mock_model, mock_data)

    def test_initialization(self, analyzer, mock_model, mock_data):
        """Test initialization."""
        assert analyzer.model == mock_model
        assert analyzer.data == mock_data
        assert analyzer._temp_data is not None
        assert analyzer._temp_data != mock_data  # Should be a new instance

    def test_compute_components_logic(self, analyzer, mock_model):
        """Test compute_components logical flow."""
        q = np.array([0.0, 0.0])
        v = np.array([0.1, 0.2])
        tau = np.array([1.0, 2.0])

        # We need to control return values of pin.aba to verify the subtraction logic

        def aba_side_effect(model, data, q_arg, v_arg, tau_arg):
            # Check inputs to return corresponding acceleration
            if np.array_equal(v_arg, np.zeros(2)) and np.array_equal(
                tau_arg, np.zeros(2)
            ):
                return np.array([10.0, 10.0])  # q_ddot_g (Gravity only)
            elif np.array_equal(tau_arg, np.zeros(2)):
                return np.array([12.0, 12.0])  # q_ddot_gv (Gravity + Velocity)
            else:
                return np.array(
                    [15.0, 15.0]
                )  # q_ddot_total (Gravity + Velocity + Control)

        mock_pin.aba.side_effect = aba_side_effect

        results = analyzer.compute_components(q, v, tau)

        # Verify logic:
        # gravity = q_ddot_g = [10, 10]
        # velocity = q_ddot_gv - q_ddot_g = [12, 12] - [10, 10] = [2, 2]
        # control = q_ddot_total - q_ddot_gv = [15, 15] - [12, 12] = [3, 3]
        # total = q_ddot_total = [15, 15]

        np.testing.assert_allclose(results["gravity"], [10.0, 10.0])
        np.testing.assert_allclose(results["velocity"], [2.0, 2.0])
        np.testing.assert_allclose(results["control"], [3.0, 3.0])
        np.testing.assert_allclose(results["total"], [15.0, 15.0])

        # Verify calls were made
        assert mock_pin.aba.call_count == 3

    def test_compute_specific_control(self, analyzer, mock_model):
        """Test compute_specific_control."""
        q = np.zeros(2)
        specific_tau = np.array([5.0, 5.0])

        def aba_side_effect(model, data, q_arg, v_arg, tau_arg):
            if np.array_equal(tau_arg, np.zeros(2)):
                return np.array([-9.8, 0])  # Gravity accel
            else:
                return np.array([-4.8, 5.0])  # Gravity + Specific Torque Accel

        mock_pin.aba.side_effect = aba_side_effect

        result = analyzer.compute_specific_control(q, specific_tau)

        # result = (a_tau_G) - (a_G)
        # [-4.8, 5.0] - [-9.8, 0] = [5.0, 5.0]

        np.testing.assert_allclose(result, [5.0, 5.0])
        assert mock_pin.aba.call_count == 2

    def test_compute_counterfactuals(self, analyzer, mock_model):
        """Test compute_counterfactuals."""
        q = np.zeros(2)
        v = np.zeros(2)

        mock_pin.aba.return_value = np.array([1.0, 2.0])
        mock_pin.computeGeneralizedGravity.return_value = np.array([3.0, 4.0])

        results = analyzer.compute_counterfactuals(q, v)

        np.testing.assert_allclose(results["ztcf_accel"], [1.0, 2.0])
        np.testing.assert_allclose(results["zvcf_torque"], [3.0, 4.0])

        mock_pin.aba.assert_called()
        mock_pin.computeGeneralizedGravity.assert_called()
