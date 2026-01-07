"""Unit tests for Drake Induced Acceleration Analyzer."""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# Mock pydrake before importing the module under test
mock_pydrake = MagicMock()
sys.modules["pydrake"] = mock_pydrake
sys.modules["pydrake.all"] = mock_pydrake

# Now import the module under test
# We use 'src' as a package anchor if possible, or relative import if path is set
from engines.physics_engines.drake.python.src.induced_acceleration import (  # noqa: E402
    DrakeInducedAccelerationAnalyzer,
)


class TestDrakeInducedAcceleration:
    """Test suite for DrakeInducedAccelerationAnalyzer."""

    @pytest.fixture
    def mock_plant(self):
        """Mock MultibodyPlant."""
        plant = MagicMock()
        # Setup basic mock behavior
        plant.num_velocities.return_value = 2
        return plant

    @pytest.fixture
    def analyzer(self, mock_plant):
        """Create analyzer instance."""
        return DrakeInducedAccelerationAnalyzer(mock_plant)

    def test_initialization(self, analyzer, mock_plant):
        """Test initialization."""
        assert analyzer.plant == mock_plant

    def test_compute_components_basic(self, analyzer, mock_plant):
        """Test compute_components with simple inputs."""
        context = MagicMock()

        # Setup mock return values for plant methods
        # 1. Mass Matrix (M) = Identity
        M = np.eye(2)
        mock_plant.CalcMassMatrix.return_value = M

        # 2. Gravity Forces (tau_g)
        # Assume gravity force is [10, 0] (e.g. gravity acting on first joint)
        tau_g = np.array([10.0, 0.0])
        mock_plant.CalcGravityGeneralizedForces.return_value = tau_g

        # 3. Bias Term (bias = C*v - tau_g)
        # Let's say C*v = [2, 2]. Then bias = [2, 2] - [10, 0] = [-8, 2]
        bias = np.array([-8.0, 2.0])
        mock_plant.CalcBiasTerm.return_value = bias

        # Call compute
        results = analyzer.compute_components(context)

        # Verify results
        # Logic:
        # acc_g = M^-1 * tau_g = I * [10, 0] = [10, 0]
        # acc_c = M^-1 * -(bias + tau_g)
        #       = -( [-8, 2] + [10, 0] ) = -[2, 2] = [-2, -2]
        # acc_t = 0 (default)
        # total = acc_g + acc_c + acc_t = [10-2, 0-2] = [8, -2]

        np.testing.assert_allclose(results["gravity"], [10.0, 0.0])
        np.testing.assert_allclose(results["velocity"], [-2.0, -2.0])
        np.testing.assert_allclose(results["control"], [0.0, 0.0])
        np.testing.assert_allclose(results["total"], [8.0, -2.0])

        # Verify plant calls
        mock_plant.CalcMassMatrix.assert_called_with(context)
        mock_plant.CalcGravityGeneralizedForces.assert_called_with(context)
        mock_plant.CalcBiasTerm.assert_called_with(context)

    def test_compute_components_with_non_identity_mass(self, analyzer, mock_plant):
        """Test compute_components with non-identity mass matrix."""
        context = MagicMock()

        # M = [[2, 0], [0, 0.5]]
        M = np.array([[2.0, 0.0], [0.0, 0.5]])
        mock_plant.CalcMassMatrix.return_value = M

        # tau_g = [4, 1]
        tau_g = np.array([4.0, 1.0])
        mock_plant.CalcGravityGeneralizedForces.return_value = tau_g

        # bias = [0, 0] => Cv = tau_g => Coriolis/Centrifugal exactly opposes gravity?
        # bias = Cv - tau_g => if bias=0, Cv = tau_g.
        bias = np.zeros(2)
        mock_plant.CalcBiasTerm.return_value = bias

        results = analyzer.compute_components(context)

        # acc_g = M^-1 * tau_g = [[0.5, 0], [0, 2]] * [4, 1] = [2, 2]
        np.testing.assert_allclose(results["gravity"], [2.0, 2.0])

        # acc_c = M^-1 * -(bias + tau_g) = M^-1 * -[4, 1] = -[2, 2]
        np.testing.assert_allclose(results["velocity"], [-2.0, -2.0])

        # total = 0
        np.testing.assert_allclose(results["total"], [0.0, 0.0], atol=1e-10)

    def test_compute_components_structure(self, analyzer, mock_plant):
        """Verify the result dictionary structure."""
        mock_plant.CalcMassMatrix.return_value = np.eye(2)
        mock_plant.CalcGravityGeneralizedForces.return_value = np.zeros(2)
        mock_plant.CalcBiasTerm.return_value = np.zeros(2)

        results = analyzer.compute_components(MagicMock())

        assert isinstance(results, dict)
        assert "gravity" in results
        assert "velocity" in results
        assert "control" in results
        assert "total" in results
