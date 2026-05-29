import logging
from unittest.mock import MagicMock, patch

import numpy as np

# Skip if opensim is not installed, but test the logic by mocking
try:
    import opensim

    OPENSIM_AVAILABLE = True
except ImportError:
    OPENSIM_AVAILABLE = False


# Mock opensim for the test if it's not available
if not OPENSIM_AVAILABLE:

    class MockOpenSim:
        class Model:
            pass

        class State:
            pass

        class Controller:
            pass

        class Matrix:
            def __init__(self):
                self.data = np.zeros((2, 2))

            def get(self, r, c):
                return self.data[r, c]

            def set(self, r, c, val):
                self.data[r, c] = val

    import sys

    sys.modules["opensim"] = MockOpenSim  # type: ignore[assignment]
    opensim = MockOpenSim  # type: ignore[assignment]

from src.engines.physics_engines.opensim.python.muscle_analysis import (
    OpenSimMuscleAnalyzer,
)


def test_mass_matrix_conditioning_fallback(caplog):
    """Test that a near-singular mass matrix triggers the Tikhonov regularization fallback."""
    if not OPENSIM_AVAILABLE:
        # Simple mock objects just to instantiate the analyzer
        # No spec= here: MockOpenSim.Model/State are empty stubs without getMuscles etc.
        mock_model = MagicMock()
        mock_state = MagicMock()
        mock_muscle_set = MagicMock()
        mock_model.getMuscles.return_value = mock_muscle_set
        mock_muscle_set.getSize.return_value = 0
    else:
        mock_model = MagicMock()
        mock_state = MagicMock()
        mock_muscle_set = MagicMock()
        mock_model.getMuscles.return_value = mock_muscle_set
        mock_muscle_set.getSize.return_value = 0

    analyzer = OpenSimMuscleAnalyzer(mock_model, mock_state)

    # Setup the internal mocks for compute_muscle_induced_accelerations
    mock_model.getNumSpeeds.return_value = 2
    mock_matter = MagicMock()
    mock_model.getMatterSubsystem.return_value = mock_matter

    M_ill = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-12]])

    def mock_calcM(state, m_mat):
        # We replace the get method to return our matrix
        def mock_get(r, c):
            return M_ill[r, c]

        m_mat.get = mock_get

    mock_matter.calcM.side_effect = mock_calcM

    # Provide fake torques
    analyzer.compute_muscle_joint_torques = MagicMock(  # type: ignore[method-assign]
        return_value={"muscle1": np.array([1.0, 0.0])}
    )

    with caplog.at_level(logging.WARNING):
        # We must make sure opensim is accessible inside the module
        import src.engines.physics_engines.opensim.python.muscle_analysis as mod

        mod.opensim = MagicMock()
        mod.opensim.Matrix = MagicMock

        with patch(
            "src.engines.physics_engines.opensim.python.muscle_analysis.opensim"
        ) as mock_opensim_mod:
            # We mock opensim.Matrix directly
            class FakeOSIM_Matrix:
                def get(self, r, c):
                    return M_ill[r, c]

            mock_opensim_mod.Matrix.return_value = FakeOSIM_Matrix()

            accels = analyzer.compute_muscle_induced_accelerations()

    # Verify fallback was triggered
    assert any(
        "Mass matrix ill-conditioned" in record.message for record in caplog.records
    )

    # Verify result
    assert "muscle1" in accels
    assert np.all(np.isfinite(accels["muscle1"]))
