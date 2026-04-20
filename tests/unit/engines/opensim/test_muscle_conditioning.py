import logging
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class _MockOpenSim:
    """Minimal opensim stub — used only when the real library is absent."""

    class Model:
        pass

    class State:
        pass

    class Matrix:
        def __init__(self):
            self.data = np.zeros((2, 2))

        def get(self, r, c):
            return self.data[r, c]

        def set(self, r, c, val):
            self.data[r, c] = val


def _make_opensim_patch() -> dict:
    """Return sys.modules entries needed to import muscle_analysis without opensim."""
    return {"opensim": _MockOpenSim}  # type: ignore[dict-item]


@pytest.fixture()
def muscle_analyzer_class():
    """Import OpenSimMuscleAnalyzer inside a patch so the real opensim isn't required."""
    with patch.dict(sys.modules, _make_opensim_patch()):
        from src.engines.physics_engines.opensim.python.muscle_analysis import (
            OpenSimMuscleAnalyzer,
        )

        yield OpenSimMuscleAnalyzer


def test_mass_matrix_conditioning_fallback(muscle_analyzer_class, caplog):
    """Test that a near-singular mass matrix triggers the Tikhonov regularization fallback."""
    mock_model = MagicMock()
    mock_state = MagicMock()
    mock_muscle_set = MagicMock()
    mock_model.getMuscles.return_value = mock_muscle_set
    mock_muscle_set.getSize.return_value = 0

    with patch.dict(sys.modules, _make_opensim_patch()):
        analyzer = muscle_analyzer_class(mock_model, mock_state)

    mock_model.getNumSpeeds.return_value = 2
    mock_matter = MagicMock()
    mock_model.getMatterSubsystem.return_value = mock_matter

    M_ill = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-12]])

    def mock_calcM(state, m_mat):
        def mock_get(r, c):
            return M_ill[r, c]

        m_mat.get = mock_get

    mock_matter.calcM.side_effect = mock_calcM

    analyzer.compute_muscle_joint_torques = MagicMock(
        return_value={"muscle1": np.array([1.0, 0.0])}
    )

    with (
        caplog.at_level(logging.WARNING),
        patch(
            "src.engines.physics_engines.opensim.python.muscle_analysis.opensim"
        ) as mock_opensim_mod,
    ):

        class FakeOSIM_Matrix:
            def get(self, r, c):
                return M_ill[r, c]

        mock_opensim_mod.Matrix.return_value = FakeOSIM_Matrix()

        accels = analyzer.compute_muscle_induced_accelerations()

    assert any(
        "Mass matrix ill-conditioned" in record.message for record in caplog.records
    )
    assert "muscle1" in accels
    assert np.all(np.isfinite(accels["muscle1"]))
