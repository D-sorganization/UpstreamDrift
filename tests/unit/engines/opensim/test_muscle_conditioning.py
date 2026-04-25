import logging
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.unit

# Skip if opensim is not installed, but test the logic by mocking
try:
    import opensim

    OPENSIM_AVAILABLE = True
except ImportError:
    OPENSIM_AVAILABLE = False


# Define a lightweight mock class that mirrors the opensim API surface used
# by this test module.  This class is referenced both at collection time (to
# stub sys.modules so the src import below succeeds) and at test time via the
# autouse ``_opensim_stub`` fixture.
class _MockOpenSim:
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


# We must register a stub in sys.modules *before* importing muscle_analysis so
# that the module-level ``import opensim`` inside it succeeds at collection time.
# The stub is removed immediately after the import; the autouse fixture below
# re-installs it per-test via patch.dict — satisfying the CLAUDE.md rule against
# persistent module-level sys.modules mutations.
if not OPENSIM_AVAILABLE:
    _prior_opensim = sys.modules.get("opensim")
    sys.modules["opensim"] = _MockOpenSim  # type: ignore[assignment]

from src.engines.physics_engines.opensim.python.muscle_analysis import (  # noqa: E402
    OpenSimMuscleAnalyzer,
)
import pytest
pytestmark = pytest.mark.unit


if not OPENSIM_AVAILABLE:
    # Remove the collection-time stub right away.
    if _prior_opensim is None:
        sys.modules.pop("opensim", None)
    else:
        sys.modules["opensim"] = _prior_opensim
    del _prior_opensim

# Convenient alias: used in tests that run when opensim is not installed.
MockOpenSim = _MockOpenSim


@pytest.fixture(autouse=True)
def _opensim_stub():
    """Ensure a stub for ``opensim`` is in sys.modules for every test.

    Uses ``patch.dict`` so the stub is automatically removed after the test,
    satisfying the CLAUDE.md rule against persistent module-level sys.modules
    mutations.  When the real opensim is installed the dict is patched with
    the real module object, which is an identity replacement.
    """
    stub = opensim if OPENSIM_AVAILABLE else _MockOpenSim  # type: ignore[possibly-undefined]
    with patch.dict(sys.modules, {"opensim": stub}):  # type: ignore[arg-type]
        yield


def test_mass_matrix_conditioning_fallback(caplog):
    """Test that a near-singular mass matrix triggers the Tikhonov regularization fallback."""
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
    analyzer.compute_muscle_joint_torques = MagicMock(
        return_value={"muscle1": np.array([1.0, 0.0])}
    )

    with (
        caplog.at_level(logging.WARNING),
        patch(
            "src.engines.physics_engines.opensim.python.muscle_analysis.opensim"
        ) as mock_opensim_mod,
    ):
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
