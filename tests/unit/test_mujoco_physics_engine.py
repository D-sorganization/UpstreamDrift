"""Tests for MuJoCo physics engine.

Uses a single module-level mock for the mujoco dependency, injected into
the physics engine module's namespace.  Per-test patching is removed to
avoid double-mocking conflicts that caused assertion failures in CI.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.shared.python.core.constants import GRAVITY_M_S2
from src.shared.python.engine_core.engine_availability import (
    skip_if_unavailable,
)

# Skip entire module if MuJoCo is not available
pytestmark = skip_if_unavailable("mujoco")


# Mock classes that need to be defined before importing the engine
class MockPhysicsEngine:
    pass


@pytest.fixture(autouse=True, scope="module")
def mock_mujoco_dependencies():
    """Fixture to mock mujoco and interfaces safely for the duration of this module."""
    mock_mujoco = MagicMock()
    mock_interfaces = MagicMock()
    mock_interfaces.PhysicsEngine = MockPhysicsEngine

    with patch.dict(
        "sys.modules",
        {
            "mujoco": mock_mujoco,
            "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.interfaces": mock_interfaces,
        },
    ):
        yield mock_mujoco, mock_interfaces


@pytest.fixture(scope="module")
def MuJoCoPhysicsEngineClass(mock_mujoco_dependencies):
    """Fixture to provide the MuJoCoPhysicsEngine class with mocked dependencies."""
    import src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine as mod

    mock_mujoco, _mock_interfaces = mock_mujoco_dependencies

    # Inject the mock so module-level `mujoco.xxx` calls see it
    mod.mujoco = mock_mujoco

    yield mod.MuJoCoPhysicsEngine

    # No restore needed – the module patch is torn down by the fixture above.


@pytest.fixture
def mock_mj(mock_mujoco_dependencies):
    """Return the shared mujoco mock **and** reset its call tracking.

    This avoids ghostly cross-test state while still using a single mock
    object that the engine module resolves at call-time.
    """
    mock_mujoco, _ = mock_mujoco_dependencies
    mock_mujoco.reset_mock()
    return mock_mujoco


@pytest.fixture
def engine(MuJoCoPhysicsEngineClass):
    """Fixture to provide a MuJoCoPhysicsEngine instance."""
    return MuJoCoPhysicsEngineClass()


def test_initialization(engine):
    assert engine.model is None
    assert engine.data is None


def test_load_from_string(engine, mock_mj):
    xml = "<mujoco/>"
    engine.load_from_string(xml)

    mock_mj.MjModel.from_xml_string.assert_called_once_with(xml)
    assert engine.model is not None
    assert engine.data is not None


def test_load_from_path(engine, mock_mj):
    path = "model.xml"

    # Mock the security validation to allow test paths
    with patch(
        "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine.validate_path"
    ) as mock_validate:
        mock_validate.return_value = Path(path).resolve()

        engine.load_from_path(path)

        # Check if called with SOMETHING ending in "model.xml"
        args, _ = mock_mj.MjModel.from_xml_path.call_args
        assert args[0].endswith("model.xml")
        assert engine.xml_path.endswith(path)


def test_step(engine, mock_mj):
    engine.model = MagicMock()
    engine.data = MagicMock()

    engine.step()

    mock_mj.mj_step.assert_called_once_with(engine.model, engine.data)


def test_reset(engine, mock_mj):
    engine.model = MagicMock()
    engine.data = MagicMock()

    engine.reset()

    mock_mj.mj_resetData.assert_called_once_with(engine.model, engine.data)
    mock_mj.mj_forward.assert_called_once()  # called by forward()


def test_set_control(engine):
    engine.model = MagicMock()
    engine.model.nu = 2
    engine.data = MagicMock()
    engine.data.ctrl = np.zeros(2)

    ctrl = np.array([1.0, 2.0])
    engine.set_control(ctrl)

    np.testing.assert_array_equal(engine.data.ctrl, ctrl)


def test_set_control_mismatch(engine):
    engine.model = MagicMock()
    engine.model.nu = 2
    engine.data = MagicMock()

    ctrl = np.array([1.0, 2.0, 3.0])
    # Now expects a ValueError for mismatched control size
    with pytest.raises(ValueError):
        engine.set_control(ctrl)


def test_compute_mass_matrix(engine, mock_mj):
    engine.model = MagicMock()
    engine.model.nv = 2
    engine.data = MagicMock()
    engine.data.qM = np.zeros(2)

    M = engine.compute_mass_matrix()
    assert M.shape == (2, 2)
    mock_mj.mj_fullM.assert_called_once()


def test_compute_bias_forces(engine):
    engine.model = MagicMock()
    engine.data = MagicMock()
    engine.data.qfrc_bias = np.array([1.0, 2.0])

    bias = engine.compute_bias_forces()
    np.testing.assert_array_equal(bias, np.array([1.0, 2.0]))


def test_compute_gravity_forces(engine):
    engine.model = MagicMock()
    engine.data = MagicMock()
    engine.data.qfrc_grav = np.array([0.0, -GRAVITY_M_S2])

    grav = engine.compute_gravity_forces()
    np.testing.assert_array_equal(grav, np.array([0.0, -GRAVITY_M_S2]))


def test_compute_inverse_dynamics(engine, mock_mj):
    engine.model = MagicMock()
    engine.model.nv = 2
    engine.data = MagicMock()
    engine.data.qfrc_inverse = np.array([10.0, 20.0])
    engine.data.qacc = np.zeros(2)  # Real array for slice assignment

    qacc = np.array([0.1, 0.2])
    tau = engine.compute_inverse_dynamics(qacc)

    assert tau is not None
    np.testing.assert_array_equal(tau, np.array([10.0, 20.0]))
    np.testing.assert_array_equal(engine.data.qacc, qacc)
    mock_mj.mj_inverse.assert_called_once()


def test_compute_affine_drift(engine, mock_mj):
    engine.model = MagicMock()
    engine.data = MagicMock()
    engine.data.ctrl = np.array([1.0])
    engine.data.qacc = np.array([0.5])  # Simulated drift acc

    drift = engine.compute_affine_drift()

    assert drift is not None
    np.testing.assert_array_equal(drift, np.array([0.5]))
    # Should have restored control
    np.testing.assert_array_equal(engine.data.ctrl, np.array([1.0]))
    assert mock_mj.mj_forward.call_count == 2  # Once for drift, once for restore


def test_compute_jacobian(engine, mock_mj):
    engine.model = MagicMock()
    engine.model.nv = 2
    engine.data = MagicMock()

    mock_mj.mj_name2id.return_value = 1

    jac = engine.compute_jacobian("torso")

    assert jac is not None
    assert "linear" in jac
    assert "angular" in jac
    assert "spatial" in jac
    assert jac["linear"].shape == (3, 2)
    mock_mj.mj_jacBody.assert_called_once()
