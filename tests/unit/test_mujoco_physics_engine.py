"""Tests for MuJoCo physics engine.

Uses shared module-level mocks for the mujoco dependency from conftest.py.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.shared.python.core.constants import GRAVITY_M_S2
from src.shared.python.engine_core.engine_availability import (
    skip_if_unavailable,
)

# Skip entire module if MuJoCo is not available
pytestmark = skip_if_unavailable("mujoco")

# Explicit attribute lists for MuJoCo C++ types that may not be importable
# as Python classes for spec=.
_MJ_MODEL_SPEC = ["nv", "nu", "nq", "nbody", "opt", "body"]
_MJ_DATA_SPEC = [
    "qpos",
    "qvel",
    "qacc",
    "ctrl",
    "qfrc_bias",
    "qfrc_grav",
    "qfrc_inverse",
    "qM",
    "time",
    "xpos",
    "xquat",
]


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


def test_load_from_path(engine, mock_mj, tmp_path):
    # Use tmp_path so the file lives under /tmp, which is in ALLOWED_MODEL_DIRS
    model_file = tmp_path / "model.xml"
    model_file.write_text("<mujoco/>")

    engine.load_from_path(str(model_file))

    # Check if called with SOMETHING ending in "model.xml"
    args, _ = mock_mj.MjModel.from_xml_path.call_args
    assert args[0].endswith("model.xml")
    assert engine.xml_path.endswith("model.xml")


def test_step(engine, mock_mj):
    engine.model = MagicMock(spec=_MJ_MODEL_SPEC)
    engine.data = MagicMock(spec=_MJ_DATA_SPEC)

    engine.step()

    mock_mj.mj_step.assert_called_once_with(engine.model, engine.data)


def test_reset(engine, mock_mj):
    engine.model = MagicMock(spec=_MJ_MODEL_SPEC)
    engine.data = MagicMock(spec=_MJ_DATA_SPEC)

    engine.reset()

    mock_mj.mj_resetData.assert_called_once_with(engine.model, engine.data)
    mock_mj.mj_forward.assert_called_once()  # called by forward()


def test_set_control(engine):
    engine.model = MagicMock(spec=_MJ_MODEL_SPEC)
    engine.model.nu = 2
    engine.data = MagicMock(spec=_MJ_DATA_SPEC)
    engine.data.ctrl = np.zeros(2)

    ctrl = np.array([1.0, 2.0])
    engine.set_control(ctrl)

    np.testing.assert_array_equal(engine.data.ctrl, ctrl)


def test_set_control_mismatch(engine):
    engine.model = MagicMock(spec=_MJ_MODEL_SPEC)
    engine.model.nu = 2
    engine.data = MagicMock(spec=_MJ_DATA_SPEC)

    ctrl = np.array([1.0, 2.0, 3.0])
    # Now expects a ValueError for mismatched control size
    with pytest.raises(ValueError):
        engine.set_control(ctrl)


def test_compute_mass_matrix(engine, mock_mj):
    engine.model = MagicMock(spec=_MJ_MODEL_SPEC)
    engine.model.nv = 2
    engine.data = MagicMock(spec=_MJ_DATA_SPEC)
    engine.data.qM = np.zeros(2)

    M = engine.compute_mass_matrix()
    assert M.shape == (2, 2)
    mock_mj.mj_fullM.assert_called_once()


@pytest.mark.parametrize(
    "method,attr,values",
    [
        ("compute_bias_forces", "qfrc_bias", np.array([1.0, 2.0])),
        ("compute_gravity_forces", "qfrc_grav", np.array([0.0, -GRAVITY_M_S2])),
    ],
    ids=["bias_forces", "gravity_forces"],
)
def test_compute_forces(engine, method, attr, values):
    engine.model = MagicMock(spec=_MJ_MODEL_SPEC)
    engine.data = MagicMock(spec=_MJ_DATA_SPEC)
    setattr(engine.data, attr, values)

    result = getattr(engine, method)()
    np.testing.assert_array_equal(result, values)


def test_compute_inverse_dynamics(engine, mock_mj):
    engine.model = MagicMock(spec=_MJ_MODEL_SPEC)
    engine.model.nv = 2
    engine.data = MagicMock(spec=_MJ_DATA_SPEC)
    engine.data.qfrc_inverse = np.array([10.0, 20.0])
    engine.data.qacc = np.zeros(2)  # Real array for slice assignment

    qacc = np.array([0.1, 0.2])
    tau = engine.compute_inverse_dynamics(qacc)

    assert tau is not None
    np.testing.assert_array_equal(tau, np.array([10.0, 20.0]))
    np.testing.assert_array_equal(engine.data.qacc, qacc)
    mock_mj.mj_inverse.assert_called_once()


def test_compute_affine_drift(engine, mock_mj):
    engine.model = MagicMock(spec=_MJ_MODEL_SPEC)
    engine.data = MagicMock(spec=_MJ_DATA_SPEC)
    engine.data.ctrl = np.array([1.0])
    engine.data.qacc = np.array([0.5])  # Simulated drift acc

    drift = engine.compute_affine_drift()

    assert drift is not None
    np.testing.assert_array_equal(drift, np.array([0.5]))
    # Should have restored control
    np.testing.assert_array_equal(engine.data.ctrl, np.array([1.0]))
    assert mock_mj.mj_forward.call_count == 2  # Once for drift, once for restore


def test_compute_jacobian(engine, mock_mj):
    engine.model = MagicMock(spec=_MJ_MODEL_SPEC)
    engine.model.nv = 2
    engine.data = MagicMock(spec=_MJ_DATA_SPEC)

    mock_mj.mj_name2id.return_value = 1

    jac = engine.compute_jacobian("torso")

    assert jac is not None
    assert "linear" in jac
    assert "angular" in jac
    assert "spatial" in jac
    assert jac["linear"].shape == (3, 2)
    mock_mj.mj_jacBody.assert_called_once()
