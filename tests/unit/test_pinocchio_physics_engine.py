from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from src.shared.python.engine_core.engine_availability import (
    skip_if_unavailable,
)

# Skip entire module if Pinocchio is not installed - mocking pinocchio at module level
# is unreliable and leads to AttributeError on patched module globals
pytestmark = skip_if_unavailable("pinocchio")

# Explicit attribute lists for Pinocchio C++ types (pinocchio bindings).
_PIN_MODEL_SPEC = [
    "nv",
    "nq",
    "existBodyName",
    "existFrame",
    "getFrameId",
    "createData",
]
_PIN_DATA_SPEC = ["M", "J"]


# Mock classes that need to be defined before importing the engine
class MockPhysicsEngine:
    pass


@pytest.fixture(autouse=True, scope="module")
def mock_pinocchio_dependencies():
    """Fixture to mock pinocchio and interfaces safely for the duration of this module."""
    mock_pin = MagicMock()
    mock_interfaces = MagicMock()
    mock_interfaces.PhysicsEngine = MockPhysicsEngine

    with patch.dict(
        "sys.modules",
        {"pinocchio": mock_pin, "shared.python.interfaces": mock_interfaces},
    ):
        yield mock_pin, mock_interfaces


@pytest.fixture(scope="module")
def PinocchioPhysicsEngineClass(mock_pinocchio_dependencies: Any) -> Any:
    """Fixture to provide the PinocchioPhysicsEngine class with mocked dependencies."""
    # Ensure module is imported via the correct src-rooted path
    import src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine as mod

    # Manually patch the module's globals
    mock_pin, mock_interfaces = mock_pinocchio_dependencies

    # Save originals
    original_pin = getattr(mod, "pin", None)

    # Inject mocks
    mod.pin = mock_pin  # type: ignore[attr-defined]

    yield mod.PinocchioPhysicsEngine

    # Restore
    if original_pin:
        mod.pin = original_pin  # type: ignore[attr-defined]


@pytest.fixture
def engine(PinocchioPhysicsEngineClass):
    """Fixture to provide a PinocchioPhysicsEngine instance."""
    return PinocchioPhysicsEngineClass()


def test_pinocchio_physics_engine_initialization(engine: Any) -> None:
    assert engine.model is None
    assert engine.data is None
    assert engine.time == 0.0


@patch("src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine.pin")
@patch(
    "src.shared.python.engine_core.base_physics_engine.BasePhysicsEngine.load_from_path",
    autospec=True,
)
def test_pinocchio_physics_engine_load_from_path(
    mock_load: Any, mock_pin: Any, engine: Any
) -> None:
    """load_from_path delegates to _load_from_path_impl with mocked pinocchio.

    We bypass the BasePhysicsEngine file-validation layer (tested separately)
    and call _load_from_path_impl directly to verify the pinocchio-specific
    logic (buildModelFromUrdf, neutral, createData calls).
    """
    # Arrange: configure mock model/data return values
    mock_model = MagicMock(spec=_PIN_MODEL_SPEC)
    mock_model.nq = 1
    mock_model.nv = 1
    mock_pin.buildModelFromUrdf.return_value = mock_model
    mock_pin.neutral.return_value = np.array([0.0])

    # Act: call _load_from_path_impl directly to bypass BasePhysicsEngine path check
    engine._load_from_path_impl("test.urdf")

    mock_pin.buildModelFromUrdf.assert_called_once_with("test.urdf")
    mock_pin.neutral.assert_called_once()
    assert engine.model is not None
    assert engine.data is not None


@patch("src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine.pin")
def test_pinocchio_physics_engine_load_from_string(mock_pin: Any, engine: Any) -> None:
    content = "<robot/>"
    mock_model = MagicMock(spec=_PIN_MODEL_SPEC)
    mock_model.nv = 2
    mock_model.nq = 2
    mock_pin.buildModelFromXML.return_value = mock_model
    mock_pin.neutral.return_value = np.zeros(2)

    engine.load_from_string(content, "urdf")

    mock_pin.buildModelFromXML.assert_called_once_with(content)
    assert engine.model is not None


def test_pinocchio_physics_engine_step(engine: Any) -> None:
    engine.model = MagicMock(spec=_PIN_MODEL_SPEC)
    engine.data = MagicMock(spec=_PIN_DATA_SPEC)
    engine.q = np.array([0.0])
    engine.v = np.array([0.0])
    engine.tau = np.array([0.0])

    # Mock aba return
    with patch(
        "src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine.pin"
    ) as mock_pin:
        mock_pin.aba.return_value = np.array([1.0])  # acceleration
        mock_pin.integrate.return_value = np.array([0.1])

        engine.step(0.1, integrator="semi_implicit")

        mock_pin.aba.assert_called_once()
        assert mock_pin.integrate.call_count == 1
        np.testing.assert_array_equal(engine.a, np.array([1.0]))
        # v = v + a*dt = 0 + 1.0*0.1 = 0.1
        np.testing.assert_array_equal(engine.v, np.array([0.1]))


def test_pinocchio_physics_engine_compute_mass_matrix(engine: Any) -> None:
    engine.model = MagicMock(spec=_PIN_MODEL_SPEC)
    engine.data = MagicMock(spec=_PIN_DATA_SPEC)
    # Mock data.M
    engine.data.M = np.array([[1.0, 0.2], [0.0, 2.0]])  # Upper triangular example

    with patch(
        "src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine.pin"
    ) as mock_pin:
        M = engine.compute_mass_matrix()

        mock_pin.crba.assert_called_once()
        # Should be symmetrized
        expected = np.array([[1.0, 0.2], [0.2, 2.0]])
        np.testing.assert_array_almost_equal(M, expected)


def test_pinocchio_physics_engine_compute_jacobian(engine: Any) -> None:
    engine.model = MagicMock(spec=_PIN_MODEL_SPEC)
    engine.data = MagicMock(spec=_PIN_DATA_SPEC)
    engine.model.existBodyName.return_value = True
    engine.model.getFrameId.return_value = 1

    with patch(
        "src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine.pin"
    ) as mock_pin:
        # 6x2 Jacobian
        mock_J = np.zeros((6, 2))
        mock_J[0, 0] = 1.0  # Linear x
        mock_J[5, 1] = 1.0  # Angular z
        mock_pin.getFrameJacobian.return_value = mock_J

        J = engine.compute_jacobian("body")

        assert J is not None
        assert "linear" in J
        assert "angular" in J
        np.testing.assert_array_equal(J["linear"], mock_J[:3, :])
        np.testing.assert_array_equal(J["angular"], mock_J[3:, :])
