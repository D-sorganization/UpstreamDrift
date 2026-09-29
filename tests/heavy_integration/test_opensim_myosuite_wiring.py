"""
Integration tests for OpenSim and MyoSuite engine wiring.

Verifies that the entire pipeline — from probe → loader → engine instance — is
correctly connected for both engines.  Tests that require the actual engine
packages are automatically skipped when the packages are not installed.

Fixes #1115, #1116
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from src.shared.python.engine_core.engine_registry import EngineType

if TYPE_CHECKING:
    pass


@pytest.fixture(scope="module")
def suite_root() -> Path:
    """Return the suite root directory."""
    return Path(__file__).parent.parent.parent


# ──────────────────────────────────────────────────────────────
#  Probe Path Consistency
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  Loader → Engine Factory Consistency
# ──────────────────────────────────────────────────────────────
class TestLoaderWiring:
    """Verify loaders are correctly mapped to engine types."""

    def test_opensim_in_loader_map(self) -> None:
        """OpenSim has a loader in LOADER_MAP."""
        from src.shared.python.engine_core.engine_loaders import LOADER_MAP

        assert EngineType.OPENSIM in LOADER_MAP

    def test_myosim_in_loader_map(self) -> None:
        """MyoSim has a loader in LOADER_MAP."""
        from src.shared.python.engine_core.engine_loaders import LOADER_MAP

        assert EngineType.MYOSIM in LOADER_MAP

    def test_opensim_loader_imports_correct_class(self) -> None:
        """OpenSim loader references OpenSimPhysicsEngine."""
        import inspect

        from src.shared.python.engine_core.engine_loaders import load_opensim_engine

        source = inspect.getsource(load_opensim_engine)
        assert "OpenSimPhysicsEngine" in source

    def test_myosim_loader_imports_correct_class(self) -> None:
        """MyoSim loader references MyoSuitePhysicsEngine."""
        import inspect

        from src.shared.python.engine_core.engine_loaders import load_myosim_engine

        source = inspect.getsource(load_myosim_engine)
        assert "MyoSuitePhysicsEngine" in source


# ──────────────────────────────────────────────────────────────
#  Engine Availability Module
# ──────────────────────────────────────────────────────────────
class TestEngineAvailability:
    """Verify engine availability detection layer."""

    def test_opensim_availability_flag_exists(self) -> None:
        """OPENSIM_AVAILABLE flag is defined."""
        from src.shared.python.engine_core.engine_availability import OPENSIM_AVAILABLE

        assert isinstance(OPENSIM_AVAILABLE, bool)

    def test_myosuite_availability_flag_exists(self) -> None:
        """MYOSUITE_AVAILABLE flag is defined."""
        from src.shared.python.engine_core.engine_availability import MYOSUITE_AVAILABLE

        assert isinstance(MYOSUITE_AVAILABLE, bool)


# ──────────────────────────────────────────────────────────────
#  OpenSim PhysicsEngine Protocol Compliance
# ──────────────────────────────────────────────────────────────
class TestOpenSimProtocol:
    """Verify OpenSimPhysicsEngine satisfies the PhysicsEngine protocol."""

    def test_opensim_has_required_methods(self) -> None:
        """OpenSimPhysicsEngine has all required protocol methods."""
        from src.engines.physics_engines.opensim.python.opensim_physics_engine import (
            OpenSimPhysicsEngine,
        )

        required_methods = [
            "load_from_path",
            "load_from_string",
            "reset",
            "step",
            "forward",
            "get_state",
            "set_state",
            "set_control",
            "get_time",
            "compute_mass_matrix",
            "compute_bias_forces",
            "compute_gravity_forces",
            "compute_inverse_dynamics",
            "compute_jacobian",
            "compute_drift_acceleration",
            "compute_control_acceleration",
        ]

        for method in required_methods:
            assert hasattr(OpenSimPhysicsEngine, method), (
                f"OpenSimPhysicsEngine missing required method: {method}"
            )
            assert callable(getattr(OpenSimPhysicsEngine, method))

    def test_opensim_has_biomech_methods(self) -> None:
        """OpenSimPhysicsEngine has golf-specific biomechanics methods."""
        from src.engines.physics_engines.opensim.python.opensim_physics_engine import (
            OpenSimPhysicsEngine,
        )

        biomech_methods = [
            "get_muscle_analyzer",
            "create_grip_model",
        ]
        for method in biomech_methods:
            assert hasattr(OpenSimPhysicsEngine, method), (
                f"OpenSimPhysicsEngine missing biomech method: {method}"
            )

    def test_opensim_uninitialized_state(self) -> None:
        """Uninitialized OpenSimPhysicsEngine reports not initialized."""
        from src.engines.physics_engines.opensim.python.opensim_physics_engine import (
            OpenSimPhysicsEngine,
        )

        engine = OpenSimPhysicsEngine()
        assert engine.is_initialized is False  # noqa: E712
        # When uninitialized, model_name may return a default marker string
        assert isinstance(engine.model_name, str)


# ──────────────────────────────────────────────────────────────
#  MyoSuite PhysicsEngine Protocol Compliance
# ──────────────────────────────────────────────────────────────
class TestMyoSuiteProtocol:
    """Verify MyoSuitePhysicsEngine satisfies the PhysicsEngine protocol."""

    def test_myosuite_has_required_methods(self) -> None:
        """MyoSuitePhysicsEngine has all required protocol methods."""
        from src.engines.physics_engines.myosuite.python.myosuite_physics_engine import (
            MyoSuitePhysicsEngine,
        )

        required_methods = [
            "load_from_path",
            "load_from_string",
            "reset",
            "step",
            "forward",
            "get_state",
            "set_state",
            "set_control",
            "get_time",
            "compute_mass_matrix",
            "compute_bias_forces",
            "compute_gravity_forces",
            "compute_inverse_dynamics",
            "compute_jacobian",
            "compute_drift_acceleration",
            "compute_control_acceleration",
        ]

        for method in required_methods:
            assert hasattr(MyoSuitePhysicsEngine, method), (
                f"MyoSuitePhysicsEngine missing required method: {method}"
            )
            assert callable(getattr(MyoSuitePhysicsEngine, method))

    def test_myosuite_has_muscle_methods(self) -> None:
        """MyoSuitePhysicsEngine has muscle control methods."""
        from src.engines.physics_engines.myosuite.python.myosuite_physics_engine import (
            MyoSuitePhysicsEngine,
        )

        muscle_methods = [
            "set_muscle_activations",
            "get_muscle_analyzer",
            "create_grip_model",
            "compute_muscle_induced_accelerations",
            "get_muscle_names",
        ]
        for method in muscle_methods:
            assert hasattr(MyoSuitePhysicsEngine, method), (
                f"MyoSuitePhysicsEngine missing muscle method: {method}"
            )

    def test_myosuite_uninitialized_state(self) -> None:
        """Uninitialized MyoSuitePhysicsEngine reports not initialized."""
        from src.engines.physics_engines.myosuite.python.myosuite_physics_engine import (
            MyoSuitePhysicsEngine,
        )

        engine = MyoSuitePhysicsEngine()
        assert engine.is_initialized is False  # noqa: E712
        # When uninitialized, model_name may return a default marker string
        assert isinstance(engine.model_name, str)


# ──────────────────────────────────────────────────────────────
#  MyoSuite Adapter Integration
# ──────────────────────────────────────────────────────────────
class TestMyoSuiteAdapter:
    """Verify the MyoSuite adapter layer is functional."""

    def test_muscle_driven_env_class_exists(self) -> None:
        """MuscleDrivenEnv is importable from the adapter module."""
        from src.shared.python.biomechanics.myosuite_adapter import MuscleDrivenEnv

        assert MuscleDrivenEnv is not None

    def test_train_policy_function_exists(self) -> None:
        """train_muscle_policy function is importable."""
        from src.shared.python.biomechanics.myosuite_adapter import train_muscle_policy

        assert callable(train_muscle_policy)

    def test_muscle_driven_env_init_with_mock(self) -> None:
        """MuscleDrivenEnv initializes with a mock muscle system."""
        from src.shared.python.biomechanics.myosuite_adapter import MuscleDrivenEnv

        mock_muscle = MagicMock()
        mock_muscle.muscles = {"biceps": MagicMock(), "triceps": MagicMock()}

        with patch.object(
            MuscleDrivenEnv, "_get_muscle_names", return_value=["biceps", "triceps"]
        ):
            env = MuscleDrivenEnv(muscle_system=mock_muscle)
            assert env is not None


# ──────────────────────────────────────────────────────────────
#  API Route Connectivity
# ──────────────────────────────────────────────────────────────


pytestmark = pytest.mark.live_simulation
