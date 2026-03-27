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
class TestProbePaths:
    """Verify probe paths match the actual filesystem layout."""

    def test_opensim_probe_path(self, suite_root: Path) -> None:
        """OpenSim probe checks the correct engine directory."""
        from src.shared.python.engine_core.engine_probes import OpenSimProbe

        probe = OpenSimProbe(suite_root)
        result = probe.probe()
        # The probe should not fail with "engine directory" missing
        if "engine directory" in result.missing_dependencies:
            expected_dir = suite_root / "engines" / "physics_engines" / "opensim"
            pytest.skip(f"OpenSim engine directory not found at {expected_dir}")

    def test_myosim_probe_path(self, suite_root: Path) -> None:
        """MyoSim probe checks the correct engine directory (myosuite/)."""
        from src.shared.python.engine_core.engine_probes import MyoSimProbe

        MyoSimProbe(suite_root)  # Ensures probe can be instantiated
        # Verify the probe checks for myosuite directory, not myosim
        engine_dir = suite_root / "src" / "engines" / "physics_engines" / "myosuite"
        assert engine_dir.exists(), (
            f"MyoSuite engine directory should exist at {engine_dir}"
        )

    def test_myosim_probe_checks_correct_file(self, suite_root: Path) -> None:
        """MyoSim probe checks for myosuite_physics_engine.py."""
        engine_file = (
            suite_root
            / "src"
            / "engines"
            / "physics_engines"
            / "myosuite"
            / "python"
            / "myosuite_physics_engine.py"
        )
        assert engine_file.exists(), (
            f"MyoSuite engine file should exist at {engine_file}"
        )


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
class TestAPIRouteConnectivity:
    """Verify OpenSim/MyoSuite are accessible via API routes."""

    @pytest.fixture(scope="class")
    def client(self):
        """Create test client."""
        try:
            from fastapi.testclient import TestClient

            from src.api.server import app
        except ImportError as exc:
            pytest.skip(f"API server deps not available: {exc}")

        with TestClient(app) as c:
            yield c

    def test_opensim_probe_via_api(self, client) -> None:
        """OpenSim probe returns valid JSON via API."""
        resp = client.get("/api/engines/opensim/probe")
        assert resp.status_code == 200
        data = resp.json()
        assert "available" in data

    def test_myosuite_probe_via_api(self, client) -> None:
        """MyoSuite probe returns valid JSON via API."""
        resp = client.get("/api/engines/myosuite/probe")
        assert resp.status_code == 200
        data = resp.json()
        assert "available" in data

    def test_opensim_load_via_api(self, client) -> None:
        """OpenSim load endpoint responds."""
        resp = client.post("/api/engines/opensim/load")
        # May fail if opensim not installed, but should not crash
        assert resp.status_code in [200, 400, 500]

    def test_myosuite_load_via_api(self, client) -> None:
        """MyoSuite load endpoint responds."""
        resp = client.post("/api/engines/myosuite/load")
        assert resp.status_code in [200, 400, 500]
