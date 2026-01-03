"""
Real integration tests for physics engine loading and operation.

These tests demonstrate proper integration testing:
- Use real dependencies where available (skip if not installed)
- Test actual integration between components
- Verify real behavior, not mocked interactions
- Test end-to-end workflows
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from shared.python.engine_manager import EngineManager, EngineStatus  # noqa: E402


# Helper to check if a module is mocked (from unit tests polluting sys.modules)
def is_mock(module_name: str) -> bool:
    """Check if a module in sys.modules is a mock."""
    mod = sys.modules.get(module_name)
    if mod is None:
        return False
    return isinstance(mod, MagicMock) or getattr(mod, "__file__", None) is None


# Test assets
ASSET_DIR = Path(__file__).parent.parent / "assets"
SIMPLE_ARM_URDF = ASSET_DIR / "simple_arm.urdf"


# ==============================================================================
# EXEMPLARY INTEGRATION TESTS
# ==============================================================================
# These demonstrate proper integration testing:
# 1. Actually load and integrate real components
# 2. Skip gracefully if dependencies unavailable (not fail)
# 3. Test real behavior across component boundaries
# 4. Verify actual outputs, not that mocks were called
# ==============================================================================


class TestEngineManagerIntegration:
    """Test EngineManager integration with real filesystem."""

    def test_engine_manager_discovers_real_engines(self):
        """Test that EngineManager discovers engines in actual project structure.

        GOOD PRACTICE: Integration test uses real project structure.
        This tests that EngineManager correctly navigates real directories.
        """
        manager = EngineManager()

        # Should have initialized with real project structure
        assert manager.suite_root.exists()
        assert manager.engines_root.exists()

        # Get available engines
        available = manager.get_available_engines()

        # Should have found at least some engines
        assert isinstance(available, list)

        # Each available engine should have a valid path
        for engine in available:
            path = manager.engine_paths[engine]
            assert path.exists(), f"{engine} path should exist: {path}"

    def test_engine_paths_match_filesystem(self):
        """Test that engine paths in manager match actual filesystem.

        GOOD PRACTICE: Verifies configuration matches reality.
        """
        manager = EngineManager()

        for engine_type, path in manager.engine_paths.items():
            # Path should be absolute and within suite root
            assert path.is_absolute()
            assert manager.suite_root in path.parents

            # If status says available, path must exist
            status = manager.get_engine_status(engine_type)
            if status == EngineStatus.AVAILABLE or status == EngineStatus.LOADED:
                assert (
                    path.exists()
                ), f"{engine_type} marked as {status} but path doesn't exist"


@pytest.mark.skipif(not SIMPLE_ARM_URDF.exists(), reason="Test asset missing")
class TestMuJoCoEngineIntegration:
    """Integration tests for MuJoCo engine with real loading.

    These tests only run if:
    1. MuJoCo is actually installed (not mocked)
    2. Test assets are available
    """

    @pytest.fixture
    def has_real_mujoco(self):
        """Check if real MuJoCo is available."""
        # Clean up any mocked mujoco from sys.modules
        if is_mock("mujoco"):
            sys.modules.pop("mujoco", None)

        try:
            import mujoco  # noqa: F401

            importlib.reload(mujoco)
            return True
        except (ImportError, OSError):
            pytest.skip("MuJoCo not installed or DLL load failed")

    def test_mujoco_engine_loads_real_urdf(self, has_real_mujoco):
        """Test that MuJoCo engine can load and process real URDF.

        GOOD PRACTICE: Real integration test that:
        - Uses actual MuJoCo library
        - Loads actual URDF file
        - Verifies real physics engine behavior
        """
        import importlib

        import engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine

        # Force reload to ensure we are using the REAL engine module, not the one cached with Mocks
        importlib.reload(
            engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine
        )

        from engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
            MuJoCoPhysicsEngine,
        )

        engine = MuJoCoPhysicsEngine()

        # Load real URDF file - mock security validation for test
        with patch(
            "engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine.validate_path"
        ) as mock_validate:
            mock_validate.return_value = SIMPLE_ARM_URDF
            engine.load_from_path(str(SIMPLE_ARM_URDF))

        # Verify engine loaded model correctly
        assert engine.model is not None
        assert engine.data is not None

        # Verify model has expected structure
        assert engine.model.nq > 0, "Model should have position DOFs"
        assert engine.model.nv > 0, "Model should have velocity DOFs"

    def test_mujoco_engine_simulation_step(self, has_real_mujoco):
        """Test that MuJoCo can actually simulate physics.

        GOOD PRACTICE: Tests actual physics simulation, not mocks.
        Verifies that state changes over time as expected.
        """
        import importlib

        import engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine

        # Force reload here too to ensure we get the clean import
        importlib.reload(
            engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine
        )

        from engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
            MuJoCoPhysicsEngine,
        )

        engine = MuJoCoPhysicsEngine()

        # Mock security validation for test
        with patch(
            "engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine.validate_path"
        ) as mock_validate:
            mock_validate.return_value = SIMPLE_ARM_URDF
            engine.load_from_path(str(SIMPLE_ARM_URDF))

        # Get initial state
        initial_time = engine.data.time if engine.data is not None else 0.0

        # Step simulation
        engine.step()

        # Verify state changed
        current_time = engine.data.time if engine.data is not None else 0.0
        assert current_time > initial_time, "Time should advance after step"


@pytest.mark.skipif(not SIMPLE_ARM_URDF.exists(), reason="Test asset missing")
class TestCrossEngineConsistency:
    """Integration tests comparing behavior across engines.

    GOOD PRACTICE: Real cross-engine integration testing.
    Only compares engines that are actually available.
    """

    @pytest.fixture
    def available_engines(self):
        """Get list of engines that are actually available (not mocked)."""
        engines = {}

        # Check MuJoCo
        if not is_mock("mujoco"):
            try:
                from engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
                    MuJoCoPhysicsEngine,
                )

                mj = MuJoCoPhysicsEngine()
                mj.load_from_path(str(SIMPLE_ARM_URDF))
                engines["mujoco"] = mj
            except Exception:
                pass

        # Check Drake
        if not is_mock("pydrake"):
            try:
                from engines.physics_engines.drake.python.drake_physics_engine import (
                    DrakePhysicsEngine,
                )

                dk = DrakePhysicsEngine()
                dk.load_from_path(str(SIMPLE_ARM_URDF))
                engines["drake"] = dk  # type: ignore[assignment]
            except Exception:
                pass

        # Check Pinocchio
        if not is_mock("pinocchio"):
            try:
                from engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
                    PinocchioPhysicsEngine,
                )

                pn = PinocchioPhysicsEngine()
                pn.load_from_path(str(SIMPLE_ARM_URDF))
                engines["pinocchio"] = pn  # type: ignore[assignment]
            except Exception:
                pass

        if len(engines) < 2:
            pytest.skip("Need at least 2 engines installed for comparison")

        return engines

    def test_engines_agree_on_model_dimensions(self, available_engines):
        """Test that different engines agree on basic model properties.

        GOOD PRACTICE: Real integration test comparing actual engine outputs.
        Verifies that different physics engines interpret the same URDF consistently.
        """
        # Get number of DOFs from each engine
        dof_counts = {}
        for name, engine in available_engines.items():
            if hasattr(engine, "model"):
                # MuJoCo-style
                dof_counts[name] = (
                    engine.model.nq if hasattr(engine.model, "nq") else None
                )
            elif hasattr(engine, "plant"):
                # Drake-style
                dof_counts[name] = engine.plant.num_positions()
            elif hasattr(engine, "get_state"):
                # Generic - try to get state
                try:
                    q, _ = engine.get_state()
                    dof_counts[name] = len(q)
                except Exception:
                    pass

        # Filter out None values
        valid_dofs = {k: v for k, v in dof_counts.items() if v is not None}

        if len(valid_dofs) >= 2:
            # All engines should agree on DOF count
            dof_values = list(valid_dofs.values())
            assert all(
                d == dof_values[0] for d in dof_values
            ), f"Engines disagree on DOFs: {valid_dofs}"
