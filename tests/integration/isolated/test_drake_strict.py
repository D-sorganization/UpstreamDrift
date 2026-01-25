import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest  # noqa: F401 - required for pytestmark

from src.shared.python.engine_availability import (
    skip_if_unavailable,
)

# Skip entire module if Drake is not installed - mocking pydrake at module level
# is unreliable and leads to AttributeError on patched module globals
pytestmark = skip_if_unavailable("drake")

# --- Global Mocking Setup (Duplicated for Isolation) ---
mock_pydrake = MagicMock()

# Setup patches
module_patches = {
    "pydrake": mock_pydrake,
    "pydrake.all": mock_pydrake,
    "pydrake.multibody": mock_pydrake,
    "pydrake.multibody.parsing": mock_pydrake,
    "pydrake.multibody.plant": mock_pydrake,
    "pydrake.systems": mock_pydrake,
    "pydrake.systems.framework": mock_pydrake,
    "pydrake.systems.analysis": mock_pydrake,
    "pydrake.math": mock_pydrake,
}

mock_pydrake.DiagramBuilder = MagicMock()
mock_pydrake.systems.framework.DiagramBuilder = mock_pydrake.DiagramBuilder


class TestDrakeStrict:
    def setup_method(self):
        """Inject mock pydrake into the module namespace."""
        # Use patch.dict context for the import to apply patches
        with patch.dict("sys.modules", module_patches):
            # We must import inside the patch context to ensure the module picks up the mocks
            # Note: In an isolated process, we don't strictly need reload(),
            # but we do need to ensure duplicate imports don't mess things up if run repeatedly.
            import engines.physics_engines.drake.python.drake_physics_engine as mod

            # Since we are in a fresh process (ideally), just importing might be enough.
            # But the 'mod' variable scope is what we need.

        # Re-import to capture reference (it will use the sys.modules cache we just populated/patched?)
        # Actually, simpler: define the class execution *inside* the patch if possible,
        # or just rely on sys.modules remaining patched if we don't exit context?
        # The correct way for this test harness:

        self.patcher = patch.dict("sys.modules", module_patches)
        self.patcher.start()

        self.mod = mod
        self.DrakePhysicsEngine = mod.DrakePhysicsEngine

        # Test Constants
        self.TEST_LINEAR_VAL = 1.0
        self.TEST_ANGULAR_VAL = 2.0

    def teardown_method(self):
        self.patcher.stop()

    def test_jacobian_standardization_mocked(self):
        engine = self.DrakePhysicsEngine()
        # Mock internals set by AddMultibodyPlantSceneGraph
        engine.plant = MagicMock()
        engine.plant_context = MagicMock()

        # Mock output of CalcJacobianSpatialVelocity
        # Drake returns (w, v) -> Angular, Linear
        J_fake = np.zeros((6, 2))
        J_fake[:3, :] = self.TEST_ANGULAR_VAL  # Angular
        J_fake[3:, :] = self.TEST_LINEAR_VAL  # Linear
        engine.plant.CalcJacobianSpatialVelocity.return_value = J_fake
        # Ensure body lookup works
        engine.plant.GetBodyByName.return_value = MagicMock()

        jac = engine.compute_jacobian("foo")
        assert jac is not None

        spatial = jac["spatial"]
        # Drake engine should pass J through directly as it is already [Angular; Linear]
        np.testing.assert_allclose(spatial[:3, :], self.TEST_ANGULAR_VAL)
        np.testing.assert_allclose(spatial[3:, :], self.TEST_LINEAR_VAL)

    def test_reset_protection(self, caplog):
        """Drake reset should warn if uninitialized."""
        engine = self.DrakePhysicsEngine()
        engine.context = None  # Force uninitialized

        with caplog.at_level(logging.WARNING):
            engine.reset()

        assert "Attempted to reset Drake engine before initialization." in caplog.text
