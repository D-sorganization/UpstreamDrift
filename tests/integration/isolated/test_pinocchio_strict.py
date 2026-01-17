
import logging
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

# --- Global Mocking Setup (Duplicated for Isolation) ---
mock_pinocchio = MagicMock()

# Setup patches
module_patches = {
    "pinocchio": mock_pinocchio,
}

class TestPinocchioStrict:
    def setup_method(self):
        """Inject mock pinocchio into the module namespace."""
        self.patcher = patch.dict("sys.modules", module_patches)
        self.patcher.start()
        
        import engines.physics_engines.pinocchio.python.pinocchio_physics_engine as mod
        self.mod = mod
        self.PinocchioPhysicsEngine = mod.PinocchioPhysicsEngine
        
        # Test Constants
        self.TEST_LINEAR_VAL = 1.0
        self.TEST_ANGULAR_VAL = 2.0

    def teardown_method(self):
        self.patcher.stop()

    def test_jacobian_standardization_mocked(self):
        engine = self.PinocchioPhysicsEngine()
        engine.model = MagicMock()
        engine.data = MagicMock()

        # Mock frame lookup success
        engine.model.existFrame.return_value = True
        engine.model.getFrameId.return_value = 1

        # Pinocchio returns [Linear; Angular] natively from getFrameJacobian
        J_native = np.zeros((6, 2))
        J_native[:3, :] = self.TEST_LINEAR_VAL  # Linear (top)
        J_native[3:, :] = self.TEST_ANGULAR_VAL  # Angular (bottom)

        mock_pinocchio.getFrameJacobian.return_value = J_native

        jac = engine.compute_jacobian("foo")
        assert jac is not None

        # We upgraded Pinocchio to re-stack to [Angular; Linear] (MuJoCo/Drake standard)
        spatial = jac["spatial"]
        # Top 3 should now be Angular (2.0)
        np.testing.assert_allclose(
            spatial[:3, :],
            self.TEST_ANGULAR_VAL,
            err_msg="Pinocchio spatial top should be re-stacked to Angular",
        )
        # Bottom 3 should now be Linear (1.0)
        np.testing.assert_allclose(
            spatial[3:, :],
            self.TEST_LINEAR_VAL,
            err_msg="Pinocchio spatial bottom should be re-stacked to Linear",
        )

    def test_compute_jacobian_missing_frame_and_body(self):
        """Test behavior when neither frame nor body exists."""
        engine = self.PinocchioPhysicsEngine()
        engine.model = MagicMock()
        engine.model.existFrame.return_value = False
        engine.model.existBodyName.return_value = False

        jac = engine.compute_jacobian("missing_link")
        assert jac is None
