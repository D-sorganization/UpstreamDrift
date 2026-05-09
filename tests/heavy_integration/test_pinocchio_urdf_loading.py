"""Heavy integration tests for Pinocchio URDF loading (fixes #1988).

Tests that pinocchio can load a URDF from disk and produce physically
consistent FK, inverse dynamics, and Jacobian computations.

All tests skip gracefully when pinocchio is not installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

GOLFER_URDF = (
    Path(__file__).parents[2]
    / "src/engines/physics_engines/pinocchio/models/generated/golfer.urdf"
)


def _pin():
    """Import pinocchio or skip."""
    try:
        import pinocchio as pin

        return pin
    except ImportError:
        pytest.skip("pinocchio not installed")


@pytest.fixture(scope="module")
def golfer_model():
    """Load the golfer URDF; skip if file or pinocchio unavailable."""
    pin = _pin()
    if not GOLFER_URDF.exists():
        pytest.skip(f"Golfer URDF not found at {GOLFER_URDF}")

    model = pin.buildModelFromUrdf(str(GOLFER_URDF))
    data = model.createData()
    return model, data


class TestPinocchioUrdfLoading:
    """Contract: Pinocchio loads a URDF and model metadata is consistent."""

    def test_model_loads_from_urdf(self, golfer_model) -> None:
        """Model has at least 1 DOF and 1 frame after loading."""
        model, _ = golfer_model
        assert model.nq > 0, "Model has no configuration DOF"
        assert model.nv > 0, "Model has no velocity DOF"
        assert model.nframes > 0, "Model has no frames"

    def test_zero_config_is_valid(self, golfer_model) -> None:
        """neutral() returns a finite configuration vector of shape (nq,)."""
        pin = _pin()
        model, _ = golfer_model
        q0 = pin.neutral(model)
        assert q0.shape == (model.nq,)
        assert np.all(np.isfinite(q0))


class TestPinocchioForwardKinematics:
    """Contract: FK produces valid SE(3) frames."""

    def test_fk_at_zero_config(self, golfer_model) -> None:
        """FK at neutral config produces valid homogeneous transforms."""
        pin = _pin()
        model, data = golfer_model
        q = pin.neutral(model)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        for fid in range(model.nframes):
            T = data.oMf[fid]
            R = T.rotation
            # Check rotation is orthogonal
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)
            assert abs(np.linalg.det(R) - 1.0) < 1e-9

    def test_fk_random_configs_finite(self, golfer_model) -> None:
        """FK at 5 random configs stays finite."""
        pin = _pin()
        model, data = golfer_model
        for _ in range(5):
            q = pin.randomConfiguration(model)
            pin.forwardKinematics(model, data, q)
            for fid in range(model.nframes):
                T = data.oMf[fid]
                assert np.all(np.isfinite(T.translation))


class TestPinocchioInverseDynamics:
    """Contract: M*qacc + bias = tau for zero acceleration."""

    def test_rnea_zero_acceleration(self, golfer_model) -> None:
        """RNEA with zero qacc returns finite torques."""
        pin = _pin()
        model, data = golfer_model
        q = pin.neutral(model)
        v = np.zeros(model.nv)
        a = np.zeros(model.nv)
        tau = pin.rnea(model, data, q, v, a)
        assert tau.shape == (model.nv,)
        assert np.all(np.isfinite(tau))

    def test_pinocchio_urdf_loading_mass_matrix_positive_definite(
        self, golfer_model
    ) -> None:
        """Mass matrix at neutral config is symmetric positive-definite."""
        pin = _pin()
        model, data = golfer_model
        q = pin.neutral(model)
        M = pin.crba(model, data, q)
        # Symmetry
        np.testing.assert_allclose(M, M.T, atol=1e-10)
        # Positive definiteness — all eigenvalues > 0
        eigvals = np.linalg.eigvalsh(M)
        assert np.all(eigvals > 0), f"Non-positive eigenvalues: {eigvals[eigvals <= 0]}"


class TestPinocchioJacobian:
    """Contract: Jacobian has correct shape and is finite."""

    def test_jacobian_shape(self, golfer_model) -> None:
        """Jacobian w.r.t. last frame has shape (6, nv)."""
        pin = _pin()
        model, data = golfer_model
        q = pin.neutral(model)
        pin.computeJointJacobians(model, data, q)
        # Use the last body frame as end-effector proxy
        last_frame_id = model.nframes - 1
        J = pin.getFrameJacobian(
            model, data, last_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        assert J.shape == (6, model.nv)
        assert np.all(np.isfinite(J))


pytestmark = pytest.mark.live_simulation
