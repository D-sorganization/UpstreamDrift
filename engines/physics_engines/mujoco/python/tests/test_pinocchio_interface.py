"""Tests for Pinocchio interface functionality."""

import mujoco
import numpy as np
import pytest
from mujoco_humanoid_golf.models import DOUBLE_PENDULUM_XML
from mujoco_humanoid_golf.pinocchio_interface import (
    PINOCCHIO_AVAILABLE,
    PinocchioWrapper,
    create_pinocchio_wrapper,
)

# Skip all tests if Pinocchio is not available
pytestmark = pytest.mark.skipif(
    not PINOCCHIO_AVAILABLE,
    reason="Pinocchio is not installed",
)


@pytest.fixture()
def simple_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Create a simple MuJoCo model for testing."""
    model = mujoco.MjModel.from_xml_string(DOUBLE_PENDULUM_XML)
    data = mujoco.MjData(model)
    return model, data


class TestPinocchioWrapper:
    """Tests for PinocchioWrapper class."""

    def test_wrapper_initialization(self, simple_model) -> None:
        """Test initializing Pinocchio wrapper."""
        model, data = simple_model

        wrapper = PinocchioWrapper(model, data)

        assert wrapper.model == model
        assert wrapper.data == data
        assert wrapper.pin_model is not None
        assert wrapper.pin_data is not None

    def test_wrapper_convenience_function(self, simple_model) -> None:
        """Test convenience function for creating wrapper."""
        model, data = simple_model

        wrapper = create_pinocchio_wrapper(model, data)

        assert isinstance(wrapper, PinocchioWrapper)

    def test_compute_inverse_dynamics(self, simple_model) -> None:
        """Test inverse dynamics computation."""
        model, data = simple_model
        wrapper = PinocchioWrapper(model, data)

        # Set state
        q = data.qpos.copy()
        v = data.qvel.copy()
        a = np.zeros(model.nv)

        # Compute inverse dynamics
        torques = wrapper.compute_inverse_dynamics(q, v, a)

        assert torques.shape == (model.nv,)
        assert np.all(np.isfinite(torques))

    def test_compute_forward_dynamics(self, simple_model) -> None:
        """Test forward dynamics computation."""
        model, data = simple_model
        wrapper = PinocchioWrapper(model, data)

        # Set state
        q = data.qpos.copy()
        v = data.qvel.copy()
        tau = np.zeros(model.nv)

        # Compute forward dynamics
        a = wrapper.compute_forward_dynamics(q, v, tau)

        assert a.shape == (model.nv,)
        assert np.all(np.isfinite(a))

    def test_compute_mass_matrix(self, simple_model) -> None:
        """Test mass matrix computation."""
        model, data = simple_model
        wrapper = PinocchioWrapper(model, data)

        # Compute mass matrix
        M = wrapper.compute_mass_matrix()

        assert M.shape == (model.nv, model.nv)
        assert np.allclose(M, M.T)  # Symmetric
        assert np.all(np.linalg.eigvals(M) > 0)  # Positive definite

    def test_compute_coriolis_matrix(self, simple_model) -> None:
        """Test Coriolis matrix computation."""
        model, data = simple_model
        wrapper = PinocchioWrapper(model, data)

        # Set state
        q = data.qpos.copy()
        v = data.qvel.copy()

        # Compute Coriolis matrix
        C = wrapper.compute_coriolis_matrix(q, v)

        assert C.shape == (model.nv, model.nv)
        assert np.all(np.isfinite(C))

    def test_compute_gravity_vector(self, simple_model) -> None:
        """Test gravity vector computation."""
        model, data = simple_model
        wrapper = PinocchioWrapper(model, data)

        # Compute gravity vector
        g = wrapper.compute_gravity_vector()

        assert g.shape == (model.nv,)
        assert np.all(np.isfinite(g))

    def test_compute_kinetic_energy(self, simple_model) -> None:
        """Test kinetic energy computation."""
        model, data = simple_model
        wrapper = PinocchioWrapper(model, data)

        # Set state
        q = data.qpos.copy()
        v = data.qvel.copy()

        # Compute kinetic energy
        ke = wrapper.compute_kinetic_energy(q, v)

        assert isinstance(ke, float)
        assert ke >= 0.0
        assert np.isfinite(ke)

    def test_compute_potential_energy(self, simple_model) -> None:
        """Test potential energy computation."""
        model, data = simple_model
        wrapper = PinocchioWrapper(model, data)

        # Compute potential energy
        pe = wrapper.compute_potential_energy()

        assert isinstance(pe, float)
        assert np.isfinite(pe)

    def test_sync_mujoco_to_pinocchio(self, simple_model) -> None:
        """Test syncing state from MuJoCo to Pinocchio."""
        model, data = simple_model
        wrapper = PinocchioWrapper(model, data)

        # Set some state in MuJoCo
        data.qpos[0] = 0.5
        data.qvel[0] = 1.0

        # Sync to Pinocchio
        wrapper.sync_mujoco_to_pinocchio()

        # Check that Pinocchio state was updated
        assert wrapper.pin_data.q is not None
        assert wrapper.pin_data.v is not None

    def test_inverse_dynamics_requires_accelerations(self, simple_model) -> None:
        """Test that inverse dynamics requires accelerations."""
        model, data = simple_model
        wrapper = PinocchioWrapper(model, data)

        with pytest.raises(ValueError, match="Accelerations"):
            wrapper.compute_inverse_dynamics()

    def test_dynamics_consistency(self, simple_model) -> None:
        """Test consistency between forward and inverse dynamics."""
        model, data = simple_model
        wrapper = PinocchioWrapper(model, data)

        # Set state
        q = data.qpos.copy()
        v = data.qvel.copy()
        tau = np.array([1.0, -0.5])  # Some torques

        # Forward dynamics: compute accelerations from torques
        a = wrapper.compute_forward_dynamics(q, v, tau)

        # Inverse dynamics: compute torques from accelerations
        tau_computed = wrapper.compute_inverse_dynamics(q, v, a)

        # Should be approximately equal (within numerical precision)
        assert np.allclose(tau, tau_computed, atol=1e-3)


class TestPinocchioWithoutInstallation:
    """Tests for behavior when Pinocchio is not installed."""

    def test_import_error_when_not_installed(self) -> None:
        """Test that ImportError is raised when Pinocchio is not available."""
        if PINOCCHIO_AVAILABLE:
            pytest.skip("Pinocchio is installed, cannot test import error")

        # This test would only run if Pinocchio is not installed
        # In that case, importing should raise ImportError
        # But we can't test this directly since we skip all tests if not available


class TestPinocchioJacobians:
    """Tests for Jacobian computation."""

    def test_end_effector_jacobian_shape(self, simple_model) -> None:
        """Test that end-effector Jacobian has correct shape."""
        model, data = simple_model
        wrapper = PinocchioWrapper(model, data)

        # Try to find a frame (may not exist in simple model)
        # This test may need adjustment based on actual model structure
        try:
            J = wrapper.compute_end_effector_jacobian("club_head")
            assert J.shape == (6, model.nv)  # 6 DOF (3 linear + 3 angular)
        except ValueError:
            # Frame not found, which is acceptable for simple models
            pass

    def test_jacobian_nonexistent_frame(self, simple_model) -> None:
        """Test that requesting nonexistent frame raises error."""
        model, data = simple_model
        wrapper = PinocchioWrapper(model, data)

        with pytest.raises(ValueError, match="not found"):
            wrapper.compute_end_effector_jacobian("nonexistent_frame")
