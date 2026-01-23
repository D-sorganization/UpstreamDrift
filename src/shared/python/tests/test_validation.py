"""Unit tests for validation.py."""

import numpy as np
import pytest
from shared.python.validation import (
    PhysicalValidationError,
    validate_friction_coefficient,
    validate_inertia_matrix,
    validate_joint_limits,
    validate_mass,
    validate_physical_bounds,
    validate_timestep,
)


class TestValidation:
    """Tests for physical validation functions."""

    def test_validate_mass(self):
        """Test validate_mass."""
        validate_mass(1.0)
        validate_mass(1e-6)

        with pytest.raises(PhysicalValidationError, match="mass > 0"):
            validate_mass(0.0)

        with pytest.raises(PhysicalValidationError, match="mass > 0"):
            validate_mass(-1.0)

    def test_validate_timestep(self):
        """Test validate_timestep."""
        validate_timestep(0.01)
        validate_timestep(1e-4)

        with pytest.raises(PhysicalValidationError, match="dt > 0"):
            validate_timestep(0.0)

        with pytest.raises(PhysicalValidationError, match="dt > 0"):
            validate_timestep(-0.01)

        # Suspiciously large timestep (should warn, not raise)
        # Assuming logger warning doesn't raise, checking logic passes
        validate_timestep(1.1)

    def test_validate_inertia_matrix(self):
        """Test validate_inertia_matrix."""
        # Valid: Identity
        validate_inertia_matrix(np.eye(3))

        # Valid: Diagonal positive
        validate_inertia_matrix(np.diag([1.0, 2.0, 3.0]))

        # Invalid: Shape
        with pytest.raises(PhysicalValidationError, match="shape"):
            validate_inertia_matrix(np.eye(2))

        # Invalid: Not symmetric
        asym = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        with pytest.raises(PhysicalValidationError, match="symmetric"):
            validate_inertia_matrix(asym)

        # Invalid: Not positive definite (negative eigenvalue)
        neg_eig = np.diag([1.0, -1.0, 1.0])
        with pytest.raises(PhysicalValidationError, match="positive definite"):
            validate_inertia_matrix(neg_eig)

        # Invalid: Not positive definite (zero eigenvalue)
        zero_eig = np.diag([1.0, 0.0, 1.0])
        with pytest.raises(PhysicalValidationError, match="positive definite"):
            validate_inertia_matrix(zero_eig)

    def test_validate_joint_limits(self):
        """Test validate_joint_limits."""
        q_min = np.array([-1.0, -1.0])
        q_max = np.array([1.0, 1.0])

        validate_joint_limits(q_min, q_max)

        # Invalid: q_min >= q_max
        q_min_bad = np.array([1.0, -1.0])
        with pytest.raises(PhysicalValidationError, match="q_min < q_max"):
            validate_joint_limits(q_min_bad, q_max)

        # Invalid: Shape mismatch
        with pytest.raises(PhysicalValidationError, match="shape mismatch"):
            validate_joint_limits(np.zeros(3), q_max)

    def test_validate_friction_coefficient(self):
        """Test validate_friction_coefficient."""
        validate_friction_coefficient(0.0)
        validate_friction_coefficient(0.5)

        with pytest.raises(PhysicalValidationError, match="friction coefficient >= 0"):
            validate_friction_coefficient(-0.1)


class TestValidatePhysicalBoundsDecorator:
    """Tests for validate_physical_bounds decorator."""

    @staticmethod
    @validate_physical_bounds
    def dummy_func(
        mass: float = 1.0,
        dt: float = 0.01,
        inertia: np.ndarray | None = None,
        friction: float = 0.5,
        q_min: np.ndarray | None = None,
        q_max: np.ndarray | None = None,
    ):
        return True

    def test_decorator_pass(self):
        """Test valid inputs pass through."""
        assert self.dummy_func()
        assert self.dummy_func(mass=2.0)
        assert self.dummy_func(friction=0.0)
        assert self.dummy_func(inertia=np.eye(3))

    def test_decorator_mass_validation(self):
        """Test mass validation via decorator."""
        with pytest.raises(PhysicalValidationError, match="mass > 0"):
            self.dummy_func(mass=-1.0)

    def test_decorator_timestep_validation(self):
        """Test timestep validation via decorator."""
        with pytest.raises(PhysicalValidationError, match="dt > 0"):
            self.dummy_func(dt=0.0)

    def test_decorator_friction_validation(self):
        """Test friction validation via decorator."""
        with pytest.raises(PhysicalValidationError, match="friction coefficient >= 0"):
            self.dummy_func(friction=-0.5)

    def test_decorator_inertia_validation(self):
        """Test inertia validation via decorator."""
        with pytest.raises(PhysicalValidationError, match="positive definite"):
            self.dummy_func(inertia=np.diag([1.0, -1.0, 1.0]))

    def test_decorator_joint_limits_validation(self):
        """Test joint limits validation via decorator."""
        q_min = np.array([0.0])
        q_max = np.array([-1.0])
        with pytest.raises(PhysicalValidationError, match="q_min < q_max"):
            self.dummy_func(q_min=q_min, q_max=q_max)
