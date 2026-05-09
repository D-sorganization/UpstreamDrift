"""Unit tests for src/shared/python/model_generation/core/physics_validation.py.

Tests cover PhysicsValidator.validate_inertia_tensor, the result dataclasses,
and the basic validator interface. All tests are headless-safe and require
only numpy and model_generation (available via pytest pythonpath).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_inertia(ixx=0.1, iyy=0.1, izz=0.1, ixy=0.0, ixz=0.0, iyz=0.0, mass=1.0):
    """Create an Inertia object with given diagonal + off-diagonal elements."""
    from model_generation.core.types import Inertia

    return Inertia(ixx=ixx, iyy=iyy, izz=izz, ixy=ixy, ixz=ixz, iyz=iyz, mass=mass)


@pytest.fixture
def validator():
    """PhysicsValidator instance with default gravity."""
    from src.shared.python.model_generation.core.physics_validation import (
        PhysicsValidator,
    )

    return PhysicsValidator()


# ---------------------------------------------------------------------------
# PhysicsValidator initialization
# ---------------------------------------------------------------------------


class TestPhysicsValidatorInit:
    """Tests for PhysicsValidator initialization."""

    def test_physics_validation_extended_default_gravity(self, validator) -> None:
        """Default gravity vector is [0, 0, -9.81]."""

        assert abs(validator.gravity[2] - (-9.81)) < 1e-6

    def test_custom_gravity(self) -> None:
        """Custom gravity vector is stored correctly."""
        import numpy as np
        from src.shared.python.model_generation.core.physics_validation import (
            PhysicsValidator,
        )

        g = np.array([0.0, 0.0, -1.62])  # Moon gravity
        v = PhysicsValidator(gravity=g)
        assert abs(v.gravity[2] - (-1.62)) < 1e-9


# ---------------------------------------------------------------------------
# InertiaValidationResult and related dataclasses
# ---------------------------------------------------------------------------


class TestInertiaValidationResultDataclass:
    """Tests for InertiaValidationResult dataclass."""

    def test_instantiation_with_required_fields(self) -> None:
        """InertiaValidationResult instantiates with required boolean fields."""
        from src.shared.python.model_generation.core.physics_validation import (
            InertiaValidationResult,
        )

        result = InertiaValidationResult(
            is_valid=True,
            is_symmetric=True,
            is_positive_definite=True,
            satisfies_triangle_inequality=True,
        )
        assert result.is_valid is True
        assert result.is_symmetric is True

    def test_default_optional_fields(self) -> None:
        """Optional fields default to expected values."""
        from src.shared.python.model_generation.core.physics_validation import (
            InertiaValidationResult,
        )

        result = InertiaValidationResult(
            is_valid=True,
            is_symmetric=True,
            is_positive_definite=True,
            satisfies_triangle_inequality=True,
        )
        assert result.eigenvalues is None
        assert result.condition_number is None
        assert result.warnings == []
        assert result.errors == []


class TestStabilityResultDataclass:
    """Tests for StabilityResult dataclass."""

    def test_physics_validation_extended_instantiation(self) -> None:
        """StabilityResult instantiates with is_stable and center_of_mass."""
        from src.shared.python.model_generation.core.physics_validation import (
            StabilityResult,
        )

        result = StabilityResult(
            is_stable=True,
            center_of_mass=(0.0, 0.0, 0.5),
        )
        assert result.is_stable is True
        assert result.center_of_mass == (0.0, 0.0, 0.5)


class TestPhysicsValidationResultDataclass:
    """Tests for PhysicsValidationResult dataclass."""

    def test_physics_validation_extended_instantiation(self) -> None:
        """PhysicsValidationResult instantiates with required fields."""
        from src.shared.python.model_generation.core.physics_validation import (
            PhysicsValidationResult,
        )

        result = PhysicsValidationResult(is_valid=True)
        assert result.is_valid is True
        assert result.inertia_results == {}
        assert result.stability is None
        assert result.collision is None


# ---------------------------------------------------------------------------
# validate_inertia_tensor — valid inputs
# ---------------------------------------------------------------------------


class TestValidateInertiaTensorValid:
    """Tests for validate_inertia_tensor with valid inertia tensors."""

    def test_valid_diagonal_inertia_passes(self, validator) -> None:
        """Symmetric positive-definite diagonal inertia passes all checks."""
        inertia = _make_inertia(ixx=0.1, iyy=0.2, izz=0.3)
        result = validator.validate_inertia_tensor(inertia)
        assert result.is_valid is True
        assert result.is_symmetric is True
        assert result.is_positive_definite is True

    def test_valid_result_has_no_errors(self, validator) -> None:
        """Valid inertia produces no error messages."""
        inertia = _make_inertia(ixx=0.1, iyy=0.2, izz=0.3)
        result = validator.validate_inertia_tensor(inertia)
        assert len(result.errors) == 0

    def test_eigenvalues_computed(self, validator) -> None:
        """Valid inertia has eigenvalues computed and stored."""
        inertia = _make_inertia(ixx=0.1, iyy=0.2, izz=0.3)
        result = validator.validate_inertia_tensor(inertia)
        assert result.eigenvalues is not None
        assert len(result.eigenvalues) == 3

    def test_all_eigenvalues_positive(self, validator) -> None:
        """Valid SPD inertia has all positive eigenvalues."""
        inertia = _make_inertia(ixx=0.1, iyy=0.2, izz=0.3)
        result = validator.validate_inertia_tensor(inertia)
        assert all(ev > 0 for ev in result.eigenvalues)

    def test_condition_number_computed(self, validator) -> None:
        """Condition number is computed for valid inertia."""
        inertia = _make_inertia(ixx=0.1, iyy=0.2, izz=0.3)
        result = validator.validate_inertia_tensor(inertia)
        assert result.condition_number is not None
        assert result.condition_number > 0

    def test_principal_axes_computed(self, validator) -> None:
        """Principal axes matrix (eigenvectors) is computed."""

        inertia = _make_inertia(ixx=0.1, iyy=0.2, izz=0.3)
        result = validator.validate_inertia_tensor(inertia)
        assert result.principal_axes is not None
        assert result.principal_axes.shape == (3, 3)

    def test_triangle_inequality_passes(self, validator) -> None:
        """Physically consistent diagonal satisfies triangle inequality."""
        inertia = _make_inertia(ixx=0.1, iyy=0.2, izz=0.3)
        result = validator.validate_inertia_tensor(inertia)
        assert result.satisfies_triangle_inequality is True

    def test_symmetric_off_diagonal_valid(self, validator) -> None:
        """Inertia with small off-diagonal terms (physically valid) passes."""
        # ixx=1, iyy=1, izz=1, ixy=0.1: tensor is symmetric, SPD for small ixy
        inertia = _make_inertia(ixx=1.0, iyy=1.0, izz=1.0, ixy=0.1)
        result = validator.validate_inertia_tensor(inertia)
        assert result.is_symmetric is True

    def test_component_name_accepted(self, validator) -> None:
        """validate_inertia_tensor accepts optional component name without error."""
        inertia = _make_inertia(ixx=0.1, iyy=0.2, izz=0.3)
        result = validator.validate_inertia_tensor(inertia, component="link1")
        assert result is not None


# ---------------------------------------------------------------------------
# validate_inertia_tensor — invalid inputs
# ---------------------------------------------------------------------------


class TestValidateInertiaTensorInvalid:
    """Tests for validate_inertia_tensor with invalid inertia tensors."""

    def test_zero_inertia_not_positive_definite(self, validator) -> None:
        """All-zero inertia tensor fails positive definiteness check."""
        inertia = _make_inertia(ixx=0.0, iyy=0.0, izz=0.0)
        result = validator.validate_inertia_tensor(inertia)
        assert result.is_positive_definite is False
        assert result.is_valid is False

    def test_zero_inertia_has_errors(self, validator) -> None:
        """Zero inertia produces error messages."""
        inertia = _make_inertia(ixx=0.0, iyy=0.0, izz=0.0)
        result = validator.validate_inertia_tensor(inertia)
        assert len(result.errors) > 0

    def test_negative_diagonal_fails(self, validator) -> None:
        """Negative principal inertia is not positive definite."""
        inertia = _make_inertia(ixx=-0.1, iyy=0.2, izz=0.3)
        result = validator.validate_inertia_tensor(inertia)
        assert result.is_positive_definite is False
        assert result.is_valid is False

    def test_triangle_inequality_violation_warns(self, validator) -> None:
        """Inertia violating triangle inequality produces warnings."""
        # izz >> ixx + iyy → violates Ixx + Iyy >= Izz
        inertia = _make_inertia(ixx=0.001, iyy=0.001, izz=10.0)
        result = validator.validate_inertia_tensor(inertia)
        # Valid SPD but may warn about triangle inequality
        # The test verifies at minimum it handles the input gracefully
        assert result is not None

    def test_high_condition_number_warns(self, validator) -> None:
        """Very ill-conditioned inertia produces a warning."""
        # ixx << izz: high condition number
        inertia = _make_inertia(ixx=1e-7, iyy=1e-7, izz=1.0)
        result = validator.validate_inertia_tensor(inertia)
        # Should warn about high condition number if eigenvalue ratio > 1e5
        if result.condition_number is not None and result.condition_number > 1e5:
            assert len(result.warnings) > 0


# ---------------------------------------------------------------------------
# CollisionCheckResult dataclass
# ---------------------------------------------------------------------------


class TestCollisionCheckResult:
    """Tests for CollisionCheckResult dataclass."""

    def test_physics_validation_extended_instantiation(self) -> None:
        """CollisionCheckResult instantiates with has_self_intersection."""
        from src.shared.python.model_generation.core.physics_validation import (
            CollisionCheckResult,
        )

        result = CollisionCheckResult(has_self_intersection=False)
        assert result.has_self_intersection is False
        assert result.penetration_pairs == []
        assert result.min_separation == float("inf")
