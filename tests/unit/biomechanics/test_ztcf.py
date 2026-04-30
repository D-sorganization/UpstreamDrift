"""Tests for Zero-Torque Counterfactual (ZTCF) force computation.

Covers:
- ZTCFResult data structure construction and validation
- compute_ztcf_forces with known physics inputs
- compute_force_delta between total and ZTCF forces
- Precondition/postcondition enforcement (DbC)
- Edge cases: zero velocity, zero gravity, single joint
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.biomechanics.ztcf import (
    ZTCFResult,
    compute_force_delta,
    compute_ztcf_accelerations,
    compute_ztcf_forces,
)
from src.shared.python.core.contracts.exceptions import (
    PreconditionError,
)

# ============================================================================
# ZTCFResult dataclass
# ============================================================================


class TestZTCFResult:
    """Tests for the ZTCFResult container."""

    def test_construction_with_valid_data(self) -> None:
        forces = np.array([[10.0, 20.0], [30.0, 40.0]])
        accelerations = np.array([1.0, 2.0])
        result = ZTCFResult(
            joint_forces=forces,
            joint_accelerations=accelerations,
            n_joints=2,
        )
        assert result.n_joints == 2
        np.testing.assert_array_equal(result.joint_forces, forces)
        np.testing.assert_array_equal(result.joint_accelerations, accelerations)

    def test_force_at_joint_valid_index(self) -> None:
        forces = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = ZTCFResult(
            joint_forces=forces,
            joint_accelerations=np.array([0.5, 0.6]),
            n_joints=2,
        )
        np.testing.assert_array_equal(result.force_at_joint(0), np.array([1.0, 2.0]))
        np.testing.assert_array_equal(result.force_at_joint(1), np.array([3.0, 4.0]))

    def test_force_at_joint_invalid_index_raises(self) -> None:
        result = ZTCFResult(
            joint_forces=np.array([[1.0, 2.0]]),
            joint_accelerations=np.array([0.5]),
            n_joints=1,
        )
        with pytest.raises(PreconditionError):
            result.force_at_joint(5)

    def test_force_at_joint_negative_index_raises(self) -> None:
        result = ZTCFResult(
            joint_forces=np.array([[1.0, 2.0]]),
            joint_accelerations=np.array([0.5]),
            n_joints=1,
        )
        with pytest.raises(PreconditionError):
            result.force_at_joint(-1)

    def test_magnitudes(self) -> None:
        forces = np.array([[3.0, 4.0], [0.0, 5.0]])
        result = ZTCFResult(
            joint_forces=forces,
            joint_accelerations=np.array([0.1, 0.2]),
            n_joints=2,
        )
        mags = result.magnitudes()
        np.testing.assert_allclose(mags, [5.0, 5.0])

    def test_max_magnitude(self) -> None:
        forces = np.array([[3.0, 4.0], [0.0, 10.0]])
        result = ZTCFResult(
            joint_forces=forces,
            joint_accelerations=np.array([0.1, 0.2]),
            n_joints=2,
        )
        assert result.max_magnitude() == pytest.approx(10.0)


# ============================================================================
# compute_ztcf_accelerations
# ============================================================================


class TestComputeZTCFAccelerations:
    """Tests for zero-torque counterfactual acceleration computation."""

    def test_identity_mass_matrix_gravity_only(self) -> None:
        """With M=I, zero velocity, ZTCF accel = -gravity_vector."""
        mass_matrix = np.eye(2)
        coriolis_vector = np.zeros(2)
        gravity_vector = np.array([5.0, 3.0])
        friction_vector = np.zeros(2)

        result = compute_ztcf_accelerations(
            mass_matrix=mass_matrix,
            coriolis_vector=coriolis_vector,
            gravity_vector=gravity_vector,
            friction_vector=friction_vector,
        )
        # rhs = friction - coriolis - gravity = 0 - 0 - [5,3] = [-5, -3]
        # qddot = M^-1 * rhs = [-5, -3]
        np.testing.assert_allclose(result, [-5.0, -3.0])

    def test_with_friction(self) -> None:
        """Friction torques contribute to passive dynamics."""
        mass_matrix = np.eye(2)
        coriolis_vector = np.zeros(2)
        gravity_vector = np.array([10.0, 0.0])
        friction_vector = np.array([2.0, 1.0])

        result = compute_ztcf_accelerations(
            mass_matrix=mass_matrix,
            coriolis_vector=coriolis_vector,
            gravity_vector=gravity_vector,
            friction_vector=friction_vector,
        )
        # rhs = [2,1] - [0,0] - [10,0] = [-8, 1]
        np.testing.assert_allclose(result, [-8.0, 1.0])

    def test_with_coriolis(self) -> None:
        """Coriolis/centrifugal terms contribute to passive dynamics."""
        mass_matrix = np.eye(2)
        coriolis_vector = np.array([3.0, 2.0])
        gravity_vector = np.array([1.0, 1.0])
        friction_vector = np.zeros(2)

        result = compute_ztcf_accelerations(
            mass_matrix=mass_matrix,
            coriolis_vector=coriolis_vector,
            gravity_vector=gravity_vector,
            friction_vector=friction_vector,
        )
        # rhs = 0 - [3,2] - [1,1] = [-4, -3]
        np.testing.assert_allclose(result, [-4.0, -3.0])

    def test_non_identity_mass_matrix(self) -> None:
        """Non-trivial mass matrix correctly inverted."""
        mass_matrix = np.array([[2.0, 0.5], [0.5, 1.0]])
        coriolis_vector = np.zeros(2)
        gravity_vector = np.array([7.0, 3.5])
        friction_vector = np.zeros(2)

        result = compute_ztcf_accelerations(
            mass_matrix=mass_matrix,
            coriolis_vector=coriolis_vector,
            gravity_vector=gravity_vector,
            friction_vector=friction_vector,
        )
        # Verify M * result == rhs
        rhs = -gravity_vector
        np.testing.assert_allclose(mass_matrix @ result, rhs, atol=1e-12)

    def test_singular_mass_matrix_raises(self) -> None:
        """Singular mass matrix must raise PreconditionError."""
        singular_M = np.array([[1.0, 1.0], [1.0, 1.0]])
        with pytest.raises(PreconditionError):
            compute_ztcf_accelerations(
                mass_matrix=singular_M,
                coriolis_vector=np.zeros(2),
                gravity_vector=np.zeros(2),
                friction_vector=np.zeros(2),
            )

    def test_mismatched_dimensions_raises(self) -> None:
        """Dimension mismatch between M and vectors must raise."""
        with pytest.raises(PreconditionError):
            compute_ztcf_accelerations(
                mass_matrix=np.eye(3),
                coriolis_vector=np.zeros(2),
                gravity_vector=np.zeros(2),
                friction_vector=np.zeros(2),
            )

    def test_non_finite_input_raises(self) -> None:
        """Non-finite values in inputs must raise."""
        with pytest.raises(PreconditionError):
            compute_ztcf_accelerations(
                mass_matrix=np.eye(2),
                coriolis_vector=np.array([np.inf, 0.0]),
                gravity_vector=np.zeros(2),
                friction_vector=np.zeros(2),
            )


# ============================================================================
# compute_ztcf_forces
# ============================================================================


class TestComputeZTCFForces:
    """Tests for full ZTCF force computation pipeline."""

    def test_returns_ztcf_result(self) -> None:
        result = compute_ztcf_forces(
            mass_matrix=np.eye(2),
            coriolis_vector=np.zeros(2),
            gravity_vector=np.zeros(2),
            friction_vector=np.zeros(2),
            joint_positions=np.array([[0.0, 0.0], [1.0, 0.0]]),
            segment_masses=np.array([1.0, 1.0]),
            segment_lengths=np.array([1.0, 1.0]),
            gravity_acceleration=9.81,
        )
        assert isinstance(result, ZTCFResult)
        assert result.n_joints == 2

    def test_zero_gravity_zero_velocity_gives_zero_forces(self) -> None:
        """No gravity, no velocity → zero ZTCF forces."""
        result = compute_ztcf_forces(
            mass_matrix=np.eye(2),
            coriolis_vector=np.zeros(2),
            gravity_vector=np.zeros(2),
            friction_vector=np.zeros(2),
            joint_positions=np.array([[0.0, 0.0], [1.0, 0.0]]),
            segment_masses=np.array([1.0, 1.0]),
            segment_lengths=np.array([1.0, 1.0]),
            gravity_acceleration=0.0,
        )
        np.testing.assert_allclose(result.joint_forces, 0.0, atol=1e-10)

    def test_forces_are_finite(self) -> None:
        """Postcondition: all returned forces must be finite."""
        result = compute_ztcf_forces(
            mass_matrix=np.eye(2),
            coriolis_vector=np.array([0.5, 0.3]),
            gravity_vector=np.array([9.81, 4.9]),
            friction_vector=np.array([0.1, 0.1]),
            joint_positions=np.array([[0.0, 0.0], [1.0, 0.0]]),
            segment_masses=np.array([5.0, 3.0]),
            segment_lengths=np.array([0.4, 0.35]),
            gravity_acceleration=9.81,
        )
        assert np.all(np.isfinite(result.joint_forces))
        assert np.all(np.isfinite(result.joint_accelerations))


# ============================================================================
# compute_force_delta
# ============================================================================


class TestComputeForceDelta:
    """Tests for delta (total - ZTCF) force computation."""

    def test_delta_is_total_minus_ztcf(self) -> None:
        total = np.array([[10.0, 20.0], [30.0, 40.0]])
        ztcf = np.array([[3.0, 5.0], [10.0, 15.0]])
        delta = compute_force_delta(total_forces=total, ztcf_forces=ztcf)
        expected = np.array([[7.0, 15.0], [20.0, 25.0]])
        np.testing.assert_allclose(delta, expected)

    def test_zero_ztcf_gives_total_as_delta(self) -> None:
        total = np.array([[5.0, 6.0]])
        ztcf = np.zeros_like(total)
        delta = compute_force_delta(total_forces=total, ztcf_forces=ztcf)
        np.testing.assert_allclose(delta, total)

    def test_equal_forces_give_zero_delta(self) -> None:
        forces = np.array([[1.0, 2.0], [3.0, 4.0]])
        delta = compute_force_delta(total_forces=forces, ztcf_forces=forces)
        np.testing.assert_allclose(delta, 0.0, atol=1e-15)

    def test_mismatched_shapes_raises(self) -> None:
        with pytest.raises(PreconditionError):
            compute_force_delta(
                total_forces=np.array([[1.0, 2.0]]),
                ztcf_forces=np.array([[1.0, 2.0], [3.0, 4.0]]),
            )

    def test_non_finite_input_raises(self) -> None:
        with pytest.raises(PreconditionError):
            compute_force_delta(
                total_forces=np.array([[np.nan, 2.0]]),
                ztcf_forces=np.array([[1.0, 2.0]]),
            )
