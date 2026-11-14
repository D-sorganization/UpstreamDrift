"""Tests for Zero-Torque Counterfactual (ZTCF) force computation.

Covers:
- ZTCFResult data structure construction and validation
- compute_ztcf_forces with known physics inputs
- compute_force_delta between total and ZTCF forces
- Precondition/postcondition enforcement (DbC)
- Edge cases: zero velocity, zero gravity, single joint
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from src.shared.python.biomechanics.ztcf import (
    ZTCFReaction,
    ZTCFResult,
    ZTCFSegmentKinematics,
    compute_force_delta,
    compute_ztcf_accelerations,
    compute_ztcf_forces,
    compute_ztcf_joint_reactions,
    pendulum_tangential_scalar,
)
from src.shared.python.core.contracts.exceptions import (
    PreconditionError,
)

pytestmark = pytest.mark.unit

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
            kinematics=ZTCFSegmentKinematics(
                joint_positions=np.array([[0.0, 0.0], [1.0, 0.0]]),
                segment_masses=np.array([1.0, 1.0]),
                segment_lengths=np.array([1.0, 1.0]),
            ),
            gravity_acceleration=9.80665,
            allow_simplified=True,
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
            kinematics=ZTCFSegmentKinematics(
                joint_positions=np.array([[0.0, 0.0], [1.0, 0.0]]),
                segment_masses=np.array([1.0, 1.0]),
                segment_lengths=np.array([1.0, 1.0]),
            ),
            gravity_acceleration=0.0,
            allow_simplified=True,
        )
        np.testing.assert_allclose(result.joint_forces, 0.0, atol=1e-10)

    def test_forces_are_finite(self) -> None:
        """Postcondition: all returned forces must be finite."""
        result = compute_ztcf_forces(
            mass_matrix=np.eye(2),
            coriolis_vector=np.array([0.5, 0.3]),
            gravity_vector=np.array([9.80665, 4.9]),
            friction_vector=np.array([0.1, 0.1]),
            kinematics=ZTCFSegmentKinematics(
                joint_positions=np.array([[0.0, 0.0], [1.0, 0.0]]),
                segment_masses=np.array([5.0, 3.0]),
                segment_lengths=np.array([0.4, 0.35]),
            ),
            gravity_acceleration=9.80665,
            allow_simplified=True,
        )
        assert np.all(np.isfinite(result.joint_forces))
        assert np.all(np.isfinite(result.joint_accelerations))

    @pytest.mark.parametrize("theta_deg", [15.0, 30.0, 45.0, 60.0, 90.0])
    def test_matches_analytic_single_pendulum(self, theta_deg: float) -> None:
        """ZTCF tangential force equals the analytic pendulum m*g*sin(theta).

        Validation for #7054. A single revolute pendulum (mass m, length L,
        angle theta from vertical) under zero applied torque obeys
        ``m*L^2 * qddot + m*g*L*sin(theta) = 0`` → ``qddot = -(g/L)*sin(theta)``.
        The helper maps this to a tangential linear force
        ``F_t = m * qddot * L = -m*g*sin(theta)``. This pins the magnitude
        against the closed form, replacing the prior finiteness-only check.
        """
        m = 2.5  # [kg]
        length = 0.9  # [m]
        g = 9.80665  # [m/s^2]
        theta = np.radians(theta_deg)

        # 1-DOF generalized dynamics for the pendulum about the pivot.
        mass_matrix = np.array([[m * length**2]])
        gravity_vector = np.array([m * g * length * np.sin(theta)])

        result = compute_ztcf_forces(
            mass_matrix=mass_matrix,
            coriolis_vector=np.zeros(1),
            gravity_vector=gravity_vector,
            friction_vector=np.zeros(1),
            kinematics=ZTCFSegmentKinematics(
                joint_positions=np.array(
                    [[length * np.sin(theta), -length * np.cos(theta)]]
                ),
                segment_masses=np.array([m]),
                segment_lengths=np.array([length]),
            ),
            gravity_acceleration=g,
        )

        # qddot must equal -(g/L) sin(theta).
        assert result.joint_accelerations[0] == pytest.approx(
            -(g / length) * np.sin(theta), rel=1e-9
        )

        # Tangential (x) force component == -m*g*sin(theta).
        f_tangential = result.joint_forces[0, 0]
        assert f_tangential == pytest.approx(-m * g * np.sin(theta), rel=1e-9)

        # Normal (y) component is the gravity reaction +m*g (g_vec = -g in y).
        assert result.joint_forces[0, 1] == pytest.approx(m * g, rel=1e-9)

    def test_multibody_rejected_without_allow_simplified(self) -> None:
        """Part of #14944: multi-segment calls must opt into the legacy placeholder."""
        with pytest.raises(PreconditionError, match="single-pendulum tangential"):
            compute_ztcf_forces(
                mass_matrix=np.eye(2),
                coriolis_vector=np.zeros(2),
                gravity_vector=np.zeros(2),
                friction_vector=np.zeros(2),
                kinematics=ZTCFSegmentKinematics(
                    joint_positions=np.array([[0.0, 0.0], [1.0, 0.0]]),
                    segment_masses=np.array([1.0, 1.0]),
                    segment_lengths=np.array([1.0, 0.8]),
                ),
                gravity_acceleration=9.80665,
            )

    def test_double_pendulum_placeholder_ignores_distal_inertia_on_proximal(
        self,
    ) -> None:
        """Part of #14944: only distal qddot — placeholder leaves proximal tangential force at 0."""
        m1, m2 = 1.0, 1.0
        l1, l2 = 1.0, 0.8
        mass_matrix = np.diag([m1 * l1**2, m2 * l2**2])
        qddot = np.array([0.0, 2.5])
        gravity_vector = -(mass_matrix @ qddot)
        result = compute_ztcf_forces(
            mass_matrix=mass_matrix,
            coriolis_vector=np.zeros(2),
            gravity_vector=gravity_vector,
            friction_vector=np.zeros(2),
            kinematics=ZTCFSegmentKinematics(
                joint_positions=np.array([[0.0, 0.0], [l1, 0.0]]),
                segment_masses=np.array([m1, m2]),
                segment_lengths=np.array([l1, l2]),
            ),
            gravity_acceleration=9.80665,
            allow_simplified=True,
        )
        np.testing.assert_allclose(result.joint_accelerations, qddot, rtol=1e-9)
        assert result.joint_forces[0, 0] == pytest.approx(0.0, abs=1e-12)
        assert result.joint_forces[1, 0] == pytest.approx(m2 * qddot[1] * l2, rel=1e-9)


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


class TestZTCFReactionContract:
    def test_reaction_defaults_and_properties(self) -> None:
        w, q = np.array([[10.0, 20.0, 5.0], [30.0, 40.0, 7.0]]), np.array([1.5, -2.5])
        r = ZTCFReaction(wrench=w, qddot_ztcf=q)
        assert r.frame == "world" and r.reference_point == "joint_origin"
        assert (
            r.sign_owner == "parent_on_child" and r.definition_version == "ztcf-rnea-v1"
        )
        assert (
            np.array_equal(r.wrench, w)
            and np.array_equal(r.qddot_ztcf, q)
            and np.array_equal(r.forces, w[:, :2])
            and np.array_equal(r.moments, w[:, 2])
            and np.array_equal(r.reaction_at_joint(0), w[0])
        )


def _rxn(q: Any, qdot: Any, m: Any, length: Any, **kw: Any) -> ZTCFReaction:
    m_a, l_a = np.atleast_1d(m), np.atleast_1d(length)
    return compute_ztcf_joint_reactions(
        joint_angles=q,
        joint_velocities=qdot,
        segment_masses=m_a,
        segment_lengths=l_a,
        com_distances=kw.pop("rc", l_a / 2.0),
        inertias_about_com=kw.pop("ic", None),
        gravity_acceleration=kw.pop("g", 9.80665),
    )


class TestComputeZTCFJointReactionsOracles:
    def test_oracles_1_to_4_point_mass_and_uniform_rods(self) -> None:
        m, length, g = 2.5, 0.9, 9.80665
        rx_h = _rxn([np.pi / 2], [0.0], m, length, rc=[length])
        rx_v = _rxn([0.0], [3.2], m, length, rc=[length])
        np.testing.assert_allclose(rx_h.wrench[0, :2], [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(
            rx_v.wrench[0, :2], [0.0, m * g + m * length * 3.2**2], atol=1e-12
        )
        with pytest.deprecated_call():
            legacy = compute_ztcf_forces(
                mass_matrix=np.array([[m * length**2]]),
                gravity_vector=np.array([m * g * length]),
                kinematics=ZTCFSegmentKinematics(
                    np.array([[length, 0.0]]), np.array([m]), np.array([length])
                ),
                gravity_acceleration=g,
            )
        np.testing.assert_allclose(legacy.joint_forces[0], [-m * g, m * g], rtol=1e-9)
        r2 = _rxn([np.pi / 2], [0.0], 3.0, 1.2)
        assert r2.qddot_ztcf[0] == pytest.approx(-1.5 * g / 1.2, rel=1e-9)
        np.testing.assert_allclose(r2.wrench[0, :2], [0.0, 3.0 * g / 4.0], atol=1e-12)
        np.testing.assert_allclose(
            _rxn([0.0], [0.0], 4.0, 0.8).wrench[0], [0.0, 4.0 * g, 0.0], atol=1e-12
        )
        r4 = _rxn([0.0, 0.0], [0.0, 0.0], [2.0, 1.5], [0.8, 0.6])
        np.testing.assert_allclose(
            r4.forces, [[0.0, 3.5 * g], [0.0, 1.5 * g]], atol=1e-12
        )

    def test_oracle_6_global_balance_arbitrary_double_pendulum_state(self) -> None:
        m1, m2, l1, l2, g = 2.5, 1.8, 0.9, 0.7, 9.80665
        q, qdot = np.array([0.5, -0.8]), np.array([1.2, -2.4])
        rc = np.array([l1 * 0.45, l2 * 0.42])
        ic = [(1.0 / 12.0) * m1 * l1**2, (1.0 / 12.0) * m2 * l2**2]
        rx = _rxn(q, qdot, [m1, m2], [l1, l2], rc=rc, ic=ic, g=g)
        phi, w, al = np.cumsum(q), np.cumsum(qdot), np.cumsum(rx.qddot_ztcf)
        e, t = (
            np.column_stack([np.sin(phi), -np.cos(phi)]),
            np.column_stack([np.cos(phi), np.sin(phi)]),
        )
        term = al[:, None] * t - (w[:, None] ** 2) * e
        a0, a1 = rc[0] * term[0], l1 * term[0] + rc[1] * term[1]
        f_exp = m1 * (a0 - [0.0, -g]) + m2 * (a1 - [0.0, -g])
        np.testing.assert_allclose(rx.wrench[0, :2], f_exp, atol=1e-10)


class TestZTCFLegacyCompatibility:
    def test_legacy_features(self) -> None:
        assert pendulum_tangential_scalar(3.0, 2.5, 1.2) == pytest.approx(9.0)
        pos, m_a, l_a = np.array([[0.0, 1.0]]), np.array([2.0]), np.array([1.0])
        with pytest.deprecated_call():
            res = compute_ztcf_forces(
                np.array([2.5]),
                np.array([[2.0]]),
                joint_positions=pos,
                segment_masses=m_a,
                segment_lengths=l_a,
            )
        assert res.n_joints == 1 and res.joint_forces[0, 0] == pytest.approx(5.0)
        with pytest.raises(ValueError):
            compute_ztcf_forces(
                np.ones(2),
                np.eye(2),
                joint_positions=np.ones((2, 2)),
                segment_masses=np.ones(2),
                segment_lengths=np.ones(2),
            )
        with pytest.raises(PreconditionError, match="singular or ill-conditioned"):
            compute_ztcf_accelerations(
                mass_matrix=np.diag([1.0, 1e-15]),
                coriolis_vector=np.zeros(2),
                gravity_vector=np.zeros(2),
            )
