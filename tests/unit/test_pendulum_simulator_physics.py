"""Unit tests for src/shared/python/pendulum_simulator/physics.py.

Tests cover PendulumParams, JointLimits, TorqueClamp, JointLimitsNDOF and all
public physics functions (mass_matrix, coriolis_vector, gravity_vector,
friction_torque_vector, joint_limit_torque, clamp_torque, equations_of_motion,
forward_kinematics, joint_velocities, kinetic_energy, potential_energy,
total_energy).

All tests are headless-safe and require only numpy (no Rust extension).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def basic_params():
    """Standard double-pendulum parameters for golf swing model."""
    from src.shared.python.pendulum_simulator.physics import PendulumParams

    return PendulumParams(m1=5.0, m2=0.3, L1=0.65, L2=1.1, mClub=0.2)


@pytest.fixture
def rest_state():
    """State vector at rest in equilibrium (hanging straight down)."""
    return np.array([0.0, 0.0, 0.0, 0.0])


@pytest.fixture
def zero_torque():
    """Zero torque function."""
    return lambda t: (0.0, 0.0)


# ---------------------------------------------------------------------------
# PendulumParams
# ---------------------------------------------------------------------------


class TestPendulumParams:
    """Tests for PendulumParams dataclass validation."""

    def test_valid_params_instantiate(self, basic_params) -> None:
        """PendulumParams with positive masses and lengths instantiates correctly."""

        p = basic_params
        assert p.m1 > 0
        assert p.m2 > 0
        assert p.L1 > 0
        assert p.L2 > 0

    def test_pendulum_simulator_physics_default_fields(self) -> None:
        """Default optional fields are non-negative."""
        from src.shared.python.pendulum_simulator.physics import PendulumParams

        p = PendulumParams(m1=1.0, m2=1.0, L1=1.0, L2=1.0)
        assert p.mClub == 0.0
        assert p.b1 == 0.0
        assert p.b2 == 0.0
        assert p.mu1 == 0.0
        assert p.mu2 == 0.0

    def test_negative_m1_raises(self) -> None:
        """Negative m1 raises AssertionError."""
        from src.shared.python.pendulum_simulator.physics import PendulumParams

        with pytest.raises(AssertionError):
            PendulumParams(m1=-1.0, m2=1.0, L1=1.0, L2=1.0)

    def test_negative_m2_raises(self) -> None:
        """Negative m2 raises AssertionError."""
        from src.shared.python.pendulum_simulator.physics import PendulumParams

        with pytest.raises(AssertionError):
            PendulumParams(m1=1.0, m2=-0.1, L1=1.0, L2=1.0)

    def test_negative_L1_raises(self) -> None:
        """Negative L1 raises AssertionError."""
        from src.shared.python.pendulum_simulator.physics import PendulumParams

        with pytest.raises(AssertionError):
            PendulumParams(m1=1.0, m2=1.0, L1=-0.5, L2=1.0)

    def test_negative_L2_raises(self) -> None:
        """Negative L2 raises AssertionError."""
        from src.shared.python.pendulum_simulator.physics import PendulumParams

        with pytest.raises(AssertionError):
            PendulumParams(m1=1.0, m2=1.0, L1=1.0, L2=-1.0)

    def test_negative_mclub_raises(self) -> None:
        """Negative mClub raises AssertionError."""
        from src.shared.python.pendulum_simulator.physics import PendulumParams

        with pytest.raises(AssertionError):
            PendulumParams(m1=1.0, m2=1.0, L1=1.0, L2=1.0, mClub=-0.1)

    def test_zero_mclub_allowed(self) -> None:
        """Zero mClub is valid (point mass at wrist)."""
        from src.shared.python.pendulum_simulator.physics import PendulumParams

        p = PendulumParams(m1=1.0, m2=1.0, L1=1.0, L2=1.0, mClub=0.0)
        assert p.mClub == 0.0

    def test_with_damping(self) -> None:
        """Positive damping and friction coefficients are accepted."""
        from src.shared.python.pendulum_simulator.physics import PendulumParams

        p = PendulumParams(
            m1=5.0, m2=0.3, L1=0.65, L2=1.1, b1=0.1, b2=0.05, mu1=0.2, mu2=0.1
        )
        assert p.b1 == 0.1
        assert p.mu1 == 0.2


# ---------------------------------------------------------------------------
# JointLimits
# ---------------------------------------------------------------------------


class TestJointLimits:
    """Tests for JointLimits dataclass."""

    def test_default_instantiation(self) -> None:
        """JointLimits with defaults instantiates without error."""
        from src.shared.python.pendulum_simulator.physics import JointLimits

        limits = JointLimits()
        assert limits.phi_min < limits.phi_max
        assert limits.theta1_min < limits.theta1_max
        assert limits.stiffness > 0
        assert limits.damping >= 0

    def test_custom_limits(self) -> None:
        """Custom angle limits are stored correctly."""
        from src.shared.python.pendulum_simulator.physics import JointLimits

        limits = JointLimits(phi_min=-0.5, phi_max=0.5, stiffness=1000.0, damping=10.0)
        assert limits.phi_min == -0.5
        assert limits.phi_max == 0.5

    def test_invalid_phi_range_raises(self) -> None:
        """phi_min >= phi_max raises AssertionError."""
        from src.shared.python.pendulum_simulator.physics import JointLimits

        with pytest.raises(AssertionError):
            JointLimits(phi_min=1.0, phi_max=-1.0)

    def test_zero_stiffness_raises(self) -> None:
        """Zero stiffness raises AssertionError."""
        from src.shared.python.pendulum_simulator.physics import JointLimits

        with pytest.raises(AssertionError):
            JointLimits(stiffness=0.0)


# ---------------------------------------------------------------------------
# TorqueClamp
# ---------------------------------------------------------------------------


class TestTorqueClamp:
    """Tests for TorqueClamp dataclass."""

    def test_default_is_infinite(self) -> None:
        """Default TorqueClamp has infinite limits (no clamping)."""
        from src.shared.python.pendulum_simulator.physics import TorqueClamp

        clamp = TorqueClamp()
        assert math.isinf(clamp.max_torque1)
        assert math.isinf(clamp.max_torque2)

    def test_finite_limits(self) -> None:
        """Finite torque limits are stored as positive values."""
        from src.shared.python.pendulum_simulator.physics import TorqueClamp

        clamp = TorqueClamp(max_torque1=50.0, max_torque2=30.0)
        assert clamp.max_torque1 == 50.0
        assert clamp.max_torque2 == 30.0

    def test_negative_input_converted_to_positive(self) -> None:
        """Negative inputs are accepted and converted via abs()."""
        from src.shared.python.pendulum_simulator.physics import TorqueClamp

        clamp = TorqueClamp(max_torque1=-50.0, max_torque2=-30.0)
        assert clamp.max_torque1 == 50.0
        assert clamp.max_torque2 == 30.0


# ---------------------------------------------------------------------------
# JointLimitsNDOF
# ---------------------------------------------------------------------------


class TestJointLimitsNDOF:
    """Tests for JointLimitsNDOF dataclass."""

    def test_valid_ndof_limits(self) -> None:
        """JointLimitsNDOF with valid 4-DOF limits instantiates correctly."""
        from src.shared.python.pendulum_simulator.physics import JointLimitsNDOF

        lims = JointLimitsNDOF(
            angle_min=np.array([-1.0, -1.0, -1.0, -1.0]),
            angle_max=np.array([1.0, 1.0, 1.0, 1.0]),
        )
        assert lims.angle_min.shape == (4,)

    def test_pendulum_simulator_physics_shape_mismatch_raises(self) -> None:
        """Mismatched min/max shapes raise AssertionError."""
        from src.shared.python.pendulum_simulator.physics import JointLimitsNDOF

        with pytest.raises(AssertionError):
            JointLimitsNDOF(
                angle_min=np.array([-1.0, -1.0]),
                angle_max=np.array([1.0, 1.0, 1.0]),
            )

    def test_min_gte_max_raises(self) -> None:
        """angle_min >= angle_max raises AssertionError."""
        from src.shared.python.pendulum_simulator.physics import JointLimitsNDOF

        with pytest.raises(AssertionError):
            JointLimitsNDOF(
                angle_min=np.array([1.0, 1.0]),
                angle_max=np.array([-1.0, -1.0]),
            )


# ---------------------------------------------------------------------------
# mass_matrix
# ---------------------------------------------------------------------------


class TestMassMatrix:
    """Tests for mass_matrix function."""

    def test_returns_2x2_array(self, basic_params) -> None:
        """mass_matrix returns a (2, 2) ndarray."""
        from src.shared.python.pendulum_simulator.physics import mass_matrix

        M = mass_matrix(0.0, basic_params)
        assert M.shape == (2, 2)

    def test_pendulum_simulator_physics_symmetric(self, basic_params) -> None:
        """Mass matrix is symmetric: M[0,1] == M[1,0]."""
        from src.shared.python.pendulum_simulator.physics import mass_matrix

        M = mass_matrix(0.3, basic_params)
        assert np.isclose(M[0, 1], M[1, 0])

    def test_pendulum_simulator_physics_positive_definite(self, basic_params) -> None:
        """Mass matrix has all positive eigenvalues."""
        from src.shared.python.pendulum_simulator.physics import mass_matrix

        M = mass_matrix(0.0, basic_params)
        eigvals = np.linalg.eigvalsh(M)
        assert np.all(eigvals > 0)

    def test_diagonal_decreasing(self, basic_params) -> None:
        """M11 > M22 for typical golf parameters (arms dominate)."""
        from src.shared.python.pendulum_simulator.physics import mass_matrix

        M = mass_matrix(0.0, basic_params)
        assert M[0, 0] > M[1, 1]

    def test_inf_phi_raises(self, basic_params) -> None:
        """Infinite phi raises AssertionError."""
        from src.shared.python.pendulum_simulator.physics import mass_matrix

        with pytest.raises(AssertionError):
            mass_matrix(float("inf"), basic_params)

    def test_components_match(self, basic_params) -> None:
        """mass_matrix_components M_full matches mass_matrix output."""
        from src.shared.python.pendulum_simulator.physics import (
            mass_matrix,
            mass_matrix_components,
        )

        phi = 0.2
        M = mass_matrix(phi, basic_params)
        comps = mass_matrix_components(phi, basic_params)
        assert np.allclose(comps["M_full"], M)
        assert np.isclose(comps["M11"], M[0, 0])
        assert np.isclose(comps["M22"], M[1, 1])


# ---------------------------------------------------------------------------
# coriolis_vector
# ---------------------------------------------------------------------------


class TestCoriolisVector:
    """Tests for coriolis_vector function."""

    def test_zero_velocities_gives_zero(self, basic_params) -> None:
        """At zero velocities, Coriolis vector is [0, 0]."""
        from src.shared.python.pendulum_simulator.physics import coriolis_vector

        C = coriolis_vector(0.0, 0.0, 0.0, basic_params)
        assert np.allclose(C, [0.0, 0.0])

    def test_pendulum_simulator_physics_returns_shape_2(self, basic_params) -> None:
        """coriolis_vector returns a shape (2,) array."""
        from src.shared.python.pendulum_simulator.physics import coriolis_vector

        C = coriolis_vector(0.1, 1.0, -0.5, basic_params)
        assert C.shape == (2,)

    def test_all_finite(self, basic_params) -> None:
        """All elements of coriolis_vector are finite."""
        from src.shared.python.pendulum_simulator.physics import coriolis_vector

        C = coriolis_vector(0.3, 2.0, -1.5, basic_params)
        assert np.all(np.isfinite(C))


# ---------------------------------------------------------------------------
# gravity_vector
# ---------------------------------------------------------------------------


class TestGravityVector:
    """Tests for gravity_vector function."""

    def test_equilibrium_gives_zero(self, basic_params) -> None:
        """At theta1=0, phi=0 (straight down), gravity generates torques (not zero)."""
        from src.shared.python.pendulum_simulator.physics import gravity_vector

        # At hanging equilibrium both segments point down (angles=0)
        # G1 = (m1+me)*g*L1*sin(0) + me*g*L2*sin(0) = 0
        # G2 = me*g*L2*sin(0) = 0
        G = gravity_vector(0.0, 0.0, basic_params)
        assert np.allclose(G, [0.0, 0.0])

    def test_pendulum_simulator_physics_returns_shape_2(self, basic_params) -> None:
        """gravity_vector returns shape (2,)."""
        from src.shared.python.pendulum_simulator.physics import gravity_vector

        G = gravity_vector(0.5, 0.1, basic_params)
        assert G.shape == (2,)

    def test_nonzero_at_nonequilibrium(self, basic_params) -> None:
        """At nonzero angle, gravity vector is nonzero."""
        from src.shared.python.pendulum_simulator.physics import gravity_vector

        G = gravity_vector(0.5, 0.0, basic_params)
        assert not np.allclose(G, [0.0, 0.0])

    def test_all_finite(self, basic_params) -> None:
        """gravity_vector returns finite values."""
        from src.shared.python.pendulum_simulator.physics import gravity_vector

        G = gravity_vector(1.0, 0.5, basic_params)
        assert np.all(np.isfinite(G))


# ---------------------------------------------------------------------------
# friction_torque_vector
# ---------------------------------------------------------------------------


class TestFrictionTorqueVector:
    """Tests for friction_torque_vector."""

    def test_zero_with_no_friction_params(self, basic_params) -> None:
        """Zero friction/damping params → zero friction torques."""
        from src.shared.python.pendulum_simulator.physics import friction_torque_vector

        # basic_params has b1=b2=mu1=mu2=0
        F = friction_torque_vector(1.0, -0.5, basic_params)
        assert np.allclose(F, [0.0, 0.0])

    def test_viscous_damping_opposes_motion(self) -> None:
        """With viscous damping, friction opposes positive velocity."""
        from src.shared.python.pendulum_simulator.physics import (
            PendulumParams,
            friction_torque_vector,
        )

        p = PendulumParams(m1=1.0, m2=1.0, L1=1.0, L2=1.0, b1=2.0, b2=1.0)
        F = friction_torque_vector(1.0, 1.0, p)
        # tau_f1 = -b1*dtheta1 = -2.0, tau_f2 = -b2*dphi = -1.0
        assert np.isclose(F[0], -2.0)
        assert np.isclose(F[1], -1.0)

    def test_pendulum_simulator_physics_returns_shape_2(self, basic_params) -> None:
        """Returns shape (2,)."""
        from src.shared.python.pendulum_simulator.physics import friction_torque_vector

        F = friction_torque_vector(0.5, -0.3, basic_params)
        assert F.shape == (2,)


# ---------------------------------------------------------------------------
# joint_limit_torque
# ---------------------------------------------------------------------------


class TestJointLimitTorque:
    """Tests for joint_limit_torque."""

    def test_within_limits_gives_zero(self, basic_params) -> None:
        """When angle is within limits, penalty torque is zero."""
        from src.shared.python.pendulum_simulator.physics import (
            JointLimits,
            joint_limit_torque,
        )

        limits = JointLimits(phi_min=-1.0, phi_max=1.0)
        tau = joint_limit_torque(0.0, 0.0, limits)
        assert np.allclose(tau, [0.0, 0.0])

    def test_pendulum_simulator_physics_returns_shape_2(self) -> None:
        """Returns shape (2,)."""
        from src.shared.python.pendulum_simulator.physics import (
            JointLimits,
            joint_limit_torque,
        )

        limits = JointLimits()
        tau = joint_limit_torque(0.0, 0.0, limits)
        assert tau.shape == (2,)

    def test_exceeding_phi_max_gives_negative_penalty(self) -> None:
        """Exceeding phi_max gives negative penalty (restoring force)."""
        from src.shared.python.pendulum_simulator.physics import (
            JointLimits,
            joint_limit_torque,
        )

        limits = JointLimits(phi_min=-0.5, phi_max=0.5)
        # phi = 0.6 exceeds phi_max=0.5
        tau = joint_limit_torque(0.6, 0.1, limits)
        assert tau[1] < 0.0  # restoring (negative)

    def test_below_phi_min_gives_positive_penalty(self) -> None:
        """Going below phi_min gives positive penalty (restoring force)."""
        from src.shared.python.pendulum_simulator.physics import (
            JointLimits,
            joint_limit_torque,
        )

        limits = JointLimits(phi_min=-0.5, phi_max=0.5)
        # phi = -0.6 below phi_min=-0.5
        tau = joint_limit_torque(-0.6, -0.1, limits)
        assert tau[1] > 0.0  # restoring (positive)


# ---------------------------------------------------------------------------
# clamp_torque
# ---------------------------------------------------------------------------


class TestClampTorque:
    """Tests for clamp_torque."""

    def test_values_within_limits_unchanged(self) -> None:
        """Torques within clamp limits are unchanged."""
        from src.shared.python.pendulum_simulator.physics import (
            TorqueClamp,
            clamp_torque,
        )

        tau = np.array([10.0, -5.0])
        clamp = TorqueClamp(max_torque1=50.0, max_torque2=50.0)
        result = clamp_torque(tau, clamp)
        assert np.allclose(result, tau)

    def test_values_exceeding_limits_clamped(self) -> None:
        """Torques outside clamp limits are clipped."""
        from src.shared.python.pendulum_simulator.physics import (
            TorqueClamp,
            clamp_torque,
        )

        tau = np.array([100.0, -100.0])
        clamp = TorqueClamp(max_torque1=50.0, max_torque2=50.0)
        result = clamp_torque(tau, clamp)
        assert result[0] == pytest.approx(50.0)
        assert result[1] == pytest.approx(-50.0)

    def test_infinite_clamp_passes_through(self) -> None:
        """Infinite TorqueClamp passes any torque unchanged."""
        from src.shared.python.pendulum_simulator.physics import (
            TorqueClamp,
            clamp_torque,
        )

        tau = np.array([1e9, -1e9])
        clamp = TorqueClamp()
        result = clamp_torque(tau, clamp)
        assert np.allclose(result, tau)


# ---------------------------------------------------------------------------
# forward_kinematics
# ---------------------------------------------------------------------------


class TestForwardKinematics:
    """Tests for forward_kinematics."""

    def test_hanging_position_at_zero_angles(self, basic_params) -> None:
        """At theta1=0, phi=0, wrist is directly below shoulder."""
        from src.shared.python.pendulum_simulator.physics import forward_kinematics

        fk = forward_kinematics(0.0, 0.0, basic_params)
        assert "shoulder" in fk
        assert "wrist" in fk
        assert "tip" in fk
        wx, wy = fk["wrist"]
        # At theta1=0: wx = L1*sin(0)=0, wy = -L1*cos(0)=-L1
        assert abs(wx) < 1e-9
        assert abs(wy + basic_params.L1) < 1e-9

    def test_wrist_distance_equals_L1(self, basic_params) -> None:
        """Distance from shoulder to wrist equals L1."""
        from src.shared.python.pendulum_simulator.physics import forward_kinematics

        fk = forward_kinematics(0.3, 0.1, basic_params)
        wx, wy = fk["wrist"]
        dist = math.sqrt(wx**2 + wy**2)
        assert abs(dist - basic_params.L1) < 1e-9

    def test_tip_distance_from_wrist_equals_L2(self, basic_params) -> None:
        """Distance from wrist to tip equals L2."""
        from src.shared.python.pendulum_simulator.physics import forward_kinematics

        fk = forward_kinematics(0.3, 0.1, basic_params)
        wx, wy = fk["wrist"]
        tx, ty = fk["tip"]
        dist = math.sqrt((tx - wx) ** 2 + (ty - wy) ** 2)
        assert abs(dist - basic_params.L2) < 1e-9

    def test_pendulum_simulator_physics_returns_dict(self, basic_params) -> None:
        """forward_kinematics returns a dict."""
        from src.shared.python.pendulum_simulator.physics import forward_kinematics

        result = forward_kinematics(0.0, 0.0, basic_params)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# joint_velocities
# ---------------------------------------------------------------------------


class TestJointVelocities:
    """Tests for joint_velocities."""

    def test_zero_velocity_at_rest(self, basic_params, rest_state) -> None:
        """At rest (zero velocities), wrist and tip speeds are zero."""
        from src.shared.python.pendulum_simulator.physics import joint_velocities

        vels = joint_velocities(rest_state, basic_params)
        assert abs(vels["wrist_speed"]) < 1e-12
        assert abs(vels["tip_speed"]) < 1e-12

    def test_returns_expected_keys(self, basic_params, rest_state) -> None:
        """joint_velocities returns dict with expected keys."""
        from src.shared.python.pendulum_simulator.physics import joint_velocities

        vels = joint_velocities(rest_state, basic_params)
        assert "wrist_speed" in vels
        assert "tip_speed" in vels
        assert "wrist_vel" in vels
        assert "tip_vel" in vels

    def test_nonzero_speed_with_nonzero_velocity(self, basic_params) -> None:
        """Nonzero angular velocity produces nonzero end-effector speed."""
        from src.shared.python.pendulum_simulator.physics import joint_velocities

        state = np.array([0.0, 0.0, 2.0, 0.0])  # dtheta1=2 rad/s
        vels = joint_velocities(state, basic_params)
        assert vels["wrist_speed"] > 0.0
        assert vels["tip_speed"] > 0.0


# ---------------------------------------------------------------------------
# equations_of_motion
# ---------------------------------------------------------------------------


class TestEquationsOfMotion:
    """Tests for equations_of_motion."""

    def test_returns_shape_4(self, basic_params, rest_state, zero_torque) -> None:
        """equations_of_motion returns state derivative with shape (4,)."""
        from src.shared.python.pendulum_simulator.physics import equations_of_motion

        state_dot = equations_of_motion(rest_state, 0.0, basic_params, zero_torque)
        assert state_dot.shape == (4,)

    def test_all_finite(self, basic_params, rest_state, zero_torque) -> None:
        """State derivative is finite."""
        from src.shared.python.pendulum_simulator.physics import equations_of_motion

        state_dot = equations_of_motion(rest_state, 0.0, basic_params, zero_torque)
        assert np.all(np.isfinite(state_dot))

    def test_at_rest_zero_initial_acceleration(
        self, basic_params, rest_state, zero_torque
    ) -> None:
        """At equilibrium (theta1=phi=0) with zero torque, qddot=[0,0]."""
        from src.shared.python.pendulum_simulator.physics import equations_of_motion

        state_dot = equations_of_motion(rest_state, 0.0, basic_params, zero_torque)
        # state_dot[0]=dtheta1=0, state_dot[1]=dphi=0 (from rest state)
        assert abs(state_dot[0]) < 1e-12
        assert abs(state_dot[1]) < 1e-12
        # qddot should also be zero at hanging equilibrium (G=0 at q=[0,0])
        assert abs(state_dot[2]) < 1e-9
        assert abs(state_dot[3]) < 1e-9

    def test_with_joint_limits(self, basic_params, rest_state, zero_torque) -> None:
        """equations_of_motion works with joint limits applied."""
        from src.shared.python.pendulum_simulator.physics import (
            JointLimits,
            equations_of_motion,
        )

        limits = JointLimits()
        state_dot = equations_of_motion(
            rest_state, 0.0, basic_params, zero_torque, limits=limits
        )
        assert state_dot.shape == (4,)

    def test_with_torque_clamp(self, basic_params, rest_state) -> None:
        """equations_of_motion works with torque clamping."""
        from src.shared.python.pendulum_simulator.physics import (
            TorqueClamp,
            equations_of_motion,
        )

        def torque(_t):
            return (200.0, 200.0)  # will be clamped

        clamp = TorqueClamp(max_torque1=100.0, max_torque2=50.0)
        state_dot = equations_of_motion(
            rest_state, 0.0, basic_params, torque, clamp=clamp
        )
        assert np.all(np.isfinite(state_dot))


# ---------------------------------------------------------------------------
# kinetic_energy, potential_energy, total_energy
# ---------------------------------------------------------------------------


class TestEnergyFunctions:
    """Tests for kinetic_energy, potential_energy, total_energy."""

    def test_kinetic_energy_zero_at_rest(self, basic_params, rest_state) -> None:
        """Kinetic energy is zero when all velocities are zero."""
        from src.shared.python.pendulum_simulator.physics import kinetic_energy

        T = kinetic_energy(rest_state, basic_params)
        assert abs(T) < 1e-12

    def test_kinetic_energy_positive_with_motion(self, basic_params) -> None:
        """Kinetic energy is positive with nonzero velocities."""
        from src.shared.python.pendulum_simulator.physics import kinetic_energy

        state = np.array([0.0, 0.0, 2.0, 1.0])
        T = kinetic_energy(state, basic_params)
        assert T > 0.0

    def test_potential_energy_at_equilibrium(self, basic_params, rest_state) -> None:
        """potential_energy returns a finite float at rest."""
        from src.shared.python.pendulum_simulator.physics import potential_energy

        V = potential_energy(rest_state, basic_params)
        assert math.isfinite(V)

    def test_potential_energy_increases_with_angle(self, basic_params) -> None:
        """Raising the pendulum (nonzero theta1) increases potential energy."""
        from src.shared.python.pendulum_simulator.physics import potential_energy

        V0 = potential_energy(np.array([0.0, 0.0, 0.0, 0.0]), basic_params)
        V1 = potential_energy(np.array([1.0, 0.0, 0.0, 0.0]), basic_params)
        # At theta1=1 rad, V > V_at_0 (pendulum raised from hanging position)
        assert V1 > V0

    def test_total_energy_equals_sum(self, basic_params) -> None:
        """total_energy == kinetic_energy + potential_energy."""
        from src.shared.python.pendulum_simulator.physics import (
            kinetic_energy,
            potential_energy,
            total_energy,
        )

        state = np.array([0.3, 0.1, 1.0, -0.5])
        E = total_energy(state, basic_params)
        T = kinetic_energy(state, basic_params)
        V = potential_energy(state, basic_params)
        assert abs(E - (T + V)) < 1e-9

    def test_total_energy_is_float(self, basic_params, rest_state) -> None:
        """total_energy returns a Python float."""
        from src.shared.python.pendulum_simulator.physics import total_energy

        E = total_energy(rest_state, basic_params)
        assert isinstance(E, float)


# ---------------------------------------------------------------------------
# joint_limit_torque_ndof
# ---------------------------------------------------------------------------


class TestJointLimitTorqueNDOF:
    """Tests for joint_limit_torque_ndof."""

    def test_within_limits_gives_zero(self) -> None:
        """Angles within limits produce zero penalty torques."""
        from src.shared.python.pendulum_simulator.physics import (
            JointLimitsNDOF,
            joint_limit_torque_ndof,
        )

        lims = JointLimitsNDOF(
            angle_min=np.array([-1.0, -1.0]),
            angle_max=np.array([1.0, 1.0]),
        )
        tau = joint_limit_torque_ndof(
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
            lims,
        )
        assert np.allclose(tau, [0.0, 0.0])

    def test_returns_correct_shape(self) -> None:
        """Returns shape matching the number of DOFs."""
        from src.shared.python.pendulum_simulator.physics import (
            JointLimitsNDOF,
            joint_limit_torque_ndof,
        )

        n = 6
        lims = JointLimitsNDOF(
            angle_min=np.full(n, -1.0),
            angle_max=np.full(n, 1.0),
        )
        tau = joint_limit_torque_ndof(
            np.zeros(n),
            np.zeros(n),
            lims,
        )
        assert tau.shape == (n,)
