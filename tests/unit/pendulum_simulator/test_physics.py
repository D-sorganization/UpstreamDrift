"""Tests for src.shared.python.pendulum_simulator.physics (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.pendulum_simulator.physics import (
    JointLimits,
    PendulumParams,
    TorqueClamp,
    coriolis_vector,
    forward_kinematics,
    friction_torque_vector,
    gravity_vector,
    kinetic_energy,
    mass_matrix,
    mass_matrix_components,
    potential_energy,
    total_energy,
)


def _make_params() -> PendulumParams:
    return PendulumParams(m1=5.0, m2=0.3, L1=0.65, L2=1.10)


_P = _make_params()


class TestPendulumParams:
    def test_physics_construction(self) -> None:
        p = _make_params()
        assert isinstance(p, PendulumParams)

    def test_masses_and_lengths(self) -> None:
        p = _make_params()
        assert p.m1 == pytest.approx(5.0)
        assert p.m2 == pytest.approx(0.3)
        assert pytest.approx(0.65) == p.L1
        assert pytest.approx(1.10) == p.L2

    def test_physics_default_gravity(self) -> None:
        p = _make_params()
        assert p.g == pytest.approx(9.81)

    def test_default_mclub_zero(self) -> None:
        p = _make_params()
        assert p.mClub == pytest.approx(0.0)

    def test_default_damping_zero(self) -> None:
        p = _make_params()
        assert p.b1 == pytest.approx(0.0)
        assert p.b2 == pytest.approx(0.0)

    def test_default_friction_zero(self) -> None:
        p = _make_params()
        assert p.mu1 == pytest.approx(0.0)
        assert p.mu2 == pytest.approx(0.0)

    def test_physics_custom_params(self) -> None:
        p = PendulumParams(m1=1.0, m2=0.2, L1=1.0, L2=0.5, mClub=0.1, g=9.81)
        assert p.mClub == pytest.approx(0.1)

    def test_physics_negative_mass_raises(self) -> None:
        with pytest.raises(AssertionError):
            PendulumParams(m1=-1.0, m2=0.3, L1=0.65, L2=1.10)

    def test_zero_length_raises(self) -> None:
        with pytest.raises(AssertionError):
            PendulumParams(m1=1.0, m2=0.3, L1=0.0, L2=1.10)


class TestJointLimits:
    def test_physics_construction(self) -> None:
        jl = JointLimits()
        assert isinstance(jl, JointLimits)

    def test_default_phi_range(self) -> None:
        jl = JointLimits()
        assert jl.phi_min < jl.phi_max

    def test_inverted_phi_raises(self) -> None:
        with pytest.raises(AssertionError):
            JointLimits(phi_min=1.0, phi_max=-1.0)


class TestTorqueClamp:
    def test_physics_construction(self) -> None:
        tc = TorqueClamp()
        assert tc.max_torque1 == float("inf")

    def test_negative_accepted_via_abs(self) -> None:
        tc = TorqueClamp(max_torque1=-10.0, max_torque2=-5.0)
        assert tc.max_torque1 == pytest.approx(10.0)
        assert tc.max_torque2 == pytest.approx(5.0)


class TestMassMatrix:
    def test_returns_2x2(self) -> None:
        M = mass_matrix(0.0, _P)
        assert M.shape == (2, 2)

    def test_physics_is_symmetric(self) -> None:
        M = mass_matrix(0.3, _P)
        np.testing.assert_allclose(M, M.T, atol=1e-12)

    def test_is_positive_definite(self) -> None:
        M = mass_matrix(0.0, _P)
        eigenvalues = np.linalg.eigvalsh(M)
        assert np.all(eigenvalues > 0)

    def test_physics_finite_values(self) -> None:
        M = mass_matrix(-0.5, _P)
        assert np.all(np.isfinite(M))

    def test_physics_angle_dependence(self) -> None:
        M1 = mass_matrix(0.0, _P)
        M2 = mass_matrix(1.0, _P)
        assert not np.allclose(M1, M2)


class TestMassMatrixComponents:
    def test_physics_returns_dict(self) -> None:
        result = mass_matrix_components(0.0, _P)
        assert isinstance(result, dict)

    def test_physics_has_expected_keys(self) -> None:
        result = mass_matrix_components(0.0, _P)
        assert "M11" in result
        assert "M12" in result
        assert "M21" in result
        assert "M22" in result
        assert "M_full" in result

    def test_m_full_is_2x2(self) -> None:
        result = mass_matrix_components(0.0, _P)
        assert result["M_full"].shape == (2, 2)

    def test_m12_equals_m21(self) -> None:
        result = mass_matrix_components(0.2, _P)
        assert result["M12"] == pytest.approx(result["M21"])


class TestGravityVector:
    def test_physics_returns_shape_2(self) -> None:
        G = gravity_vector(0.0, 0.0, _P)
        assert G.shape == (2,)

    def test_physics_finite_values(self) -> None:
        G = gravity_vector(0.3, -0.2, _P)
        assert np.all(np.isfinite(G))

    def test_zero_gravity(self) -> None:
        p_no_g = PendulumParams(m1=1.0, m2=0.3, L1=0.65, L2=1.10, g=0.0)
        G = gravity_vector(0.3, 0.2, p_no_g)
        np.testing.assert_allclose(G, [0.0, 0.0])

    def test_physics_angle_dependence(self) -> None:
        G1 = gravity_vector(0.0, 0.0, _P)
        G2 = gravity_vector(0.5, 0.5, _P)
        assert not np.allclose(G1, G2)


class TestCoriolisVector:
    def test_physics_returns_shape_2(self) -> None:
        C = coriolis_vector(0.0, 0.0, 0.0, _P)
        assert C.shape == (2,)

    def test_physics_finite_values(self) -> None:
        C = coriolis_vector(0.2, 0.5, -0.3, _P)
        assert np.all(np.isfinite(C))

    def test_physics_zero_velocities_zero_coriolis(self) -> None:
        C = coriolis_vector(0.0, 0.0, 0.0, _P)
        np.testing.assert_allclose(C, [0.0, 0.0], atol=1e-12)


class TestFrictionTorqueVector:
    def test_physics_returns_shape_2(self) -> None:
        tau = friction_torque_vector(0.0, 0.0, _P)
        assert tau.shape == (2,)

    def test_zero_velocity_zero_viscous(self) -> None:
        # With only viscous damping (default mu=0), zero velocity → zero torque
        tau = friction_torque_vector(0.0, 0.0, _P)
        np.testing.assert_allclose(tau, [0.0, 0.0], atol=1e-12)

    def test_viscous_opposes_motion(self) -> None:
        p = PendulumParams(m1=1.0, m2=0.3, L1=0.65, L2=1.10, b1=1.0, b2=1.0)
        tau = friction_torque_vector(1.0, 1.0, p)
        assert tau[0] < 0.0
        assert tau[1] < 0.0


class TestForwardKinematics:
    def test_physics_returns_dict(self) -> None:
        result = forward_kinematics(0.0, 0.0, _P)
        assert isinstance(result, dict)

    def test_physics_has_expected_keys(self) -> None:
        result = forward_kinematics(0.0, 0.0, _P)
        assert "shoulder" in result
        assert "wrist" in result
        assert "tip" in result

    def test_shoulder_at_origin(self) -> None:
        result = forward_kinematics(0.0, 0.0, _P)
        assert result["shoulder"] == pytest.approx((0.0, 0.0))

    def test_wrist_distance_equals_l1(self) -> None:
        result = forward_kinematics(0.5, 0.3, _P)
        wx, wy = result["wrist"]
        dist = np.hypot(wx, wy)
        assert dist == pytest.approx(_P.L1, abs=1e-9)

    def test_tip_distance_from_wrist_equals_l2(self) -> None:
        result = forward_kinematics(0.5, 0.3, _P)
        wx, wy = result["wrist"]
        tx, ty = result["tip"]
        dist = np.hypot(tx - wx, ty - wy)
        assert dist == pytest.approx(_P.L2, abs=1e-9)

    def test_physics_finite_values(self) -> None:
        result = forward_kinematics(0.3, -0.2, _P)
        for key, val in result.items():
            assert np.isfinite(val[0]) and np.isfinite(val[1]), f"Non-finite at {key}"

    def test_hanging_position(self) -> None:
        # At theta1=0, phi=0: wrist directly below shoulder, tip directly below wrist
        result = forward_kinematics(0.0, 0.0, _P)
        wx, wy = result["wrist"]
        assert wx == pytest.approx(0.0, abs=1e-9)
        assert wy == pytest.approx(-_P.L1, abs=1e-9)


class TestKineticEnergy:
    def test_zero_state_zero_ke(self) -> None:
        state = np.zeros(4)
        ke = kinetic_energy(state, _P)
        assert ke == pytest.approx(0.0)

    def test_positive_for_nonzero_velocity(self) -> None:
        state = np.array([0.0, 0.0, 1.0, 0.0])  # dtheta1 = 1.0
        ke = kinetic_energy(state, _P)
        assert ke > 0.0

    def test_finite(self) -> None:
        state = np.array([0.2, -0.1, 0.5, 0.3])
        ke = kinetic_energy(state, _P)
        assert np.isfinite(ke)


class TestPotentialEnergy:
    def test_finite(self) -> None:
        state = np.array([0.2, -0.1, 0.5, 0.3])
        pe = potential_energy(state, _P)
        assert np.isfinite(pe)

    def test_physics_angle_dependence(self) -> None:
        state1 = np.array([0.0, 0.0, 0.0, 0.0])
        state2 = np.array([1.0, 0.5, 0.0, 0.0])
        pe1 = potential_energy(state1, _P)
        pe2 = potential_energy(state2, _P)
        assert pe1 != pytest.approx(pe2)


class TestTotalEnergy:
    def test_equals_ke_plus_pe(self) -> None:
        state = np.array([0.2, -0.1, 0.5, 0.3])
        te = total_energy(state, _P)
        ke = kinetic_energy(state, _P)
        pe = potential_energy(state, _P)
        assert te == pytest.approx(ke + pe)

    def test_finite(self) -> None:
        state = np.array([0.2, -0.1, 0.5, 0.3])
        assert np.isfinite(total_energy(state, _P))
