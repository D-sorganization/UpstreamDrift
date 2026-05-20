"""Tests for TriplePendulumDynamics — mass matrix, bias, forward/inverse, step."""

from __future__ import annotations

import math

import numpy as np
import pytest
from src.engines.pendulum_models.python.double_pendulum_model.physics.triple_pendulum import (
    DAMPING_DEFAULT,
    GRAVITATIONAL_ACCELERATION,
    PolynomialProfile,
    TripleJointTorques,
    TriplePendulumDynamics,
    TriplePendulumParameters,
    TriplePendulumState,
    TripleSegmentProperties,
)


@pytest.fixture()
def dyn() -> TriplePendulumDynamics:
    return TriplePendulumDynamics()


@pytest.fixture()
def zero_state() -> TriplePendulumState:
    return TriplePendulumState(
        theta1=0.0, theta2=0.0, theta3=0.0, omega1=0.0, omega2=0.0, omega3=0.0
    )


@pytest.fixture()
def nonzero_state() -> TriplePendulumState:
    return TriplePendulumState(
        theta1=0.4, theta2=-0.3, theta3=0.2, omega1=0.5, omega2=-0.2, omega3=0.1
    )


class TestTripleSegmentProperties:
    def test_center_of_mass_distance(self) -> None:
        seg = TripleSegmentProperties(
            length_m=2.0, mass_kg=1.0, center_of_mass_ratio=0.5, inertia_about_com=0.1
        )
        assert seg.center_of_mass_distance == pytest.approx(1.0)

    def test_inertia_about_proximal_joint(self) -> None:
        seg = TripleSegmentProperties(
            length_m=2.0, mass_kg=3.0, center_of_mass_ratio=0.5, inertia_about_com=0.1
        )
        assert seg.inertia_about_proximal_joint == pytest.approx(0.1 + 3.0 * 1.0)


class TestTriplePendulumParameters:
    def test_default_constructs(self) -> None:
        params = TriplePendulumParameters.default()
        assert len(params.segments) == 3
        assert params.damping == DAMPING_DEFAULT
        assert params.gravity_enabled is True

    def test_gravity_enabled_true(self) -> None:
        params = TriplePendulumParameters.default()
        assert params.gravity == GRAVITATIONAL_ACCELERATION

    def test_gravity_disabled(self) -> None:
        params = TriplePendulumParameters.default()
        params.gravity_enabled = False
        assert params.gravity == 0.0


class TestPolynomialProfile:
    def test_omega_evaluates_polynomial(self) -> None:
        # coefficients are in numpy poly1d order (highest power first)
        prof = PolynomialProfile(coefficients=(2.0, 1.0, 3.0))  # 2t^2 + t + 3
        assert prof.omega(0.0) == pytest.approx(3.0)
        assert prof.omega(1.0) == pytest.approx(6.0)

    def test_alpha_is_derivative(self) -> None:
        prof = PolynomialProfile(coefficients=(2.0, 1.0, 3.0))  # d/dt = 4t + 1
        assert prof.alpha(0.0) == pytest.approx(1.0)
        assert prof.alpha(1.0) == pytest.approx(5.0)


class TestTriplePendulumMassMatrix:
    def test_mass_matrix_shape(
        self, dyn: TriplePendulumDynamics, zero_state: TriplePendulumState
    ) -> None:
        M = dyn.mass_matrix(zero_state)
        assert M.shape == (3, 3)

    def test_mass_matrix_symmetric(
        self, dyn: TriplePendulumDynamics, nonzero_state: TriplePendulumState
    ) -> None:
        M = dyn.mass_matrix(nonzero_state)
        np.testing.assert_allclose(M, M.T, atol=1e-12)

    def test_mass_matrix_positive_definite(
        self, dyn: TriplePendulumDynamics, nonzero_state: TriplePendulumState
    ) -> None:
        M = dyn.mass_matrix(nonzero_state)
        eigs = np.linalg.eigvalsh(M)
        assert np.all(eigs > 0)

    def test_mass_matrix_positive_definite_random(
        self, dyn: TriplePendulumDynamics
    ) -> None:
        rng = np.random.default_rng(42)
        for _ in range(10):
            t = rng.uniform(-math.pi, math.pi, 3)
            o = rng.uniform(-2.0, 2.0, 3)
            state = TriplePendulumState(
                theta1=t[0],
                theta2=t[1],
                theta3=t[2],
                omega1=o[0],
                omega2=o[1],
                omega3=o[2],
            )
            M = dyn.mass_matrix(state)
            eigs = np.linalg.eigvalsh(M)
            assert np.all(eigs > 0)


class TestTriplePendulumBias:
    def test_bias_zero_at_rest_no_gravity(self) -> None:
        params = TriplePendulumParameters.default()
        params.gravity_enabled = False
        dyn = TriplePendulumDynamics(params)
        state = TriplePendulumState(
            theta1=0.0, theta2=0.0, theta3=0.0, omega1=0.0, omega2=0.0, omega3=0.0
        )
        bias = dyn.bias_vector(state)
        np.testing.assert_allclose(bias, np.zeros(3), atol=1e-12)

    def test_bias_includes_damping(self, dyn: TriplePendulumDynamics) -> None:
        # State with zero angles and gravity off but nonzero omegas -> only damping
        params = TriplePendulumParameters.default()
        params.gravity_enabled = False
        d = TriplePendulumDynamics(params)
        state = TriplePendulumState(
            theta1=0.0, theta2=0.0, theta3=0.0, omega1=1.0, omega2=2.0, omega3=3.0
        )
        bias = d.bias_vector(state)
        expected_damping = np.array(d.parameters.damping) * np.array([1.0, 2.0, 3.0])
        # Coriolis terms vanish at theta=0 (all sin terms are 0)
        np.testing.assert_allclose(bias, expected_damping, atol=1e-10)


class TestTriplePendulumDynamicsAlgorithms:
    def test_inverse_then_forward_identity(
        self, dyn: TriplePendulumDynamics, nonzero_state: TriplePendulumState
    ) -> None:
        target_acc = (0.1, -0.2, 0.05)
        torques = dyn.inverse_dynamics(nonzero_state, target_acc)
        recovered = dyn.forward_dynamics(nonzero_state, torques)
        for r, t in zip(recovered, target_acc, strict=True):
            assert r == pytest.approx(t, abs=1e-9)

    def test_forward_dynamics_returns_three_floats(
        self, dyn: TriplePendulumDynamics, zero_state: TriplePendulumState
    ) -> None:
        accs = dyn.forward_dynamics(zero_state, (0.0, 0.0, 0.0))
        assert len(accs) == 3
        for a in accs:
            assert isinstance(a, float)
            assert math.isfinite(a)

    def test_joint_torque_breakdown(
        self, dyn: TriplePendulumDynamics, nonzero_state: TriplePendulumState
    ) -> None:
        breakdown = dyn.joint_torque_breakdown(nonzero_state, control=(0.1, 0.2, 0.3))
        assert isinstance(breakdown, TripleJointTorques)
        assert breakdown.applied == (0.1, 0.2, 0.3)
        assert len(breakdown.gravitational) == 3
        assert len(breakdown.damping) == 3
        assert len(breakdown.coriolis_centripetal) == 3

    def test_damping_breakdown_proportional_to_omega(
        self, dyn: TriplePendulumDynamics
    ) -> None:
        state = TriplePendulumState(
            theta1=0.0, theta2=0.0, theta3=0.0, omega1=2.0, omega2=-1.0, omega3=0.5
        )
        breakdown = dyn.joint_torque_breakdown(state, control=(0.0, 0.0, 0.0))
        expected = tuple(DAMPING_DEFAULT[i] * o for i, o in enumerate([2.0, -1.0, 0.5]))
        for got, exp in zip(breakdown.damping, expected, strict=True):
            assert got == pytest.approx(exp)


class TestTriplePendulumStep:
    def test_step_at_rest_with_zero_control_stays_at_rest_no_gravity(self) -> None:
        params = TriplePendulumParameters.default()
        params.gravity_enabled = False
        params.damping = (0.0, 0.0, 0.0)
        dyn = TriplePendulumDynamics(params)
        state = TriplePendulumState(
            theta1=0.0, theta2=0.0, theta3=0.0, omega1=0.0, omega2=0.0, omega3=0.0
        )
        new_state = dyn.step(0.0, state, dt=0.01, control=(0.0, 0.0, 0.0))
        assert new_state.theta1 == pytest.approx(0.0, abs=1e-12)
        assert new_state.theta2 == pytest.approx(0.0, abs=1e-12)
        assert new_state.theta3 == pytest.approx(0.0, abs=1e-12)
        assert new_state.omega1 == pytest.approx(0.0, abs=1e-12)
        assert new_state.omega2 == pytest.approx(0.0, abs=1e-12)
        assert new_state.omega3 == pytest.approx(0.0, abs=1e-12)

    def test_step_returns_new_state(
        self, dyn: TriplePendulumDynamics, nonzero_state: TriplePendulumState
    ) -> None:
        new_state = dyn.step(0.0, nonzero_state, dt=0.001, control=(0.0, 0.0, 0.0))
        assert isinstance(new_state, TriplePendulumState)
        # State must have moved due to nonzero omegas
        assert new_state.theta1 != nonzero_state.theta1

    def test_step_energy_approximately_conserved_no_damping_no_gravity(self) -> None:
        params = TriplePendulumParameters.default()
        params.gravity_enabled = False
        params.damping = (0.0, 0.0, 0.0)
        dyn = TriplePendulumDynamics(params)
        state = TriplePendulumState(
            theta1=0.1, theta2=0.0, theta3=0.0, omega1=1.0, omega2=0.0, omega3=0.0
        )

        def kinetic(s: TriplePendulumState) -> float:
            M = dyn.mass_matrix(s)
            omega = np.array([s.omega1, s.omega2, s.omega3])
            return float(0.5 * omega @ M @ omega)

        e0 = kinetic(state)
        s = state
        for _ in range(20):
            s = dyn.step(0.0, s, dt=0.005, control=(0.0, 0.0, 0.0))
        e1 = kinetic(s)
        # Allow generous tolerance — RK4 with short horizon
        assert abs(e1 - e0) / max(abs(e0), 1e-9) < 0.05
