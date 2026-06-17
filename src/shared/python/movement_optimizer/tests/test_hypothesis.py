# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Property-based tests using Hypothesis.

These tests fuzz the core models with randomly generated inputs to
catch edge cases that unit tests might miss.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from movement_optimizer.models import (
    BodyModel,
    HillTorqueModel,
    LagrangianDynamics,
    clamp_joint_angles,
)
from movement_optimizer.trajectory import (
    build_initial_guess,
    build_perturbed_guess,
    build_splines,
    eval_trajectory,
)
from movement_optimizer.trajectory.optimizer_constraints import joint_limit_constraint_values
from movement_optimizer.trajectory.optimizer_cost import (
    compute_torque_cost,
    compute_torque_rate_cost,
)

finite_angle = st.floats(
    min_value=-1.5,
    max_value=1.5,
    allow_nan=False,
    allow_infinity=False,
)


def angle_vector() -> st.SearchStrategy[np.ndarray]:
    return st.builds(
        lambda q0, q1, q2: np.array([q0, q1, q2], dtype=float),
        finite_angle,
        finite_angle,
        finite_angle,
    )


def bounded_angle_vector() -> st.SearchStrategy[np.ndarray]:
    return st.builds(
        lambda q0, q1, q2: np.array([q0, q1, q2], dtype=float),
        st.floats(min_value=-0.9, max_value=0.9, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-0.8, max_value=0.8, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-0.7, max_value=0.7, allow_nan=False, allow_infinity=False),
    )


class TestBodyModelProperties:
    @given(
        body_mass=st.floats(min_value=30.0, max_value=200.0),
        height=st.floats(min_value=1.2, max_value=2.2),
    )
    @settings(max_examples=100)
    def test_body_model_valid_inputs(self, body_mass: float, height: float):
        """BodyModel should accept any reasonable mass and height."""
        body = BodyModel(body_mass=body_mass, height=height)
        assert body.body_mass == body_mass
        assert body.height == height
        assert len(body.L) == 3
        assert all(body.L > 0)
        assert body.inner_heel < body.inner_toe

    @given(
        body_mass=st.floats(min_value=-100.0, max_value=0.0),
    )
    @settings(max_examples=100)
    def test_body_model_rejects_nonpositive_mass(self, body_mass: float):
        """BodyModel should reject non-positive mass."""
        with pytest.raises(ValueError, match="body_mass"):
            BodyModel(body_mass=body_mass, height=1.75)

    @given(
        ll=st.floats(min_value=0.5, max_value=2.0),
        ul=st.floats(min_value=0.5, max_value=2.0),
        to=st.floats(min_value=0.5, max_value=2.0),
    )
    @settings(max_examples=100)
    def test_segment_multipliers_preserve_proportionality(self, ll: float, ul: float, to: float):
        """Segment lengths should scale linearly with multipliers."""
        base = BodyModel(75.0, 1.75)
        scaled = BodyModel(
            75.0,
            1.75,
            seg_multipliers={"lower_leg": ll, "upper_leg": ul, "torso": to},
        )
        np.testing.assert_allclose(scaled.L[0], base.L[0] * ll, rtol=1e-10)
        np.testing.assert_allclose(scaled.L[1], base.L[1] * ul, rtol=1e-10)
        np.testing.assert_allclose(scaled.L[2], base.L[2] * to, rtol=1e-10)


class TestHillTorqueModelProperties:
    @given(
        q=st.floats(min_value=-2.0, max_value=2.0),
        qd=st.floats(min_value=-10.0, max_value=10.0),
    )
    @settings(max_examples=100)
    def test_available_torque_nonnegative(self, q: float, qd: float):
        """Available torque should always be non-negative."""
        model = HillTorqueModel(tau_max=200.0, q_optimal=1.0)
        t = model.available_torque(q, qd)
        assert float(t) >= 0.0

    @given(
        tau_max=st.floats(min_value=10.0, max_value=1000.0),
    )
    @settings(max_examples=100)
    def test_peak_torque_at_optimal_angle(self, tau_max: float):
        """Torque at optimal angle with zero velocity should equal tau_max."""
        model = HillTorqueModel(tau_max=tau_max, q_optimal=1.0)
        t = model.available_torque(1.0, 0.0)
        np.testing.assert_allclose(float(t), tau_max, rtol=1e-6)


class TestClampJointAnglesProperties:
    @given(
        q0=st.floats(min_value=-3.0, max_value=3.0),
        q1=st.floats(min_value=-3.0, max_value=3.0),
        q2=st.floats(min_value=-3.0, max_value=3.0),
    )
    @settings(max_examples=100)
    def test_clamped_angles_within_limits(self, q0: float, q1: float, q2: float):
        """Clamped angles should always be within joint limits."""
        from movement_optimizer.constants import JOINT_LIMITS, JOINT_NAMES

        q = np.array([q0, q1, q2])
        clamped = clamp_joint_angles(q)
        for i, name in enumerate(JOINT_NAMES):
            lo, hi = JOINT_LIMITS[name]
            assert clamped[i] >= lo - 1e-10
            assert clamped[i] <= hi + 1e-10


class TestTrajectoryHelperProperties:
    @given(
        q_start=angle_vector(),
        q_end=angle_vector(),
        n_waypoints=st.integers(min_value=1, max_value=12),
    )
    @settings(max_examples=75)
    def test_initial_guess_is_deterministic_linear_interpolation(
        self,
        q_start: np.ndarray,
        q_end: np.ndarray,
        n_waypoints: int,
    ) -> None:
        """Initial guesses should be deterministic interior points on the endpoint line."""
        guess = build_initial_guess(q_start, q_end, n_waypoints, 3)

        assert guess.shape == (n_waypoints, 3)
        for idx, waypoint in enumerate(guess, start=1):
            alpha = idx / (n_waypoints + 1)
            expected = q_start + alpha * (q_end - q_start)
            np.testing.assert_allclose(waypoint, expected, rtol=1e-12, atol=1e-12)

    @given(
        q_start=bounded_angle_vector(),
        q_end=bounded_angle_vector(),
        seed=st.integers(min_value=0, max_value=100_000),
    )
    @settings(max_examples=75)
    def test_perturbed_guess_is_seed_deterministic_and_clipped(
        self,
        q_start: np.ndarray,
        q_end: np.ndarray,
        seed: int,
    ) -> None:
        """Seeded multi-start guesses should be reproducible and stay within joint bounds."""
        q_bounds = np.array([[-1.0, 1.0], [-0.9, 0.9], [-0.8, 0.8]], dtype=float)

        first = build_perturbed_guess(q_start, q_end, q_bounds, 6, 3, seed)
        second = build_perturbed_guess(q_start, q_end, q_bounds, 6, 3, seed)

        np.testing.assert_array_equal(first, second)
        assert np.all(first >= q_bounds[:, 0])
        assert np.all(first <= q_bounds[:, 1])

    @given(q=angle_vector(), duration=st.floats(min_value=0.5, max_value=5.0))
    @settings(max_examples=75, deadline=None)
    def test_constant_spline_has_no_motion_derivatives(
        self,
        q: np.ndarray,
        duration: float,
    ) -> None:
        """A no-movement spline should evaluate to constant pose with zero derivatives."""
        n_waypoints = 5
        t_ctrl = np.linspace(0.0, duration, n_waypoints + 2)
        x = np.tile(q, n_waypoints)
        splines = build_splines(x, q, q, None, t_ctrl, n_waypoints, 3)
        t_eval = np.linspace(0.0, duration, 11)

        q_eval, qd, qdd, qddd = eval_trajectory(splines, t_eval)

        np.testing.assert_allclose(q_eval, np.tile(q, (len(t_eval), 1)), atol=1e-10)
        np.testing.assert_allclose(qd, 0.0, atol=1e-10)
        np.testing.assert_allclose(qdd, 0.0, atol=1e-10)
        np.testing.assert_allclose(qddd, 0.0, atol=1e-10)

    @given(q=bounded_angle_vector())
    @settings(max_examples=75, deadline=None)
    def test_constant_in_bounds_spline_satisfies_joint_constraints(
        self,
        q: np.ndarray,
    ) -> None:
        """Joint-limit constraints should be non-negative for an in-bounds trajectory."""
        n_waypoints = 4
        q_bounds = np.array([[-1.0, 1.0], [-0.9, 0.9], [-0.8, 0.8]], dtype=float)
        t_ctrl = np.linspace(0.0, 1.0, n_waypoints + 2)
        t_eval = np.linspace(0.0, 1.0, 9)
        x = np.tile(q, n_waypoints)

        def build_splines_fn(flat_x: np.ndarray):
            return build_splines(flat_x, q, q, None, t_ctrl, n_waypoints, 3)

        constraints = joint_limit_constraint_values(x, build_splines_fn, t_eval, q_bounds)

        assert constraints.shape == (2 * len(t_eval) * 3,)
        assert np.all(constraints >= -1e-10)


class TestOptimizationCostProperties:
    @given(
        values=st.lists(
            st.floats(min_value=-200.0, max_value=200.0, allow_nan=False, allow_infinity=False),
            min_size=6,
            max_size=30,
        ),
        dt=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        scale=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=75)
    def test_torque_cost_scales_quadratically(
        self,
        values: list[float],
        dt: float,
        scale: float,
    ) -> None:
        """Torque objective should be deterministic and homogeneous of degree two."""
        usable = values[: len(values) - (len(values) % 3)]
        torques = np.array(usable, dtype=float).reshape(-1, 3)

        base_cost = compute_torque_cost(torques, dt)
        scaled_cost = compute_torque_cost(scale * torques, dt)

        np.testing.assert_allclose(scaled_cost, scale**2 * base_cost, rtol=1e-12, atol=1e-9)

    @given(
        row=st.tuples(
            st.floats(min_value=-200.0, max_value=200.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=-200.0, max_value=200.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=-200.0, max_value=200.0, allow_nan=False, allow_infinity=False),
        ),
        n_eval=st.integers(min_value=2, max_value=20),
        dt=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        weight=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=75)
    def test_torque_rate_cost_zero_for_constant_torque(
        self,
        row: tuple[float, float, float],
        n_eval: int,
        dt: float,
        weight: float,
    ) -> None:
        """A constant torque trajectory has no rate-change penalty."""
        torques = np.tile(np.array(row, dtype=float), (n_eval, 1))

        cost = compute_torque_rate_cost(torques, dt, weight)

        assert cost == 0.0


class TestInverseDynamicsProperties:
    @given(
        q0=st.floats(min_value=-1.5, max_value=1.5),
        q1=st.floats(min_value=-1.5, max_value=1.5),
        q2=st.floats(min_value=-1.5, max_value=1.5),
    )
    @settings(max_examples=100)
    def test_static_torques_finite(self, q0: float, q1: float, q2: float):
        """Static torques (zero velocity/acceleration) should be finite."""
        body = BodyModel(75.0, 1.75)
        dyn = LagrangianDynamics(body, body.m_squat.copy(), body.I_squat.copy(), 60.0)
        q = np.array([q0, q1, q2])
        qd = np.zeros(3)
        qdd = np.zeros(3)
        tau = dyn.inverse_dynamics(q, qd, qdd)
        assert np.all(np.isfinite(tau))

    @given(
        q0=st.floats(min_value=-1.5, max_value=1.5),
        q1=st.floats(min_value=-1.5, max_value=1.5),
        q2=st.floats(min_value=-1.5, max_value=1.5),
    )
    @settings(max_examples=100)
    def test_mass_matrix_positive_definite(self, q0: float, q1: float, q2: float):
        """Mass matrix should be positive definite for any configuration."""
        body = BodyModel(75.0, 1.75)
        dyn = LagrangianDynamics(body, body.m_squat.copy(), body.I_squat.copy(), 60.0)
        q = np.array([q0, q1, q2])
        M = dyn.mass_matrix(q)
        eigenvalues = np.linalg.eigvalsh(M)
        assert np.all(eigenvalues > 0)


class TestTrajectoryOptimizerProperties:
    @given(
        body_mass=st.floats(min_value=50.0, max_value=120.0),
        height=st.floats(min_value=1.5, max_value=1.9),
        bar_mass=st.floats(min_value=0.0, max_value=200.0),
    )
    @settings(max_examples=50)
    def test_optimizer_produces_finite_cost(self, body_mass: float, height: float, bar_mass: float):
        """Optimizer should always produce a finite cost for valid inputs."""
        from movement_optimizer.models.exercise_configs import make_squat_config
        from movement_optimizer.trajectory import TrajectoryOptimizer

        body = BodyModel(body_mass, height)
        dyn, qs, qe, qb = make_squat_config(body, bar_mass)

        opt = TrajectoryOptimizer(
            body,
            dyn,
            "squat",
            bar_mass,
            qs,
            qe,
            qb,
            duration=2.0,
            n_waypoints=4,
            n_starts=1,
        )
        wp0 = opt._initial_guess()
        cost = opt.cost(wp0.flatten())
        assert np.isfinite(cost)

    @given(
        q0=st.floats(min_value=-1.0, max_value=1.0),
        q1=st.floats(min_value=-1.0, max_value=1.0),
        q2=st.floats(min_value=-1.0, max_value=1.0),
    )
    @settings(max_examples=50)
    def test_cost_at_start_equals_end_for_static_pose(self, q0: float, q1: float, q2: float):
        """Cost should be consistent for static start/end poses."""
        from movement_optimizer.models.exercise_configs import make_squat_config
        from movement_optimizer.trajectory import TrajectoryOptimizer

        body = BodyModel(75.0, 1.75)
        q = np.array([q0, q1, q2])
        dyn, _, _, qb = make_squat_config(body, 0.0)

        opt1 = TrajectoryOptimizer(
            body, dyn, "squat", 0.0, q, q, qb, duration=2.0, n_waypoints=4, n_starts=1
        )
        opt2 = TrajectoryOptimizer(
            body, dyn, "squat", 0.0, q, q, qb, duration=2.0, n_waypoints=4, n_starts=1
        )
        wp = opt1._initial_guess().flatten()
        c1 = opt1.cost(wp)
        c2 = opt2.cost(wp)
        assert c1 == pytest.approx(c2, abs=1e-12)


class TestCostFunctionProperties:
    @given(
        n=st.integers(min_value=10, max_value=200),
    )
    @settings(max_examples=50)
    def test_torque_cost_nonnegative(self, n: int):
        """Torque cost should always be non-negative."""
        from movement_optimizer.trajectory.optimizer_cost import compute_torque_cost

        torques = np.random.RandomState(42).randn(n, 3) * 100
        dt = 0.01
        cost = compute_torque_cost(torques, dt)
        assert cost >= 0.0

    @given(
        n=st.integers(min_value=10, max_value=200),
    )
    @settings(max_examples=50)
    def test_jerk_cost_nonnegative(self, n: int):
        """Jerk cost should always be non-negative."""
        from movement_optimizer.trajectory.optimizer_cost import compute_jerk_cost

        qddd = np.random.RandomState(43).randn(n, 3) * 10
        dt = 0.01
        cost = compute_jerk_cost(qddd, dt, 1.0)
        assert cost >= 0.0

    @given(
        n=st.integers(min_value=10, max_value=200),
    )
    @settings(max_examples=50)
    def test_balance_cost_nonnegative(self, n: int):
        """Balance cost should always be non-negative."""
        from movement_optimizer.trajectory.optimizer_cost import compute_balance_cost

        com_x = np.random.RandomState(44).randn(n) * 0.1
        dt = 0.01
        cost = compute_balance_cost(com_x, 0.0, dt, 1.0)
        assert cost >= 0.0
