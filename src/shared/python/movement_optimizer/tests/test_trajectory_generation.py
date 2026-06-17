# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for trajectory generation concerns.

Covers: optimizer construction and preconditions, spline construction,
and cost sub-term evaluation.
"""

from __future__ import annotations

import numpy as np
import pytest

from movement_optimizer.models import (
    BodyModel,
    make_squat_config,
)
from movement_optimizer.trajectory import (
    TrajectoryOptimizer,
)
from movement_optimizer.trajectory.optimizer_cost import (
    compute_balance_cost,
    compute_endpoint_damping_cost,
    compute_jerk_cost,
    compute_torque_cost,
    compute_torque_rate_cost,
)

# ==============================================================
# Construction & Preconditions
# ==============================================================


class TestConstruction:
    def test_too_few_waypoints_raises(self) -> None:
        body = BodyModel(75.0, 1.75)
        dyn, qs, qe, qb = make_squat_config(body, 60.0)
        with pytest.raises(ValueError, match="waypoints"):
            TrajectoryOptimizer(
                body,
                dyn,
                "squat",
                60.0,
                qs,
                qe,
                qb,
                n_waypoints=2,
            )

    def test_bad_bounds_shape_raises(self) -> None:
        body = BodyModel(75.0, 1.75)
        dyn, qs, qe, _ = make_squat_config(body, 60.0)
        bad_bounds = np.zeros((2, 3))  # wrong second dim (3 instead of 2)
        with pytest.raises(ValueError, match="q_bounds"):
            TrajectoryOptimizer(
                body,
                dyn,
                "squat",
                60.0,
                qs,
                qe,
                bad_bounds,
            )

    def test_inner_bos_stored(self, squat_optimizer) -> None:
        """Optimizer should store inner BOS from body model."""
        opt, body, _, _, _ = squat_optimizer
        assert opt.inner_heel == body.inner_heel
        assert opt.inner_toe == body.inner_toe


# ==============================================================
# Spline & Trajectory
# ==============================================================


class TestSplines:
    def test_spline_endpoints(self, squat_optimizer) -> None:
        opt, _, _, qs, qe = squat_optimizer
        wp = opt._initial_guess()
        splines = opt.build_splines(wp.flatten())
        q, _, _, _ = opt.eval_trajectory(splines)
        np.testing.assert_allclose(q[0], qs, atol=1e-6)
        np.testing.assert_allclose(q[-1], qe, atol=1e-6)

    def test_clamped_boundary_velocities(self, squat_optimizer) -> None:
        opt, _, _, _, _ = squat_optimizer
        wp = opt._initial_guess()
        splines = opt.build_splines(wp.flatten())
        _, qd, _, _ = opt.eval_trajectory(splines)
        np.testing.assert_allclose(qd[0], 0, atol=0.1)
        np.testing.assert_allclose(qd[-1], 0, atol=0.1)

    def test_via_point_trajectory(self, full_squat_optimizer) -> None:
        opt, _, _ = full_squat_optimizer
        wp = opt._initial_guess()
        splines = opt.build_splines(wp.flatten())
        q, _, _, _ = opt.eval_trajectory(splines)
        mid = len(q) // 2
        assert q[mid, 1] < np.radians(-60), "Thigh should flex significantly at midpoint"


# ==============================================================
# Cost sub-terms
# ==============================================================


class TestCostTerms:
    def test_torque_cost_positive(self, squat_optimizer) -> None:
        opt, _, dyn, _, _ = squat_optimizer
        wp = opt._initial_guess()
        splines = opt.build_splines(wp.flatten())
        q, qd, qdd, _ = opt.eval_trajectory(splines)
        torques = dyn.inverse_dynamics_batch(q, qd, qdd)
        cost = compute_torque_cost(torques, opt.dt)
        assert cost > 0

    def test_jerk_cost_positive(self, squat_optimizer) -> None:
        opt, _, _, _, _ = squat_optimizer
        wp = opt._initial_guess()
        splines = opt.build_splines(wp.flatten())
        _, _, _, qddd = opt.eval_trajectory(splines)
        cost = compute_jerk_cost(qddd, opt.dt, opt.jerk_weight)
        assert cost > 0

    def test_torque_rate_cost_positive(self, squat_optimizer) -> None:
        opt, _, dyn, _, _ = squat_optimizer
        wp = opt._initial_guess()
        splines = opt.build_splines(wp.flatten())
        q, qd, qdd, _ = opt.eval_trajectory(splines)
        torques = dyn.inverse_dynamics_batch(q, qd, qdd)
        cost = compute_torque_rate_cost(torques, opt.dt, opt.torque_rate_weight)
        assert cost > 0

    def test_endpoint_damping_zero_at_rest(self) -> None:
        body = BodyModel(75.0, 1.75)
        dyn, qs, qe, qb = make_squat_config(body, 60.0)
        opt = TrajectoryOptimizer(
            body,
            dyn,
            "squat",
            60.0,
            qs,
            qe,
            qb,
            n_waypoints=6,
            n_eval=20,
            n_starts=1,
        )
        qd = np.zeros((20, 3))
        qdd = np.zeros((20, 3))
        cost = compute_endpoint_damping_cost(
            qd, qdd, opt.dt, opt.endpoint_weight, opt._n_damp, opt._damp_weights
        )
        assert cost == 0.0

    def test_balance_cost_inside_is_centering_only(self, squat_optimizer) -> None:
        """COM inside inner bounds should incur only centering cost (no barrier)."""
        opt, body, _, _, _ = squat_optimizer
        center = body.inner_center
        com_x = np.full(20, center)
        cost = compute_balance_cost(com_x, opt.inner_center, opt.dt, opt.balance_center_weight)
        # Should be zero since COM == center
        assert cost < 1e-10

    def test_balance_cost_outside_is_very_high(self, squat_optimizer) -> None:
        """COM outside inner bounds should incur very high barrier cost."""
        opt, body, _, _, _ = squat_optimizer
        # Place COM well outside inner bounds
        com_x = np.full(20, body.heel_x - 0.05)
        cost_outside = compute_balance_cost(
            com_x, opt.inner_center, opt.dt, opt.balance_center_weight
        )
        # Compare to COM at center
        com_x_center = np.full(20, body.inner_center)
        cost_center = compute_balance_cost(
            com_x_center, opt.inner_center, opt.dt, opt.balance_center_weight
        )
        assert cost_outside > cost_center * 100

    def test_total_cost_is_sum(self, squat_optimizer) -> None:
        opt, _, dyn, _, _ = squat_optimizer
        wp = opt._initial_guess()
        x = wp.flatten()
        splines = opt.build_splines(x)
        q, qd, qdd, qddd = opt.eval_trajectory(splines)

        torques = dyn.inverse_dynamics_batch(q, qd, qdd)
        com_x = dyn.com_x_batch(q, "squat", 60.0)

        total = (
            compute_torque_cost(torques, opt.dt)
            + compute_jerk_cost(qddd, opt.dt, opt.jerk_weight)
            + compute_torque_rate_cost(torques, opt.dt, opt.torque_rate_weight)
            + compute_endpoint_damping_cost(
                qd, qdd, opt.dt, opt.endpoint_weight, opt._n_damp, opt._damp_weights
            )
            + compute_balance_cost(com_x, opt.inner_center, opt.dt, opt.balance_center_weight)
        )
        computed = opt._compute_cost(x)
        np.testing.assert_allclose(computed, total, rtol=1e-10)
