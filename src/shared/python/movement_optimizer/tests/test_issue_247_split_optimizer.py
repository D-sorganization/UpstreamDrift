# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Characterization and unit tests for helpers extracted in issue #247.

Covers:
- optimizer_spline.build_splines
- optimizer_spline.eval_trajectory
- optimizer_bench.compute_bench_bar_cost
"""

from __future__ import annotations

import numpy as np
import pytest

from movement_optimizer.models import BodyModel, make_bench_press_config, make_squat_config
from movement_optimizer.trajectory import TrajectoryOptimizer
from movement_optimizer.trajectory.optimizer_bench import compute_bench_bar_cost
from movement_optimizer.trajectory.optimizer_spline import build_splines, eval_trajectory

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def squat_spline_args():
    """Minimal squat configuration for spline tests."""
    body = BodyModel(75.0, 1.75)
    _dyn, qs, qe, qb = make_squat_config(body, 60.0)
    n_waypoints = 6
    n_dof = qb.shape[0]
    n_ctrl = n_waypoints + 2
    t_ctrl = np.linspace(0, 2.0, n_ctrl)
    t_eval = np.linspace(0, 2.0, 20, dtype=np.float64)
    wp = np.zeros(n_waypoints * n_dof)
    for j in range(n_dof):
        wp_j = np.linspace(qs[j], qe[j], n_waypoints + 2)[1:-1]
        wp[j::n_dof] = wp_j
    return dict(
        x=wp,
        q_start=qs,
        q_end=qe,
        q_via=None,
        t_ctrl=t_ctrl,
        n_waypoints=n_waypoints,
        n_dof=n_dof,
        t_eval=t_eval,
    )


@pytest.fixture()
def bench_optimizer():
    body = BodyModel(75.0, 1.75)
    dyn, qs, qe, qb, *_ = make_bench_press_config(body, 60.0)
    opt = TrajectoryOptimizer(
        body,
        dyn,
        "bench_press",
        60.0,
        qs,
        qe,
        qb,
        duration=2.0,
        n_waypoints=6,
        n_eval=20,
        n_starts=1,
    )
    return opt


# ---------------------------------------------------------------------------
# build_splines tests
# ---------------------------------------------------------------------------


class TestBuildSplines:
    def test_returns_correct_number_of_splines(self, squat_spline_args) -> None:
        a = squat_spline_args
        spline = build_splines(
            a["x"],
            a["q_start"],
            a["q_end"],
            a["q_via"],
            a["t_ctrl"],
            a["n_waypoints"],
            a["n_dof"],
        )
        # Evaluated spline output shape should have n_dof as the last dimension.
        assert spline(a["t_eval"]).shape[-1] == a["n_dof"]

    def test_splines_satisfy_boundary_conditions(self, squat_spline_args) -> None:
        """Clamped spline must pass through start and end control points."""
        a = squat_spline_args
        spline = build_splines(
            a["x"],
            a["q_start"],
            a["q_end"],
            a["q_via"],
            a["t_ctrl"],
            a["n_waypoints"],
            a["n_dof"],
        )
        t0 = a["t_ctrl"][0]
        tf = a["t_ctrl"][-1]
        q_start_eval = spline(t0)
        q_end_eval = spline(tf)
        for j in range(a["n_dof"]):
            assert abs(float(q_start_eval[j]) - a["q_start"][j]) < 1e-10, (
                f"DOF {j}: spline does not pass through q_start"
            )
            assert abs(float(q_end_eval[j]) - a["q_end"][j]) < 1e-10, (
                f"DOF {j}: spline does not pass through q_end"
            )

    def test_splines_with_via_point(self) -> None:
        """Via-point variant must also honour boundary conditions."""
        body = BodyModel(75.0, 1.75)
        from movement_optimizer.models import make_full_squat_config

        _dyn, qs, qe, qb, q_via = make_full_squat_config(body, 60.0)
        n_waypoints = 8
        n_dof = qb.shape[0]
        n_ctrl = n_waypoints + 3  # +1 for via
        t_ctrl = np.linspace(0, 4.0, n_ctrl)
        wp = np.zeros(n_waypoints * n_dof)
        spline = build_splines(wp, qs, qe, q_via, t_ctrl, n_waypoints, n_dof)
        assert spline(t_ctrl).shape[-1] == n_dof
        # Start and end boundary conditions
        q_start_eval = spline(t_ctrl[0])
        q_end_eval = spline(t_ctrl[-1])
        for j in range(n_dof):
            assert abs(float(q_start_eval[j]) - qs[j]) < 1e-10
            assert abs(float(q_end_eval[j]) - qe[j]) < 1e-10

    def test_zero_waypoints_vector_uses_linear_interp(self, squat_spline_args) -> None:
        """All-zero waypoint vector still produces valid splines."""
        a = squat_spline_args
        x_zeros = np.zeros_like(a["x"])
        spline = build_splines(
            x_zeros,
            a["q_start"],
            a["q_end"],
            a["q_via"],
            a["t_ctrl"],
            a["n_waypoints"],
            a["n_dof"],
        )
        assert spline(a["t_eval"]).shape[-1] == a["n_dof"]


# ---------------------------------------------------------------------------
# eval_trajectory tests
# ---------------------------------------------------------------------------


class TestEvalTrajectory:
    def test_output_shapes(self, squat_spline_args) -> None:
        a = squat_spline_args
        splines = build_splines(
            a["x"],
            a["q_start"],
            a["q_end"],
            a["q_via"],
            a["t_ctrl"],
            a["n_waypoints"],
            a["n_dof"],
        )
        q, qd, qdd, qddd = eval_trajectory(splines, a["t_eval"])
        n_eval = len(a["t_eval"])
        n_dof = a["n_dof"]
        assert q.shape == (n_eval, n_dof)
        assert qd.shape == (n_eval, n_dof)
        assert qdd.shape == (n_eval, n_dof)
        assert qddd.shape == (n_eval, n_dof)

    def test_output_dtype_float64(self, squat_spline_args) -> None:
        a = squat_spline_args
        splines = build_splines(
            a["x"],
            a["q_start"],
            a["q_end"],
            a["q_via"],
            a["t_ctrl"],
            a["n_waypoints"],
            a["n_dof"],
        )
        q, qd, qdd, qddd = eval_trajectory(splines, a["t_eval"])
        for arr in (q, qd, qdd, qddd):
            assert arr.dtype == np.float64

    def test_position_is_finite(self, squat_spline_args) -> None:
        a = squat_spline_args
        splines = build_splines(
            a["x"],
            a["q_start"],
            a["q_end"],
            a["q_via"],
            a["t_ctrl"],
            a["n_waypoints"],
            a["n_dof"],
        )
        q, _qd, _qdd, _qddd = eval_trajectory(splines, a["t_eval"])
        assert np.all(np.isfinite(q)), "Position trajectory must be finite"

    def test_velocity_at_t0_near_zero_for_clamped(self, squat_spline_args) -> None:
        """Clamped spline has zero first derivative at endpoints."""
        a = squat_spline_args
        splines = build_splines(
            a["x"],
            a["q_start"],
            a["q_end"],
            a["q_via"],
            a["t_ctrl"],
            a["n_waypoints"],
            a["n_dof"],
        )
        _, qd, _, _ = eval_trajectory(splines, a["t_eval"])
        # Clamped BC → velocity ≈ 0 at first eval point
        np.testing.assert_allclose(qd[0], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# optimizer_bench.compute_bench_bar_cost tests
# ---------------------------------------------------------------------------


class TestComputeBenchBarCost:
    def test_returns_non_negative_float(self, bench_optimizer) -> None:
        opt = bench_optimizer
        wp0 = opt._initial_guess()
        splines = opt.build_splines(wp0.flatten())
        q, _, _, _ = opt.eval_trajectory(splines)
        cost = compute_bench_bar_cost(q, opt.dynamics.L, opt.dt)
        assert isinstance(cost, float)
        assert cost >= 0.0

    def test_zero_q_gives_zero_cost(self, bench_optimizer) -> None:
        """All-zero joint angles → all hand_x are zero → zero cost."""
        opt = bench_optimizer
        q_zero = np.zeros((opt.n_eval, opt.n_dof))
        cost = compute_bench_bar_cost(q_zero, opt.dynamics.L, opt.dt)
        assert cost == pytest.approx(0.0, abs=1e-12)

    def test_cost_scales_with_magnitude(self, bench_optimizer) -> None:
        """Larger joint angles → larger bench bar cost."""
        opt = bench_optimizer
        q_small = np.full((opt.n_eval, opt.n_dof), 0.1)
        q_large = np.full((opt.n_eval, opt.n_dof), 0.5)
        cost_small = compute_bench_bar_cost(q_small, opt.dynamics.L, opt.dt)
        cost_large = compute_bench_bar_cost(q_large, opt.dynamics.L, opt.dt)
        assert cost_large > cost_small

    def test_cost_is_finite_for_initial_guess(self, bench_optimizer) -> None:
        opt = bench_optimizer
        wp0 = opt._initial_guess()
        splines = opt.build_splines(wp0.flatten())
        q, _, _, _ = opt.eval_trajectory(splines)
        assert np.isfinite(compute_bench_bar_cost(q, opt.dynamics.L, opt.dt))


# ---------------------------------------------------------------------------
# Round-trip: optimizer uses spline module internally
# ---------------------------------------------------------------------------


class TestOptimizerUsesSplineModule:
    def test_build_splines_matches_standalone(self, squat_spline_args) -> None:
        """TrajectoryOptimizer.build_splines must produce same result as the
        standalone build_splines function."""
        a = squat_spline_args
        body = BodyModel(75.0, 1.75)
        _dyn, qs, qe, qb = make_squat_config(body, 60.0)
        opt = TrajectoryOptimizer(
            body,
            _dyn,
            "squat",
            60.0,
            qs,
            qe,
            qb,
            duration=2.0,
            n_waypoints=a["n_waypoints"],
            n_eval=20,
            n_starts=1,
        )
        wp = opt._initial_guess()
        spline_opt = opt.build_splines(wp.flatten())
        spline_direct = build_splines(
            wp.flatten(),
            qs,
            qe,
            None,
            opt.t_ctrl,
            opt.n_waypoints,
            opt.n_dof,
        )
        t_test = np.linspace(0, 2.0, 15)
        np.testing.assert_allclose(
            spline_opt(t_test),
            spline_direct(t_test),
            rtol=1e-12,
            err_msg="TrajectoryOptimizer.build_splines diverges from standalone",
        )

    def test_eval_trajectory_matches_standalone(self, squat_spline_args) -> None:
        """TrajectoryOptimizer.eval_trajectory must match standalone eval_trajectory."""
        a = squat_spline_args
        body = BodyModel(75.0, 1.75)
        _dyn, qs, qe, qb = make_squat_config(body, 60.0)
        opt = TrajectoryOptimizer(
            body,
            _dyn,
            "squat",
            60.0,
            qs,
            qe,
            qb,
            duration=2.0,
            n_waypoints=a["n_waypoints"],
            n_eval=20,
            n_starts=1,
        )
        wp = opt._initial_guess()
        splines = opt.build_splines(wp.flatten())
        q_opt, qd_opt, qdd_opt, qddd_opt = opt.eval_trajectory(splines)
        q_fn, qd_fn, qdd_fn, qddd_fn = eval_trajectory(splines, opt.t_eval)
        np.testing.assert_array_equal(q_opt, q_fn)
        np.testing.assert_array_equal(qd_opt, qd_fn)
        np.testing.assert_array_equal(qdd_opt, qdd_fn)
        np.testing.assert_array_equal(qddd_opt, qddd_fn)
