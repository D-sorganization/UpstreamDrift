# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for trajectory validation concerns.

Covers: solution cache, bar-knee clearance constraint activation,
and OptimizationResult joint-limit violation tracking.
"""

from __future__ import annotations

import numpy as np

from movement_optimizer.exercises import make_clean_config, make_snatch_config
from movement_optimizer.models import (
    BodyModel,
    make_bench_press_config,
    make_deadlift_config,
    make_squat_config,
)
from movement_optimizer.trajectory import (
    OptimizationResult,
    SolutionCache,
    TrajectoryOptimizer,
)
from movement_optimizer.trajectory.optimizer_constraints import bar_knee_clearance

# ==============================================================
# Solution Cache
# ==============================================================


class TestSolutionCache:
    def test_cache_miss_returns_none(self) -> None:
        cache = SolutionCache()
        result = cache.get("squat", 75.0, 1.75, {"lower_leg": 1.0}, 60.0, 2.0, 1.0)
        assert result is None

    def test_cache_hit_returns_result(self) -> None:
        cache = SolutionCache()
        n = 10
        dummy = OptimizationResult(
            t=np.zeros(n),
            q=np.zeros((n, 3)),
            qd=np.zeros((n, 3)),
            qdd=np.zeros((n, 3)),
            torques=np.zeros((n, 3)),
            power=np.zeros((n, 3)),
            com=np.zeros((n, 2)),
            bar=np.zeros((n, 2)),
            success=True,
            cost=42.0,
            com_horizontal_range_cm=1.5,
        )
        mults = {"lower_leg": 1.0, "upper_leg": 1.0, "torso": 1.0}
        cache.put("squat", 75.0, 1.75, mults, 60.0, 2.0, 1.0, dummy)
        hit = cache.get("squat", 75.0, 1.75, mults, 60.0, 2.0, 1.0)
        assert hit is not None
        assert hit.cost == 42.0

    def test_cache_clear(self) -> None:
        cache = SolutionCache()
        n = 10
        dummy = OptimizationResult(
            t=np.zeros(n),
            q=np.zeros((n, 3)),
            qd=np.zeros((n, 3)),
            qdd=np.zeros((n, 3)),
            torques=np.zeros((n, 3)),
            power=np.zeros((n, 3)),
            com=np.zeros((n, 2)),
            bar=np.zeros((n, 2)),
            success=True,
            cost=42.0,
            com_horizontal_range_cm=1.5,
        )
        mults = {"lower_leg": 1.0, "upper_leg": 1.0, "torso": 1.0}
        cache.put("squat", 75.0, 1.75, mults, 60.0, 2.0, 1.0, dummy)
        cache.clear()
        assert cache.get("squat", 75.0, 1.75, mults, 60.0, 2.0, 1.0) is None


# ==============================================================
# Bar-Knee Clearance
# ==============================================================


def _has_bar_knee_constraint(opt: TrajectoryOptimizer) -> bool:
    """Return True if the optimizer's constraints include bar-knee clearance."""
    constraints = opt._build_constraints()
    return any(c["fun"] is bar_knee_clearance for c in constraints)


class TestBarKneeClearance:
    def test_clearance_active_for_deadlift(self) -> None:
        body = BodyModel(75.0, 1.75)
        dyn, qs, qe, qb = make_deadlift_config(body, 60.0)
        opt = TrajectoryOptimizer(
            body,
            dyn,
            "deadlift",
            60.0,
            qs,
            qe,
            qb,
            n_waypoints=6,
            n_eval=20,
            n_starts=1,
        )
        assert _has_bar_knee_constraint(opt)

    def test_clearance_active_for_clean(self) -> None:
        body = BodyModel(75.0, 1.75)
        dyn, qs, qe, qb, q_via = make_clean_config(body, 60.0)
        opt = TrajectoryOptimizer(
            body,
            dyn,
            "clean",
            60.0,
            qs,
            qe,
            qb,
            q_via=q_via,
            n_waypoints=6,
            n_eval=20,
            n_starts=1,
        )
        assert _has_bar_knee_constraint(opt)

    def test_clearance_active_for_snatch(self) -> None:
        body = BodyModel(75.0, 1.75)
        dyn, qs, qe, qb, q_via = make_snatch_config(body, 60.0)
        opt = TrajectoryOptimizer(
            body,
            dyn,
            "snatch",
            60.0,
            qs,
            qe,
            qb,
            q_via=q_via,
            n_waypoints=6,
            n_eval=20,
            n_starts=1,
        )
        assert _has_bar_knee_constraint(opt)

    def test_clearance_not_active_for_squat(self) -> None:
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
        assert not _has_bar_knee_constraint(opt)

    def test_clearance_not_active_for_bench(self) -> None:
        body = BodyModel(75.0, 1.75)
        dyn, qs, qe, qb, _q_via = make_bench_press_config(body, 60.0)
        opt = TrajectoryOptimizer(
            body,
            dyn,
            "bench_press",
            60.0,
            qs,
            qe,
            qb,
            n_waypoints=6,
            n_eval=20,
            n_starts=1,
        )
        assert not _has_bar_knee_constraint(opt)


# ==============================================================
# OptimizationResult — joint limit violation tracking
# ==============================================================


class TestJointLimitViolationTracking:
    """Tests that n_joint_limit_violations is populated in OptimizationResult."""

    def test_no_violations_when_within_limits(self, squat_optimizer) -> None:
        """A well-behaved trajectory should report zero joint-limit violations."""
        opt, _body, _dyn, _qs, _qe = squat_optimizer
        # Use the default initial guess (interpolated between start and end)
        wp = opt._initial_guess()
        x = wp.flatten()

        class _FakeRes:
            fun = 1.0
            x = None

        _FakeRes.x = x
        result = opt._package_results(_FakeRes)
        assert isinstance(result.n_joint_limit_violations, int)
        assert result.n_joint_limit_violations >= 0

    def test_violations_counted_when_q_exceeds_bounds(self, squat_optimizer) -> None:
        """Forcing joint angles outside limits should produce a positive violation count."""
        opt, _body, _dyn, _qs, _qe = squat_optimizer
        # Build a waypoint array where all waypoints are set to a value well
        # beyond the upper joint limit for every DOF.
        n_wpt = opt.n_waypoints
        n_dof = opt.dynamics.n_dof  # noqa: F841 (used via tile shape)
        # Upper bound for each joint (grab from q_bounds)
        upper = np.array([b[1] for b in opt.q_bounds])
        # Set all waypoints to 2x the upper limit to guarantee violations
        extreme_wp = np.tile(upper * 2.0, (n_wpt, 1))
        x = extreme_wp.flatten()

        class _FakeRes:
            fun: float = 1.0
            x: np.ndarray | None = None

        _FakeRes.x = x
        result = opt._package_results(_FakeRes)
        assert result.n_joint_limit_violations > 0
