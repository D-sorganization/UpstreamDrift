# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for exercise configuration factories (clean, jerk, snatch, bench press)."""

from __future__ import annotations

import numpy as np

from movement_optimizer.constants import PLATE_RADIUS_STD_M
from movement_optimizer.exercises import (
    make_clean_config,
    make_jerk_config,
    make_snatch_config,
)
from movement_optimizer.models import (
    BodyModel,
    LagrangianDynamics,
    make_bench_press_config,
)

# ------------------------------------------------------------------
# Clean
# ------------------------------------------------------------------


class TestCleanConfig:
    def test_clean_config_shapes(self, default_body: BodyModel) -> None:
        dyn, qs, qe, qb, q_via = make_clean_config(default_body, 60.0)
        assert isinstance(dyn, LagrangianDynamics)
        assert qs.shape == (3,)
        assert qe.shape == (3,)
        assert qb.shape == (3, 2)
        assert q_via is not None
        assert q_via.shape == (3,)

    def test_clean_start_near_floor(self, default_body: BodyModel) -> None:
        dyn, qs, _qe, _qb, _q_via = make_clean_config(default_body, 60.0)
        # Bar at start should be near plate height (floor)
        bar_pos = dyn.bar_position(qs, "deadlift")
        assert abs(bar_pos[1] - PLATE_RADIUS_STD_M) < 0.20

    def test_clean_end_at_front_rack(self, default_body: BodyModel) -> None:
        dyn, _qs, qe, _qb, _q_via = make_clean_config(default_body, 60.0)
        # Bar at end should be near shoulder height (front rack), NOT overhead
        fk = dyn.forward_kinematics(qe)
        shoulder_h = fk["shoulder"][1]
        bar_pos = dyn.bar_position(qe, "deadlift")
        # Front rack: bar is at shoulder height (deadlift bar_position = shoulder - arm)
        # It should be well above the floor (at least 40% of shoulder height)
        assert bar_pos[1] > shoulder_h * 0.4
        # Torso should be near vertical (front rack standing position)
        # balance_pose may adjust the torso angle to maintain COM balance
        assert abs(qe[2]) < np.radians(30), "Clean end: torso should be near vertical"

    def test_clean_has_via_point(self, default_body: BodyModel) -> None:
        _dyn, qs, qe, _qb, q_via = make_clean_config(default_body, 60.0)
        assert q_via is not None
        # Via-point should be distinct from start and end
        assert not np.allclose(q_via, qs, atol=0.01)
        assert not np.allclose(q_via, qe, atol=0.01)


# ------------------------------------------------------------------
# Jerk
# ------------------------------------------------------------------


class TestJerkConfig:
    def test_jerk_config_shapes(self, default_body: BodyModel) -> None:
        dyn, qs, qe, qb, q_via = make_jerk_config(default_body, 60.0)
        assert isinstance(dyn, LagrangianDynamics)
        assert qs.shape == (3,)
        assert qe.shape == (3,)
        assert qb.shape == (3, 2)
        assert q_via is not None

    def test_jerk_start_at_rack(self, default_body: BodyModel) -> None:
        dyn, qs, _qe, _qb, _q_via = make_jerk_config(default_body, 60.0)
        # Start should be near-standing (front rack position)
        fk = dyn.forward_kinematics(qs)
        shoulder_h = fk["shoulder"][1]
        total_h = default_body.L.sum()
        # Shoulder should be relatively high (standing-ish)
        assert shoulder_h > total_h * 0.85

    def test_jerk_end_overhead(self, default_body: BodyModel) -> None:
        dyn, _qs, qe, _qb, _q_via = make_jerk_config(default_body, 60.0)
        # End: torso near vertical (bar overhead)
        assert abs(qe[2]) < np.radians(10), "Jerk end: torso must be near vertical (overhead)"
        # Shoulder should be near standing height (overhead lockout)
        fk = dyn.forward_kinematics(qe)
        shoulder_h = fk["shoulder"][1]
        total_h = default_body.L.sum()
        assert shoulder_h > total_h * 0.90, "Jerk end: shoulder must be high (overhead)"


# ------------------------------------------------------------------
# Snatch
# ------------------------------------------------------------------


class TestSnatchConfig:
    def test_snatch_config_shapes(self, default_body: BodyModel) -> None:
        dyn, qs, qe, qb, q_via = make_snatch_config(default_body, 60.0)
        assert isinstance(dyn, LagrangianDynamics)
        assert qs.shape == (3,)
        assert qe.shape == (3,)
        assert qb.shape == (3, 2)
        assert q_via is not None

    def test_snatch_start_near_floor(self, default_body: BodyModel) -> None:
        dyn, qs, _qe, _qb, _q_via = make_snatch_config(default_body, 60.0)
        bar_pos = dyn.bar_position(qs, "deadlift")
        assert abs(bar_pos[1] - PLATE_RADIUS_STD_M) < 0.20

    def test_snatch_end_overhead(self, default_body: BodyModel) -> None:
        dyn, _qs, qe, _qb, _q_via = make_snatch_config(default_body, 60.0)
        # End: torso near vertical (bar overhead)
        assert abs(qe[2]) < np.radians(10), "Snatch end: torso must be near vertical (overhead)"
        fk = dyn.forward_kinematics(qe)
        shoulder_h = fk["shoulder"][1]
        total_h = default_body.L.sum()
        assert shoulder_h > total_h * 0.90, "Snatch end: shoulder must be high (overhead)"

    def test_snatch_has_via_points(self, default_body: BodyModel) -> None:
        _dyn, _qs, _qe, _qb, q_via = make_snatch_config(default_body, 60.0)
        assert q_via is not None

    def test_snatch_via_is_overhead_squat(self, default_body: BodyModel) -> None:
        _dyn, _qs, _qe, _qb, q_via = make_snatch_config(default_body, 60.0)
        # Via-point should have significant knee flexion (deep squat)
        assert q_via[1] < np.radians(-60), "Snatch via: should be deep squat"
        # Torso relatively upright for overhead position
        # balance_pose may adjust the torso angle to maintain COM balance
        assert abs(q_via[2]) < np.radians(80), "Snatch via: torso should be reasonably upright"


# ------------------------------------------------------------------
# Bench Press
# ------------------------------------------------------------------


class TestBenchPressConfig:
    def test_bench_is_full_rep(self, default_body: BodyModel) -> None:
        _dyn, qs, qe, _qb, q_via = make_bench_press_config(default_body, 60.0)
        # Full rep: start and end should be the same (lockout)
        np.testing.assert_allclose(qs, qe, atol=1e-10)
        # Via-point should exist (bar at chest)
        assert q_via is not None
        # Via-point should be different from start/end
        assert not np.allclose(q_via, qs, atol=0.01)

    def test_bench_config_shapes(self, default_body: BodyModel) -> None:
        dyn, qs, qe, qb, q_via = make_bench_press_config(default_body, 60.0)
        assert isinstance(dyn, LagrangianDynamics)
        assert qs.shape == (3,)
        assert qe.shape == (3,)
        assert qb.shape == (3, 2)
        assert q_via is not None
        assert q_via.shape == (3,)

    def test_bench_no_com_constraint(self, default_body: BodyModel) -> None:
        """Bench press optimizer should skip COM constraints only."""
        from movement_optimizer.trajectory import TrajectoryOptimizer
        from movement_optimizer.trajectory.optimizer_constraints import (
            joint_limit_constraint_values,
        )

        dyn, qs, qe, qb, q_via = make_bench_press_config(default_body, 60.0)
        opt = TrajectoryOptimizer(
            default_body,
            dyn,
            "bench_press",
            60.0,
            qs,
            qe,
            qb,
            q_via=q_via,
            duration=3.0,
            n_waypoints=8,
        )
        constraints = opt._build_constraints()
        assert len(constraints) == 1, "Bench press should keep only the joint-limit constraint"
        assert constraints[0]["fun"] is joint_limit_constraint_values


# ------------------------------------------------------------------
# Balance: all endpoints COM at inner_center
# ------------------------------------------------------------------


class TestAllEndpointsBalanced:
    """Start and end poses for all exercises should have COM in the inner BOS zone."""

    def _check_com_in_inner_bos(
        self,
        body: BodyModel,
        dyn: LagrangianDynamics,
        q: np.ndarray,
        exercise_type: str,
        bar_mass: float,
        label: str,
    ) -> None:
        com_x = dyn.com_position(q, exercise_type, bar_mass)[0]
        assert body.inner_heel <= com_x <= body.inner_toe, (
            f"{label} COM_x {com_x:.4f} outside inner BOS "
            f"[{body.inner_heel:.4f}, {body.inner_toe:.4f}]"
        )

    def test_clean_endpoints_balanced(self, default_body: BodyModel) -> None:
        dyn, qs, qe, _qb, _q_via = make_clean_config(default_body, 60.0)
        self._check_com_in_inner_bos(default_body, dyn, qs, "deadlift", 60.0, "clean start")
        self._check_com_in_inner_bos(default_body, dyn, qe, "deadlift", 60.0, "clean end")

    def test_jerk_endpoints_balanced(self, default_body: BodyModel) -> None:
        dyn, qs, qe, _qb, _q_via = make_jerk_config(default_body, 60.0)
        self._check_com_in_inner_bos(default_body, dyn, qs, "squat", 60.0, "jerk start")
        self._check_com_in_inner_bos(default_body, dyn, qe, "squat", 60.0, "jerk end")

    def test_snatch_endpoints_balanced(self, default_body: BodyModel) -> None:
        dyn, qs, qe, _qb, _q_via = make_snatch_config(default_body, 60.0)
        self._check_com_in_inner_bos(default_body, dyn, qs, "deadlift", 60.0, "snatch start")
        self._check_com_in_inner_bos(default_body, dyn, qe, "squat", 60.0, "snatch end")
