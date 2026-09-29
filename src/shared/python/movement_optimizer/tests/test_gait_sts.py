# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for gait and sit-to-stand exercise configurations."""

from __future__ import annotations

import math

import numpy as np
import pytest

from movement_optimizer.exercises import (
    GaitAnalyzer,
    make_gait_config,
    make_sit_to_stand_config,
)
from movement_optimizer.models import BodyModel, LagrangianDynamics

# ------------------------------------------------------------------
# Gait config
# ------------------------------------------------------------------


class TestGaitConfig:
    def test_gait_config_returns_valid_tuple(self, default_body: BodyModel) -> None:
        dyn, qs, qe, qb, via = make_gait_config(default_body)
        assert isinstance(dyn, LagrangianDynamics)
        assert qs.shape == (3,)
        assert qe.shape == (3,)
        assert qb.shape == (3, 2)
        assert isinstance(via, list)

    def test_gait_is_cyclic(self, default_body: BodyModel) -> None:
        """Start and end angles should be identical (full cycle)."""
        _dyn, qs, qe, _qb, _via = make_gait_config(default_body)
        np.testing.assert_allclose(qs, qe, atol=1e-10)

    def test_gait_has_8_phases(self, default_body: BodyModel) -> None:
        _dyn, _qs, _qe, _qb, via = make_gait_config(default_body)
        assert len(via) == 8

    def test_gait_via_fractions_span_0_to_1(self, default_body: BodyModel) -> None:
        _dyn, _qs, _qe, _qb, via = make_gait_config(default_body)
        assert via[0][0] == 0.0
        assert via[-1][0] == 1.0

    def test_gait_zero_load(self, default_body: BodyModel) -> None:
        dyn, _qs, _qe, _qb, _via = make_gait_config(default_body)
        assert dyn.m_load == 0.0

    def test_gait_rejects_bad_stride(self, default_body: BodyModel) -> None:
        with pytest.raises(ValueError, match="stride_length"):
            make_gait_config(default_body, stride_length=-1.0)

    def test_gait_rejects_bad_duration(self, default_body: BodyModel) -> None:
        with pytest.raises(ValueError, match="cycle_duration"):
            make_gait_config(default_body, cycle_duration=0.0)


# ------------------------------------------------------------------
# GaitAnalyzer
# ------------------------------------------------------------------


class TestGaitAnalyzer:
    def test_spatiotemporal_basic(self, default_body: BodyModel) -> None:
        analyzer = GaitAnalyzer(default_body)
        t = np.linspace(0, 1.0, 100)
        # Simulate knee angles varying through swing
        knee = np.where(t > 0.6, math.radians(-40), math.radians(-10))
        q = np.column_stack([np.zeros(100), knee, np.zeros(100)])
        result = analyzer.compute_spatiotemporal(q, t, stride_length=0.7)
        assert result["stride_length_m"] == 0.7
        assert result["walking_speed_m_s"] == pytest.approx(0.7, rel=1e-6)
        assert result["cycle_duration_s"] == pytest.approx(1.0, rel=1e-6)
        assert 0.0 < result["stance_phase_pct"] < 100.0
        assert result["stance_phase_pct"] + result["swing_phase_pct"] == pytest.approx(100.0)

    def test_symmetry_index_identical(self, default_body: BodyModel) -> None:
        analyzer = GaitAnalyzer(default_body)
        angles = np.linspace(-0.5, 0.5, 50)
        si = analyzer.compute_symmetry_index(angles, angles)
        assert si == pytest.approx(0.0)

    def test_symmetry_index_asymmetric(self, default_body: BodyModel) -> None:
        analyzer = GaitAnalyzer(default_body)
        left = np.linspace(0, 1.0, 50)
        right = np.linspace(0, 0.5, 50)
        si = analyzer.compute_symmetry_index(left, right)
        assert si > 0.0
        assert si == pytest.approx(50.0, rel=1e-6)

    def test_symmetry_index_zero_range(self, default_body: BodyModel) -> None:
        analyzer = GaitAnalyzer(default_body)
        constant = np.ones(10) * 0.5
        si = analyzer.compute_symmetry_index(constant, constant)
        assert si == 0.0

    def test_rejects_bad_model(self) -> None:
        with pytest.raises(TypeError):
            GaitAnalyzer("not a model")  # type: ignore[arg-type]

    def test_spatiotemporal_rejects_bad_stride(self, default_body: BodyModel) -> None:
        analyzer = GaitAnalyzer(default_body)
        with pytest.raises(ValueError, match="stride_length"):
            analyzer.compute_spatiotemporal(np.zeros(10), np.linspace(0, 1, 10), -1.0)


# ------------------------------------------------------------------
# Sit-to-stand config
# ------------------------------------------------------------------


class TestSitToStandConfig:
    def test_sts_config_returns_valid_tuple(self, default_body: BodyModel) -> None:
        dyn, qs, qe, qb, via = make_sit_to_stand_config(default_body)
        assert isinstance(dyn, LagrangianDynamics)
        assert qs.shape == (3,)
        assert qe.shape == (3,)
        assert qb.shape == (3, 2)
        assert isinstance(via, list)

    def test_sts_seated_to_standing(self, default_body: BodyModel) -> None:
        """Start should be seated (large flexion), end near upright."""
        _dyn, qs, qe, _qb, _via = make_sit_to_stand_config(default_body)
        # Seated: knee should be heavily flexed
        assert qs[1] < math.radians(-60)
        # Standing: knee should be near zero
        assert abs(qe[1]) < math.radians(10)

    def test_sts_has_6_phases(self, default_body: BodyModel) -> None:
        _dyn, _qs, _qe, _qb, via = make_sit_to_stand_config(default_body)
        assert len(via) == 6

    def test_sts_via_fractions_span_0_to_1(self, default_body: BodyModel) -> None:
        _dyn, _qs, _qe, _qb, via = make_sit_to_stand_config(default_body)
        assert via[0][0] == 0.0
        assert via[-1][0] == 1.0

    def test_sts_zero_load(self, default_body: BodyModel) -> None:
        dyn, _qs, _qe, _qb, _via = make_sit_to_stand_config(default_body)
        assert dyn.m_load == 0.0

    def test_sts_rejects_bad_seat_height(self, default_body: BodyModel) -> None:
        with pytest.raises(ValueError, match="seat_height"):
            make_sit_to_stand_config(default_body, seat_height=0.0)

    def test_sts_forward_lean_phase(self, default_body: BodyModel) -> None:
        """The forward lean phase should have hip angle > start hip angle."""
        _dyn, _qs, _qe, _qb, via = make_sit_to_stand_config(default_body)
        # via[1] is forward lean; hip is index 3
        lean_hip = via[1][3]
        start_hip = via[0][3]
        assert lean_hip > start_hip, "Forward lean should increase hip flexion"


# ------------------------------------------------------------------
# Optimization smoke tests (structure only -- no full solve)
# ------------------------------------------------------------------


class TestGaitOptimizationSmoke:
    """Verify gait config can be wired into the optimizer without errors."""

    def test_gait_optimizer_construction(self, default_body: BodyModel) -> None:
        from movement_optimizer.trajectory import TrajectoryOptimizer

        dyn, qs, qe, qb, _via = make_gait_config(default_body)
        opt = TrajectoryOptimizer(
            default_body, dyn, "gait", 0.0, qs, qe, qb, duration=1.0, n_waypoints=8
        )
        assert opt.n_dof == 3
        assert opt.duration == 1.0

    def test_gait_initial_guess_shape(self, default_body: BodyModel) -> None:
        from movement_optimizer.trajectory import TrajectoryOptimizer

        dyn, qs, qe, qb, _via = make_gait_config(default_body)
        opt = TrajectoryOptimizer(
            default_body, dyn, "gait", 0.0, qs, qe, qb, duration=1.0, n_waypoints=8
        )
        guess = opt._initial_guess()
        assert guess.shape == (8, 3)

    def test_gait_cost_is_finite(self, default_body: BodyModel) -> None:
        from movement_optimizer.trajectory import TrajectoryOptimizer

        dyn, qs, qe, qb, _via = make_gait_config(default_body)
        opt = TrajectoryOptimizer(
            default_body, dyn, "gait", 0.0, qs, qe, qb, duration=1.0, n_waypoints=8
        )
        cost = opt._compute_cost(opt._initial_guess().flatten())
        assert np.isfinite(cost)


class TestSitToStandOptimizationSmoke:
    """Verify STS config can be wired into the optimizer without errors."""

    def test_sts_optimizer_construction(self, default_body: BodyModel) -> None:
        from movement_optimizer.trajectory import TrajectoryOptimizer

        dyn, qs, qe, qb, _via = make_sit_to_stand_config(default_body)
        opt = TrajectoryOptimizer(
            default_body,
            dyn,
            "sit_to_stand",
            0.0,
            qs,
            qe,
            qb,
            duration=2.0,
            n_waypoints=8,
        )
        assert opt.n_dof == 3
        assert opt.duration == 2.0

    def test_sts_initial_guess_shape(self, default_body: BodyModel) -> None:
        from movement_optimizer.trajectory import TrajectoryOptimizer

        dyn, qs, qe, qb, _via = make_sit_to_stand_config(default_body)
        opt = TrajectoryOptimizer(
            default_body,
            dyn,
            "sit_to_stand",
            0.0,
            qs,
            qe,
            qb,
            duration=2.0,
            n_waypoints=8,
        )
        guess = opt._initial_guess()
        assert guess.shape == (8, 3)

    def test_sts_cost_is_finite(self, default_body: BodyModel) -> None:
        from movement_optimizer.trajectory import TrajectoryOptimizer

        dyn, qs, qe, qb, _via = make_sit_to_stand_config(default_body)
        opt = TrajectoryOptimizer(
            default_body,
            dyn,
            "sit_to_stand",
            0.0,
            qs,
            qe,
            qb,
            duration=2.0,
            n_waypoints=8,
        )
        cost = opt._compute_cost(opt._initial_guess().flatten())
        assert np.isfinite(cost)
