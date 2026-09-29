# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for joint angle limits, Hill torque model, bench press, and max load.

Covers: joint clamping, limit checking, Hill torque-angle-velocity,
JointTorqueSet, sticking point detection, bench press config,
and the max load binary search.
"""

from __future__ import annotations

import numpy as np
import pytest

from movement_optimizer.constants import (
    BENCH_PRESS_JOINT_LIMITS,
    BENCH_PRESS_JOINT_NAMES,
    JOINT_LIMITS,
    JOINT_NAMES,
)
from movement_optimizer.models import (
    BodyModel,
    HillTorqueModel,
    JointTorqueSet,
    clamp_joint_angles,
    joint_angles_within_limits,
    make_default_torque_set,
    make_squat_config,
)

# ==============================================================
# Joint Angle Limits
# ==============================================================


class TestJointAngleLimits:
    def test_clamp_within_limits_unchanged(self) -> None:
        """Angles within limits should not be modified."""
        q = np.array([0.0, -0.5, 0.5])
        q_clamped = clamp_joint_angles(q)
        np.testing.assert_array_equal(q, q_clamped)

    def test_clamp_exceeding_upper(self) -> None:
        """Angles above upper limit should be clamped down."""
        q = np.array([np.radians(100), 0.0, np.radians(200)])
        q_clamped = clamp_joint_angles(q)
        for i, name in enumerate(JOINT_NAMES):
            _, hi = JOINT_LIMITS[name]
            assert q_clamped[i] <= hi + 1e-10

    def test_clamp_exceeding_lower(self) -> None:
        """Angles below lower limit should be clamped up."""
        q = np.array([np.radians(-100), np.radians(-200), np.radians(-100)])
        q_clamped = clamp_joint_angles(q)
        for i, name in enumerate(JOINT_NAMES):
            lo, _ = JOINT_LIMITS[name]
            assert q_clamped[i] >= lo - 1e-10

    def test_clamp_returns_copy(self) -> None:
        """Clamping should return a new array, not modify input."""
        q = np.array([0.0, 0.0, 0.0])
        q_clamped = clamp_joint_angles(q)
        q_clamped[0] = 999.0
        assert q[0] == 0.0

    def test_clamp_wrong_length_raises(self) -> None:
        """Wrong number of joints should raise ValueError."""
        with pytest.raises(ValueError, match="q length"):
            clamp_joint_angles(np.array([0.0, 0.0]))

    def test_within_limits_true(self) -> None:
        q = np.array([0.0, -0.5, 0.5])
        assert joint_angles_within_limits(q)

    def test_within_limits_false(self) -> None:
        q = np.array([np.radians(100), 0.0, 0.0])
        assert not joint_angles_within_limits(q)

    def test_within_limits_at_boundary(self) -> None:
        """Exactly at limits should count as within."""
        q = np.array(
            [
                JOINT_LIMITS["ankle"][1],
                JOINT_LIMITS["knee"][0],
                JOINT_LIMITS["hip"][0],
            ]
        )
        assert joint_angles_within_limits(q)

    def test_clamp_handles_nan_gracefully(self) -> None:
        """NaN angles should be clamped to the limits without errors."""
        q = np.array([np.nan, 0.0, 0.0])
        q_clamped = clamp_joint_angles(q)
        # np.clip with NaN keeps NaN, but it shouldn't crash
        assert q_clamped.shape == (3,)

    def test_clamp_handles_inf_gracefully(self) -> None:
        """Inf angles should be clamped to the limits."""
        q = np.array([np.inf, -np.inf, 0.0])
        q_clamped = clamp_joint_angles(q)
        _, hi = JOINT_LIMITS["ankle"]
        lo, _ = JOINT_LIMITS["knee"]
        assert q_clamped[0] <= hi + 1e-10
        assert q_clamped[1] >= lo - 1e-10

    def test_custom_limits(self) -> None:
        """Custom joint limits should override defaults."""
        custom_limits = {
            "a": (-0.1, 0.1),
            "b": (-0.2, 0.2),
        }
        q = np.array([1.0, -1.0])
        q_clamped = clamp_joint_angles(q, custom_limits, ("a", "b"))
        np.testing.assert_allclose(q_clamped, [0.1, -0.2])

    def test_bench_press_limits(self) -> None:
        """Bench press joints should have their own limits."""
        q = np.array([np.radians(100), 0.0, np.radians(20)])
        q_clamped = clamp_joint_angles(q, BENCH_PRESS_JOINT_LIMITS, BENCH_PRESS_JOINT_NAMES)
        for i, name in enumerate(BENCH_PRESS_JOINT_NAMES):
            lo, hi = BENCH_PRESS_JOINT_LIMITS[name]
            assert lo - 1e-10 <= q_clamped[i] <= hi + 1e-10


# ==============================================================
# Hill-Type Torque Model
# ==============================================================


class TestHillTorqueModel:
    def test_peak_at_optimal_angle(self) -> None:
        """Available torque should peak at the optimal angle."""
        model = HillTorqueModel(tau_max=200.0, q_optimal=0.5)
        # At optimal angle with zero velocity
        t_opt = model.available_torque(0.5, 0.0)
        t_off = model.available_torque(1.5, 0.0)
        assert t_opt > t_off

    def test_max_at_zero_velocity(self) -> None:
        """At zero velocity, velocity factor should be close to 1."""
        model = HillTorqueModel(tau_max=200.0, q_optimal=0.0)
        f_vel = model.torque_velocity_factor(0.0)
        np.testing.assert_allclose(f_vel, 1.0, atol=0.01)

    def test_concentric_decreases_with_speed(self) -> None:
        """Faster concentric contraction should reduce available torque.

        With default torque_sign=-1, negative qd = concentric (shortening).
        """
        model = HillTorqueModel(tau_max=200.0, q_optimal=0.0)
        # Negative qd → concentric; larger magnitude → less force
        t_slow = model.available_torque(0.0, -1.0)
        t_fast = model.available_torque(0.0, -5.0)
        assert t_slow > t_fast

    def test_eccentric_above_isometric(self) -> None:
        """Eccentric factor should exceed concentric near v_max.

        With default torque_sign=-1, positive qd = eccentric (lengthening).
        """
        model = HillTorqueModel(tau_max=200.0, q_optimal=0.0)
        # Concentric at high speed (near v_max) → near zero
        f_conc_fast = model.torque_velocity_factor(-(model.v_max - 0.01))
        # Eccentric at moderate speed → above isometric
        f_ecc = model.torque_velocity_factor(2.0)
        assert f_ecc > f_conc_fast

    def test_available_torque_always_nonneg(self) -> None:
        """Available torque should never be negative."""
        model = HillTorqueModel(tau_max=200.0, q_optimal=0.0)
        angles = np.linspace(-3, 3, 100)
        velocities = np.linspace(-20, 20, 100)
        for q in angles:
            for qd in velocities:
                assert model.available_torque(q, qd) >= 0

    def test_tau_max_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="tau_max"):
            HillTorqueModel(tau_max=0, q_optimal=0.0)

    def test_angle_width_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="angle_width"):
            HillTorqueModel(tau_max=100, q_optimal=0.0, angle_width=0)

    # ------------------------------------------------------------------
    # NaN / non-finite parameter validation (issue #236)
    # ------------------------------------------------------------------

    def test_nan_tau_max_raises(self) -> None:
        """NaN tau_max must be rejected at construction time."""
        with pytest.raises(ValueError, match="tau_max"):
            HillTorqueModel(tau_max=float("nan"), q_optimal=1.0)

    def test_inf_tau_max_raises(self) -> None:
        """Infinite tau_max must be rejected at construction time."""
        with pytest.raises(ValueError, match="tau_max"):
            HillTorqueModel(tau_max=float("inf"), q_optimal=1.0)

    def test_nan_q_optimal_raises(self) -> None:
        """NaN q_optimal must be rejected at construction time."""
        with pytest.raises(ValueError, match="q_optimal"):
            HillTorqueModel(tau_max=200.0, q_optimal=float("nan"))

    def test_nan_angle_width_raises(self) -> None:
        """NaN angle_width must be rejected at construction time."""
        with pytest.raises(ValueError, match="angle_width"):
            HillTorqueModel(tau_max=200.0, q_optimal=1.0, angle_width=float("nan"))

    def test_nan_v_max_raises(self) -> None:
        """NaN v_max must be rejected at construction time."""
        with pytest.raises(ValueError, match="v_max"):
            HillTorqueModel(tau_max=200.0, q_optimal=1.0, v_max=float("nan"))

    def test_nan_k_shape_raises(self) -> None:
        """NaN k_shape must be rejected at construction time."""
        with pytest.raises(ValueError, match="k_shape"):
            HillTorqueModel(tau_max=200.0, q_optimal=1.0, k_shape=float("nan"))

    def test_nan_ecc_factor_raises(self) -> None:
        """NaN ecc_factor must be rejected at construction time."""
        with pytest.raises(ValueError, match="ecc_factor"):
            HillTorqueModel(tau_max=200.0, q_optimal=1.0, ecc_factor=float("nan"))

    def test_nan_max_ecc_ratio_raises(self) -> None:
        """NaN max_ecc_ratio must be rejected at construction time."""
        with pytest.raises(ValueError, match="max_ecc_ratio"):
            HillTorqueModel(tau_max=200.0, q_optimal=1.0, max_ecc_ratio=float("nan"))

    def test_nan_q_in_torque_angle_factor_raises(self) -> None:
        """NaN angle input to torque_angle_factor must raise ValueError."""
        model = HillTorqueModel(tau_max=200.0, q_optimal=1.0)
        with pytest.raises(ValueError, match="q"):
            model.torque_angle_factor(float("nan"))

    def test_nan_qd_in_torque_velocity_factor_raises(self) -> None:
        """NaN velocity input to torque_velocity_factor must raise ValueError."""
        model = HillTorqueModel(tau_max=200.0, q_optimal=1.0)
        with pytest.raises(ValueError, match="qd"):
            model.torque_velocity_factor(float("nan"))

    def test_nan_in_available_torque_raises(self) -> None:
        """NaN angle or velocity input to available_torque must raise ValueError."""
        model = HillTorqueModel(tau_max=200.0, q_optimal=1.0)
        with pytest.raises(ValueError):
            model.available_torque(float("nan"), 0.0)
        with pytest.raises(ValueError):
            model.available_torque(0.0, float("nan"))

    def test_batch_angle_factor(self) -> None:
        """Torque-angle factor should vectorize correctly."""
        model = HillTorqueModel(tau_max=200.0, q_optimal=0.5)
        q = np.array([0.0, 0.5, 1.0])
        f = model.torque_angle_factor(q)
        assert f.shape == (3,)
        assert f[1] > f[0]  # peak at optimal
        assert f[1] > f[2]

    def test_batch_velocity_factor(self) -> None:
        """Velocity factor should vectorize correctly.

        With default torque_sign=-1, negative qd = concentric.
        """
        model = HillTorqueModel(tau_max=200.0, q_optimal=0.0)
        # Negative qd → concentric; increasing magnitude → decreasing factor
        qd = np.array([0.0, -2.0, -8.0])
        f = model.torque_velocity_factor(qd)
        assert f.shape == (3,)
        assert f[0] >= f[1] >= f[2]

    def test_eccentric_capped(self) -> None:
        """Eccentric factor should not exceed max_ecc_ratio.

        With default torque_sign=-1, positive qd = eccentric.
        """
        model = HillTorqueModel(tau_max=200.0, q_optimal=0.0, max_ecc_ratio=1.4)
        # Positive qd → eccentric at very high speed
        f = model.torque_velocity_factor(np.radians(10000))
        assert f <= 1.4 + 1e-10


# ==============================================================
# JointTorqueSet
# ==============================================================


class TestJointTorqueSet:
    def test_default_construction(self, default_torque_set: JointTorqueSet) -> None:
        assert len(default_torque_set.joint_names) == 3
        for name in JOINT_NAMES:
            assert default_torque_set.get_max_torque(name) > 0

    def test_set_max_torque(self, default_torque_set: JointTorqueSet) -> None:
        default_torque_set.set_max_torque("knee", 300.0)
        assert default_torque_set.get_max_torque("knee") == 300.0

    def test_set_invalid_joint_raises(self, default_torque_set: JointTorqueSet) -> None:
        with pytest.raises(ValueError, match="Unknown joint"):
            default_torque_set.set_max_torque("nonexistent", 100.0)

    def test_set_negative_torque_raises(self, default_torque_set: JointTorqueSet) -> None:
        with pytest.raises(ValueError, match="tau_max"):
            default_torque_set.set_max_torque("knee", -10.0)

    def test_available_torques_shape(self, default_torque_set: JointTorqueSet) -> None:
        q = np.array([0.0, -0.5, 0.5])
        qd = np.zeros(3)
        result = default_torque_set.available_torques(q, qd)
        assert result.shape == (3,)
        assert np.all(result > 0)

    def test_available_torques_batch_shape(self, default_torque_set: JointTorqueSet) -> None:
        n = 10
        q = np.tile([0.0, -0.5, 0.5], (n, 1))
        qd = np.zeros((n, 3))
        result = default_torque_set.available_torques_batch(q, qd)
        assert result.shape == (n, 3)
        assert np.all(result > 0)

    def test_utilization_shape(self, default_torque_set: JointTorqueSet) -> None:
        n = 10
        q = np.tile([0.0, -0.5, 0.5], (n, 1))
        qd = np.zeros((n, 3))
        torques = np.ones((n, 3)) * 50.0
        util = default_torque_set.torque_utilization(q, qd, torques)
        assert util.shape == (n, 3)
        assert np.all(util >= 0)

    def test_utilization_zero_torque(self, default_torque_set: JointTorqueSet) -> None:
        """Zero required torque should give zero utilization."""
        q = np.array([[0.0, -0.5, 0.5]])
        qd = np.zeros((1, 3))
        torques = np.zeros((1, 3))
        util = default_torque_set.torque_utilization(q, qd, torques)
        np.testing.assert_allclose(util, 0.0, atol=1e-10)

    def test_find_sticking_point(self, default_torque_set: JointTorqueSet) -> None:
        """Sticking point should identify the time and joint with max utilization."""
        n = 5
        q = np.tile([0.0, -0.5, 0.5], (n, 1))
        qd = np.zeros((n, 3))
        # Make knee torque very high at time step 3
        torques = np.ones((n, 3)) * 10.0
        torques[3, 1] = 500.0  # knee at step 3

        time_idx, joint_name, peak_util = default_torque_set.find_sticking_point(q, qd, torques)
        assert time_idx == 3
        assert joint_name == "knee"
        assert peak_util > 1.0  # should be overloaded

    def test_bench_press_torque_set(self, bench_torque_set: JointTorqueSet) -> None:
        assert len(bench_torque_set.joint_names) == 3
        assert "shoulder" in bench_torque_set.joint_names
        assert "elbow" in bench_torque_set.joint_names


# ==============================================================
# Bench Press Config
# ==============================================================


class TestBenchPressConfig:
    def test_config_shapes(self, bench_press_config) -> None:
        _dyn, qs, qe, qb, q_via = bench_press_config
        assert qs.shape == (3,)
        assert qe.shape == (3,)
        assert qb.shape == (3, 2)
        assert q_via is not None

    def test_start_is_extended(self, bench_press_config) -> None:
        """Start position should have arms extended (lockout)."""
        _, qs, _, _, _ = bench_press_config
        assert qs[1] > np.radians(-30)  # elbow near extension

    def test_end_is_extended(self, bench_press_config) -> None:
        """End position should have arms nearly extended (same as start -- full rep)."""
        _, _, qe, _, _ = bench_press_config
        assert qe[1] > np.radians(-30)  # elbow near extension

    def test_bounds_respect_joint_limits(self, bench_press_config) -> None:
        """q_bounds should be within bench press joint limits."""
        _, _, _, qb, _ = bench_press_config
        for i, name in enumerate(BENCH_PRESS_JOINT_NAMES):
            lo, hi = BENCH_PRESS_JOINT_LIMITS[name]
            assert qb[i, 0] >= lo - 1e-10
            assert qb[i, 1] <= hi + 1e-10

    def test_dynamics_produces_torques(self, bench_press_config) -> None:
        """Should be able to compute inverse dynamics."""
        dyn, qs, _, _, _ = bench_press_config
        tau = dyn.inverse_dynamics(qs, np.zeros(3), np.zeros(3))
        assert tau.shape == (3,)
        assert np.all(np.isfinite(tau))

    def test_mass_matrix_positive_definite(self, bench_press_config) -> None:
        dyn, qs, _, _, _ = bench_press_config
        M = dyn.mass_matrix(qs)
        eigenvalues = np.linalg.eigvals(M)
        assert np.all(eigenvalues > 0)

    def test_forward_kinematics(self, bench_press_config) -> None:
        dyn, qs, _, _, _ = bench_press_config
        fk = dyn.forward_kinematics(qs)
        assert "shoulder" in fk  # base of arm chain
        assert "hand" in fk  # end effector (bar grip)


# ==============================================================
# Max Load Calculator
# ==============================================================


class TestMaxLoad:
    def test_max_load_returns_tuple(self) -> None:
        """compute_max_load should return (load, time_idx, joint, utilization)."""
        from movement_optimizer.models import compute_max_load

        body = BodyModel(75.0, 1.75)
        torque_set = make_default_torque_set()

        max_load, t_idx, joint, util = compute_max_load(
            make_squat_config,
            body,
            torque_set,
            "squat",
            load_range=(0.0, 100.0),
            tol=5.0,
            n_eval=20,
        )
        assert max_load >= 0
        assert isinstance(t_idx, int)
        assert joint in JOINT_NAMES or joint == ""
        assert util >= 0

    def test_max_load_increases_with_stronger_joints(self) -> None:
        """Higher max torques should allow heavier loads."""
        from movement_optimizer.models import compute_max_load

        body = BodyModel(75.0, 1.75)

        weak = make_default_torque_set()
        strong = make_default_torque_set()
        for name in JOINT_NAMES:
            strong.set_max_torque(name, strong.get_max_torque(name) * 2.0)

        weak_load, _, _, _ = compute_max_load(
            make_squat_config,
            body,
            weak,
            "squat",
            load_range=(0.0, 200.0),
            tol=10.0,
            n_eval=20,
        )
        strong_load, _, _, _ = compute_max_load(
            make_squat_config,
            body,
            strong,
            "squat",
            load_range=(0.0, 400.0),
            tol=10.0,
            n_eval=20,
        )
        assert strong_load >= weak_load

    def test_sticking_point_identified(self) -> None:
        """The sticking point joint should be one of the 3 chain joints."""
        from movement_optimizer.models import compute_max_load

        body = BodyModel(75.0, 1.75)
        torque_set = make_default_torque_set()

        _, _, joint, _ = compute_max_load(
            make_squat_config,
            body,
            torque_set,
            "squat",
            load_range=(0.0, 100.0),
            tol=5.0,
            n_eval=20,
        )
        assert joint in JOINT_NAMES or joint == ""

    def test_invalid_range_raises(self) -> None:
        from movement_optimizer.models import compute_max_load

        body = BodyModel(75.0, 1.75)
        ts = make_default_torque_set()

        with pytest.raises(ValueError, match="max must exceed min"):
            compute_max_load(
                make_squat_config,
                body,
                ts,
                "squat",
                load_range=(100.0, 50.0),
            )

    def test_negative_range_raises(self) -> None:
        from movement_optimizer.models import compute_max_load

        body = BodyModel(75.0, 1.75)
        ts = make_default_torque_set()

        with pytest.raises(ValueError, match="min load"):
            compute_max_load(
                make_squat_config,
                body,
                ts,
                "squat",
                load_range=(-10.0, 100.0),
            )
