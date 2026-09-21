"""Tests for tour baseline calibration, fixed geometry, and initial states (TB-03 #10588)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.shared.python.tour_baselines.calibration import (
    GeometryCalibrationResult,
    InitialStateResult,
    MovingHubMotion,
    PlanarDoublePendulumPose,
    calibrate_fixed_geometry,
    compute_moving_hub_power,
    forward_kinematics_planar_double_pendulum,
    map_initial_state_double_pendulum,
)

pytestmark = pytest.mark.unit


def test_rigid_length_and_positive_inertia_constraints_hold() -> None:
    """Rigid link lengths and physical inertia parameters must satisfy positive bounds."""
    # Synthetic arm and club observations
    n_frames = 50
    shoulder = np.zeros((n_frames, 3))
    # Grip at nominal distance 0.65m
    grip = np.zeros((n_frames, 3))
    grip[:, 0] = 0.65
    # Clubhead at nominal grip-to-head distance 1.05m
    head = np.zeros((n_frames, 3))
    head[:, 0] = 1.70

    calib = calibrate_fixed_geometry(
        shoulder_pts=shoulder,
        grip_pts=grip,
        clubhead_pts=head,
    )

    assert isinstance(calib, GeometryCalibrationResult)
    assert 0.4 <= calib.l1_arm_m <= 0.9
    assert 0.7 <= calib.l2_club_m <= 1.3
    assert calib.m1_arm_kg > 0.0
    assert calib.m2_shaft_kg > 0.0
    assert calib.m_head_kg > 0.0
    assert calib.i1_arm_kg_m2 > 0.0
    assert calib.is_inertia_frozen is True
    # Sensitivity matrix rank diagnostic must report non-identifiable inertia
    assert calib.identifiability.rank < calib.identifiability.total_parameters
    assert calib.identifiability.has_unidentifiable_parameters is True


def test_absolute_and_relative_angle_fk_agrees_on_known_poses() -> None:
    """Forward kinematics agrees exactly with geometry across absolute/relative angle poses."""
    l1 = 0.65
    l2 = 1.00
    pivot = np.array([0.0, 0.0])

    # Pose 1: Straight down (theta1 = 0, theta2 = 0)
    pose1 = PlanarDoublePendulumPose(theta1_rad=0.0, theta2_rad=0.0)
    grip1, head1 = forward_kinematics_planar_double_pendulum(pivot, l1, l2, pose1)
    np.testing.assert_allclose(grip1, [0.0, -l1], atol=1e-7)
    np.testing.assert_allclose(head1, [0.0, -l1 - l2], atol=1e-7)

    # Pose 2: 90 degrees horizontal (theta1 = pi/2, theta2 = 0)
    pose2 = PlanarDoublePendulumPose(theta1_rad=math.pi / 2, theta2_rad=0.0)
    grip2, head2 = forward_kinematics_planar_double_pendulum(pivot, l1, l2, pose2)
    np.testing.assert_allclose(grip2, [l1, 0.0], atol=1e-7)
    np.testing.assert_allclose(head2, [l1 + l2, 0.0], atol=1e-7)

    # Pose 3: Cocked wrist (theta1 = 0, theta2 = pi/2)
    pose3 = PlanarDoublePendulumPose(theta1_rad=0.0, theta2_rad=math.pi / 2)
    grip3, head3 = forward_kinematics_planar_double_pendulum(pivot, l1, l2, pose3)
    np.testing.assert_allclose(grip3, [0.0, -l1], atol=1e-7)
    np.testing.assert_allclose(head3, [l2, -l1], atol=1e-7)


def test_map_initial_state_exact_round_trip() -> None:
    """Initial state mapping maps t0 observation to q0 such that FK(q0) = observation."""
    l1 = 0.68
    l2 = 1.02
    pivot = np.array([0.1, 0.2])

    target_theta1 = 0.35  # rad
    target_theta2 = 0.75  # rad
    pose = PlanarDoublePendulumPose(theta1_rad=target_theta1, theta2_rad=target_theta2)
    target_grip, target_head = forward_kinematics_planar_double_pendulum(
        pivot, l1, l2, pose
    )

    # Trajectory of 5 frames around t0
    times = np.array([0.0, 0.001, 0.002, 0.003, 0.004])
    grips = np.tile(target_grip, (5, 1))
    heads = np.tile(target_head, (5, 1))
    pivots = np.tile(pivot, (5, 1))

    init_state: InitialStateResult = map_initial_state_double_pendulum(
        times=times,
        pivot_pts=pivots,
        grip_pts=grips,
        clubhead_pts=heads,
        l1=l1,
        l2=l2,
        t0_idx=0,
    )

    assert abs(init_state.q0[0] - target_theta1) < 1e-6
    assert abs(init_state.q0[1] - target_theta2) < 1e-6
    # Exact FK verification: q(t0) = q0
    np.testing.assert_allclose(init_state.fk_grip_error_m, 0.0, atol=1e-6)
    np.testing.assert_allclose(init_state.fk_head_error_m, 0.0, atol=1e-6)


def test_velocity_estimation_does_not_cross_gaps_blindly() -> None:
    """Missing data/gaps at or immediately after t0 must raise an error rather than extrapolate."""
    times = np.array([0.0, 0.001, 0.002, 0.003])
    # Pivot and grip valid, but head has NaN at t0+1
    pivots = np.zeros((4, 2))
    grips = np.zeros((4, 2))
    heads = np.zeros((4, 2))
    heads[1] = [np.nan, np.nan]

    with pytest.raises(ValueError, match="window.*missing.*gap"):
        map_initial_state_double_pendulum(
            times=times,
            pivot_pts=pivots,
            grip_pts=grips,
            clubhead_pts=heads,
            l1=0.6,
            l2=1.0,
            t0_idx=0,
        )


def test_prescribed_moving_hub_records_motion_and_power_contribution() -> None:
    """A moving hub computes and saves external motion and non-zero power contribution."""
    times = np.linspace(0, 0.3, 301)
    dt = times[1] - times[0]
    # Linear motion of hub: x = 0.5 * t, y = 0
    hub_pos = np.column_stack([0.5 * times, np.zeros_like(times)])
    # Applied external reaction forces on the hub (e.g. constant 20 N in x)
    hub_forces = np.column_stack([np.full_like(times, 20.0), np.zeros_like(times)])

    hub_motion: MovingHubMotion = compute_moving_hub_power(
        times=times,
        hub_positions=hub_pos,
        hub_reaction_forces=hub_forces,
    )

    # Velocity should be 0.5 m/s in x
    np.testing.assert_allclose(hub_motion.velocity[:, 0], 0.5, atol=1e-4)
    # Power P = F . v = 20 * 0.5 = 10 W
    np.testing.assert_allclose(hub_motion.power_watts, 10.0, atol=1e-4)
    # Total external work W = P * T = 10 * 0.3 = 3.0 J
    assert abs(hub_motion.total_work_joules - 3.0) < 0.05
    assert hub_motion.is_moving_hub is True
