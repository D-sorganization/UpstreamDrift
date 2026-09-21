"""Tests for geometry calibration, initial state mapping, and hub conditions (TB-03 #10588)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.shared.python.tour_baselines.geometry_calibration import (
    CalibratedGeometry,
    calibrate_link_geometry,
)
from src.shared.python.tour_baselines.hub_conditions import (
    FixedPivotHub,
    MovingHub,
)
from src.shared.python.tour_baselines.initial_state import (
    InitialStateMapping,
    estimate_initial_state,
)

pytestmark = pytest.mark.unit


def test_calibrate_link_geometry_bounds_and_rigidity() -> None:
    """Calibrate positive bounded link lengths from trajectories with rigidity diagnostics."""
    n_frames = 100
    # Hub at origin
    hub = np.zeros((n_frames, 3))
    # Grip at 0.65m from hub
    arm_length_true = 0.65
    angles1 = np.linspace(0.0, math.pi / 2, n_frames)
    grip = np.column_stack(
        [
            arm_length_true * np.sin(angles1),
            -arm_length_true * np.cos(angles1),
            np.zeros(n_frames),
        ]
    )
    # Clubhead at 1.1m from grip
    club_length_true = 1.10
    angles2 = angles1 + 0.3
    clubhead = grip + np.column_stack(
        [
            club_length_true * np.sin(angles2),
            -club_length_true * np.cos(angles2),
            np.zeros(n_frames),
        ]
    )

    geom = calibrate_link_geometry(hub_traj=hub, grip_traj=grip, head_traj=clubhead)

    assert isinstance(geom, CalibratedGeometry)
    assert abs(geom.arm_length_m - arm_length_true) < 1e-4
    assert abs(geom.club_length_m - club_length_true) < 1e-4
    assert geom.arm_length_std_m < 1e-6  # rigid motion
    assert geom.club_length_std_m < 1e-6
    assert geom.is_mass_frozen is True
    assert geom.is_inertia_frozen is True
    assert geom.shaft_mass_kg > 0.0
    assert geom.clubhead_mass_kg > 0.0


def test_calibrate_link_geometry_rejects_out_of_bounds() -> None:
    """Non-physical link lengths (e.g. negative or outside physiological bounds) raise ValueError."""
    # Negative / zero length
    hub = np.zeros((10, 3))
    grip = np.zeros((10, 3))  # 0 length
    head = np.ones((10, 3))

    with pytest.raises(ValueError, match="Arm length"):
        calibrate_link_geometry(hub_traj=hub, grip_traj=grip, head_traj=head)


def test_initial_state_mapping_and_forward_kinematics_agreement() -> None:
    """Initial state angles map into double pendulum FK conventions and agree on initial pose."""
    arm_len = 0.60
    club_len = 1.05

    # Known state: theta1 = 30 deg (0.5236 rad), wrist theta2 = 45 deg (0.7854 rad)
    th1 = math.radians(30.0)
    th2 = math.radians(45.0)

    # In-plane coordinates: downward vertical is -V
    # Hub at (0, 0)
    p_hub_2d = np.array([0.0, 0.0])
    p_grip_2d = p_hub_2d + np.array([arm_len * math.sin(th1), -arm_len * math.cos(th1)])
    th_abs_club = th1 + th2
    p_head_2d = p_grip_2d + np.array(
        [club_len * math.sin(th_abs_club), -club_len * math.cos(th_abs_club)]
    )

    state = estimate_initial_state(
        p_hub_2d=p_hub_2d,
        p_grip_2d=p_grip_2d,
        p_head_2d=p_head_2d,
        arm_length_m=arm_len,
        club_length_m=club_len,
    )

    assert isinstance(state, InitialStateMapping)
    assert abs(state.theta1_rad - th1) < 1e-6
    assert abs(state.theta2_rad - th2) < 1e-6
    assert abs(state.theta_club_abs_rad - th_abs_club) < 1e-6

    # Forward Kinematics verification
    fk_grip, fk_head = state.compute_fk(p_hub_2d, arm_len, club_len)
    np.testing.assert_allclose(fk_grip, p_grip_2d, atol=1e-6)
    np.testing.assert_allclose(fk_head, p_head_2d, atol=1e-6)
    assert state.closure_residual_m < 1e-6


def test_velocity_estimation_avoids_nan_gaps() -> None:
    """Velocities are estimated from declared contiguous windows without crossing NaN gaps."""
    times = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
    # Angle trajectory with velocity = 10 rad/s
    thetas = 10.0 * times
    thetas_with_nan = thetas.copy()
    thetas_with_nan[3] = np.nan  # Gap at index 3

    state_clean = estimate_initial_state(
        p_hub_2d=np.array([0.0, 0.0]),
        p_grip_2d=np.array([0.0, -0.6]),
        p_head_2d=np.array([0.0, -1.6]),
        arm_length_m=0.6,
        club_length_m=1.0,
        times=times,
        theta1_traj=thetas,
        theta2_traj=thetas,
        t0_idx=0,
    )
    assert abs(state_clean.omega1_rad_s - 10.0) < 1e-3

    # If t0 is adjacent to a NaN, estimation fails or raises ValueError
    with pytest.raises(ValueError, match="adjacent to missing / NaN data"):
        estimate_initial_state(
            p_hub_2d=np.array([0.0, 0.0]),
            p_grip_2d=np.array([0.0, -0.6]),
            p_head_2d=np.array([0.0, -1.6]),
            arm_length_m=0.6,
            club_length_m=1.0,
            times=times,
            theta1_traj=thetas_with_nan,
            theta2_traj=thetas_with_nan,
            t0_idx=2,  # t0_idx=2 is adjacent to index 3 (NaN)
        )


def test_fixed_pivot_vs_moving_hub_distinction() -> None:
    """Fixed pivot and moving hub conditions are distinct; moving hub records power contribution."""
    fixed = FixedPivotHub(pivot_world=np.array([0.0, 1.4, 0.0]))
    assert fixed.is_moving is False
    assert fixed.external_work_joules == 0.0

    times = np.linspace(0.0, 0.2, 21)
    # Hub moves along X with velocity 2.0 m/s
    hub_pos = np.column_stack(
        [2.0 * times, np.full_like(times, 1.4), np.zeros_like(times)]
    )
    # Applied force on hub
    hub_force = np.column_stack(
        [np.full_like(times, 50.0), np.zeros_like(times), np.zeros_like(times)]
    )

    moving = MovingHub.from_trajectory(times=times, positions=hub_pos, forces=hub_force)
    assert moving.is_moving is True
    # Power = F * v = 50 * 2 = 100 W; Work = integral(P dt) = 100 * 0.2 = 20 J
    assert moving.external_work_joules > 0.0
    assert abs(moving.external_work_joules - 20.0) < 0.5
