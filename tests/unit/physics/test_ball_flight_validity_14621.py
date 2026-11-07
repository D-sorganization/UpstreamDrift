"""Flight model validity envelope for extreme sidespin (#14621)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.physics.ball_flight_validity import (
    MAX_DRIVER_LAUNCH_SPIN_RPM,
    assess_trajectory_validity,
    clamp_launch_spin_rate,
    landing_azimuth_deg,
)
from src.shared.python.physics.ball_launch_conditions import LaunchConditions
from src.shared.python.physics.ball_properties import BallProperties
from src.shared.python.physics.swing_ball_flight_pipeline import (
    SwingBallFlightPipeline,
    SwingState,
)
from src.shared.python.impact_parameters.target_frame import TargetFrame

pytestmark = pytest.mark.unit


def test_assess_trajectory_flags_down_range_reversal() -> None:
    """Non-physical reversal (SHOW-C7) must be flagged, not presented as carry."""
    path = np.array(
        [
            [0.0, 0.0, 10.0],
            [85.0, -5.0, 12.0],
            [83.0, -8.0, 11.0],
            [94.0, -40.0, 0.0],
        ],
        dtype=float,
    )
    verdict = assess_trajectory_validity(path)
    assert not verdict.valid
    assert "down-range reverses" in verdict.invalid_reason


def test_assess_trajectory_flags_extreme_landing_azimuth() -> None:
    path = np.array(
        [
            [0.0, 0.0, 5.0],
            [35.0, -25.0, 3.0],
            [50.0, -60.0, 0.0],
        ],
        dtype=float,
    )
    verdict = assess_trajectory_validity(path)
    assert not verdict.valid
    assert "deg off the target line" in verdict.invalid_reason


def test_clamp_launch_spin_preserves_axis() -> None:
    axis = np.array([0.1, -0.9, -0.4])
    axis /= np.linalg.norm(axis)
    launch = LaunchConditions(
        velocity=65.0,
        launch_angle=math.radians(12.0),
        azimuth_angle=0.0,
        spin_rate=6582.0,
        spin_axis=axis,
    )
    clamped, did_clamp = clamp_launch_spin_rate(launch)
    assert did_clamp
    assert clamped.spin_rate == pytest.approx(MAX_DRIVER_LAUNCH_SPIN_RPM)
    assert clamped.spin_axis == pytest.approx(axis, abs=1e-9)


def _c7_measured_swing_state() -> SwingState:
    """Face +21 deg, path +6.5 deg at ~110 mph (SHOW-C7 measured card)."""
    speed = 49.2
    face_open_deg = 21.0
    path_deg = 6.5
    loft_deg = 12.0
    loft, op = math.radians(loft_deg), math.radians(face_open_deg)
    normal = np.array(
        [
            -math.cos(loft) * math.sin(op),
            -math.cos(loft) * math.cos(op),
            math.sin(loft),
        ]
    )
    pa = math.radians(path_deg)
    vel_ud = np.array([-speed * math.sin(pa), -speed * math.cos(pa), 0.0])
    frame = TargetFrame()
    velocity = np.asarray(frame.components(vel_ud), dtype=float)
    normal_p = np.asarray(frame.components(normal), dtype=float)
    normal_p /= np.linalg.norm(normal_p)
    return SwingState(
        clubhead_velocity=velocity,
        clubhead_angular_velocity=np.zeros(3),
        clubhead_orientation=normal_p,
        clubhead_loft_deg=math.degrees(math.asin(float(normal_p[2]))),
        engine_name="test",
    )


def test_c7_face_21_path_65_pipeline_monotone_and_valid() -> None:
    """Regression: measured open-face driver stays inside validity after clamp."""
    result = SwingBallFlightPipeline().run(_c7_measured_swing_state())
    path = np.array([p.position for p in result.trajectory])
    assert float(np.min(np.diff(path[:, 0]))) >= -1e-6
    assert result.metadata["flight_valid"] is True
    assert result.metadata.get("launch_spin_clamped") is True
    assert result.launch_conditions.spin_rate == pytest.approx(
        MAX_DRIVER_LAUNCH_SPIN_RPM
    )
    assert landing_azimuth_deg(path) <= 45.0


def test_square_driver_strike_unclamped_and_valid() -> None:
    """Normal straight/draw/fade shots must stay valid without clamping."""
    loft_deg = 10.5
    loft = math.radians(loft_deg)
    normal = np.array([math.cos(loft), 0.0, math.sin(loft)])
    normal /= np.linalg.norm(normal)
    swing = SwingState(
        clubhead_velocity=np.array([49.2, 0.0, 0.0]),
        clubhead_angular_velocity=np.zeros(3),
        clubhead_orientation=normal,
        clubhead_loft_deg=loft_deg,
    )
    result = SwingBallFlightPipeline().run(swing)
    assert result.metadata["flight_valid"] is True
    assert not result.metadata.get("launch_spin_clamped")
    assert result.launch_conditions.spin_rate < MAX_DRIVER_LAUNCH_SPIN_RPM
    assert 200.0 < result.carry_m < 260.0


def test_pipeline_default_ball_uses_trackman_spin_decay() -> None:
    pipeline = SwingBallFlightPipeline()
    assert pipeline._ball.spin_decay_rate == pytest.approx(
        BallProperties().spin_decay_rate
    )
    assert pipeline._ball.spin_decay_rate > 0.0
