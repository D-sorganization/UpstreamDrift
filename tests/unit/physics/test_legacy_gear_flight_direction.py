"""Check legacy toe/heel gear contribution against the actual flight frame."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from src.shared.python.physics.impact_model import (
    ImpactParameters,
    ImpactSolverAPI,
    PreImpactState,
    compute_gear_effect_spin,
)


@pytest.mark.unit
@pytest.mark.parametrize("offset", [0.00635, -0.00635])
@pytest.mark.parametrize("yaw_deg", [0.0, 25.0, -30.0])
def test_toe_gear_force_points_left_of_forward_heading(offset, yaw_deg):
    yaw = math.radians(yaw_deg)
    forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    left = np.cross([0.0, 0.0, 1.0], forward)
    spin = compute_gear_effect_spin(np.array([offset, 0.0]), 45 * forward, forward)
    # Native Magnus direction is omega cross velocity. Toe is left for RH,
    # heel is right; this tests the physical direction rather than its label.
    magnus_direction = np.cross(spin, forward)
    assert float(np.dot(magnus_direction, left)) * offset > 0


def _state(offset):
    loft = math.radians(12)
    return PreImpactState(
        clubhead_velocity=np.array([45.0, 0.0, 0.0]),
        clubhead_angular_velocity=np.zeros(3),
        clubhead_orientation=np.array([math.cos(loft), 0.0, math.sin(loft)]),
        ball_position=np.zeros(3),
        ball_velocity=np.zeros(3),
        ball_angular_velocity=np.zeros(3),
        clubhead_mass=0.2,
        impact_offset=np.array([offset, 0.0]),
    )


@pytest.mark.unit
@pytest.mark.parametrize("offset", [0.00635, -0.00635])
def test_public_solver_preserves_correct_gear_increment(offset):
    state = _state(offset)
    params = ImpactParameters()
    with_gear = ImpactSolverAPI(params=params).solve_pre_impact_state(
        0.0, state, record=False
    )
    without_gear = ImpactSolverAPI(
        params=replace(params, gear_effect_factor=0.0)
    ).solve_pre_impact_state(0.0, state, record=False)
    delta = with_gear.ball_angular_velocity - without_gear.ball_angular_velocity
    expected = compute_gear_effect_spin(
        state.impact_offset, state.clubhead_velocity, state.clubhead_orientation
    )
    np.testing.assert_allclose(delta, expected, atol=1e-12, rtol=0)
    assert delta[2] * offset > 0
    np.testing.assert_array_equal(with_gear.ball_velocity, without_gear.ball_velocity)


@pytest.mark.integration
def test_public_solver_to_native_flight_curves_toe_left_and_heel_right():
    pytest.importorskip("upstream_physics", reason="Native flight kernel unavailable")
    from src.shared.python.physics.ball_launch_conditions import LaunchConditions
    from src.shared.python.physics.ball_simulator import BallFlightSimulator

    lateral = []
    simulator = BallFlightSimulator()
    for offset in (-0.00635, 0.0, 0.00635):
        post = ImpactSolverAPI().solve_pre_impact_state(
            0.0, _state(offset), record=False
        )
        velocity = post.ball_velocity
        spin = post.ball_angular_velocity
        spin_norm = float(np.linalg.norm(spin))
        launch = LaunchConditions.from_user_units(
            velocity=float(np.linalg.norm(velocity)),
            launch_angle_deg=math.degrees(
                math.atan2(velocity[2], np.linalg.norm(velocity[:2]))
            ),
            azimuth_deg=math.degrees(math.atan2(velocity[1], velocity[0])),
            spin_rate_rpm=spin_norm * 60 / (2 * math.pi),
            spin_axis=spin / spin_norm,
        )
        landing = simulator.simulate_landing(launch, max_time=12.0, dt=0.01)
        lateral.append(float(landing[1]))
    assert lateral[0] < lateral[1] < lateral[2]
    assert abs(lateral[1]) < 1e-12
