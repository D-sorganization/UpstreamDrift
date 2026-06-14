"""Source-backed validation tests for golf ball-flight and impact assumptions."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from src.shared.python.core.contracts import ContractViolationError
from src.shared.python.core.physics_constants import (
    DRIVER_COR,
    GOLF_BALL_DIAMETER_M,
    GOLF_BALL_MASS_KG,
    SPIN_DECAY_RATE_S,
)
from src.shared.python.physics.ball_flight_physics import (
    MAX_LIFT_COEFFICIENT,
    BallFlightSimulator,
    BallProperties,
    LaunchConditions,
)
from src.shared.python.physics.flight_model_options import compute_spin_decay
from src.shared.python.physics.impact_model import (
    ImpactModelType,
    ImpactParameters,
    ImpactSolverAPI,
    PreImpactState,
    RigidBodyImpactModel,
)

pytestmark = pytest.mark.unit

SOURCE_MAP = Path("docs/physics/GOLF_BALL_FLIGHT_IMPACT_SOURCE_MAP.md")


def _driver_pre_impact(clubhead_speed: float = 45.0) -> PreImpactState:
    return PreImpactState(
        clubhead_velocity=np.array([clubhead_speed, 0.0, 0.0]),
        clubhead_angular_velocity=np.zeros(3),
        clubhead_orientation=np.array([1.0, 0.0, 0.0]),
        ball_position=np.array([0.05, 0.0, 0.0]),
        ball_velocity=np.zeros(3),
        ball_angular_velocity=np.zeros(3),
        clubhead_mass=0.2,
        clubhead_loft=math.radians(10.5),
        clubhead_lie=math.radians(60.0),
    )


def test_source_map_tracks_selected_hard_coded_golf_assumptions() -> None:
    text = SOURCE_MAP.read_text(encoding="utf-8")

    for required_phrase in (
        "Ball mass",
        "Ball diameter",
        "COR sanity",
        "smash factor",
        "Spin decay rate",
        "Illustrative assumption",
    ):
        assert required_phrase in text


def test_ball_properties_match_usga_ball_limits_and_coefficients_are_finite() -> None:
    ball = BallProperties()

    assert math.isclose(ball.mass, float(GOLF_BALL_MASS_KG), rel_tol=0.001)
    assert math.isclose(ball.diameter, float(GOLF_BALL_DIAMETER_M), rel_tol=0.001)

    for spin_parameter in (0.0, 0.05, 0.15, 0.30):
        cd = ball.calculate_cd(spin_parameter)
        cl = ball.calculate_cl(spin_parameter)

        assert math.isfinite(cd)
        assert math.isfinite(cl)
        assert cd > 0.0
        assert 0.0 <= cl <= MAX_LIFT_COEFFICIENT


def test_drag_lift_force_slice_is_finite_and_directionally_sane() -> None:
    simulator = BallFlightSimulator()
    launch = LaunchConditions(
        velocity=70.0,
        launch_angle=math.radians(11.0),
        spin_rate=2600.0,
        spin_axis=np.array([0.0, -1.0, 0.0]),
    )
    velocity = np.array([70.0, 0.0, 14.0])

    forces = simulator._calculate_forces(velocity, launch)

    for force in forces.values():
        assert np.all(np.isfinite(force))

    assert np.dot(forces["drag"], velocity) < 0.0
    assert abs(float(np.dot(forces["magnus"], velocity))) < 1e-10
    assert forces["gravity"][2] < 0.0


def test_spin_decay_is_finite_and_monotone_for_positive_decay_rate() -> None:
    omega_initial = 2600.0 * 2.0 * math.pi / 60.0
    values = [
        compute_spin_decay(omega_initial, time, float(SPIN_DECAY_RATE_S))
        for time in (0.0, 1.0, 2.0, 5.0)
    ]

    assert all(math.isfinite(value) for value in values)
    assert values == sorted(values, reverse=True)
    assert values[-1] < values[0]
    assert compute_spin_decay(omega_initial, 5.0, 0.0) == omega_initial


def test_rigid_impact_smash_factor_is_finite_and_driver_sane() -> None:
    pre_state = _driver_pre_impact()
    params = ImpactParameters(cor=min(float(DRIVER_COR), 0.83))

    post_state = RigidBodyImpactModel().solve(pre_state, params)
    club_speed = float(np.linalg.norm(pre_state.clubhead_velocity))
    ball_speed = float(np.linalg.norm(post_state.ball_velocity))
    smash_factor = ball_speed / club_speed

    assert np.all(np.isfinite(post_state.ball_velocity))
    assert np.all(np.isfinite(post_state.clubhead_velocity))
    assert 1.0 < smash_factor <= 1.52
    assert float(np.linalg.norm(post_state.clubhead_velocity)) < club_speed


def test_lofted_driver_impact_derives_lift_producing_spin_axis() -> None:
    """Pipeline-derived backspin should add upward Magnus force."""
    loft = math.radians(10.5)
    normal = np.array([math.cos(loft), 0.0, math.sin(loft)])
    post_state = ImpactSolverAPI(ImpactModelType.RIGID_BODY).solve_impact(
        timestamp=0.0,
        clubhead_velocity=np.array([50.5, 0.0, 0.0]),
        clubhead_orientation=normal,
        record=False,
    )
    spin_rate_rad_s = float(np.linalg.norm(post_state.ball_angular_velocity))
    spin_axis = post_state.ball_angular_velocity / spin_rate_rad_s
    ball_velocity = post_state.ball_velocity
    launch_speed = float(np.linalg.norm(ball_velocity))
    launch_angle = math.atan2(
        float(ball_velocity[2]),
        float(math.hypot(ball_velocity[0], ball_velocity[1])),
    )
    launch = LaunchConditions(
        velocity=launch_speed,
        launch_angle=launch_angle,
        spin_rate=spin_rate_rad_s * 60.0 / (2.0 * math.pi),
        spin_axis=spin_axis,
    )
    forces = BallFlightSimulator()._calculate_forces(
        ball_velocity,
        LaunchConditions(
            velocity=launch_speed,
            launch_angle=launch_angle,
            spin_rate=launch.spin_rate,
            spin_axis=spin_axis,
        ),
    )

    assert spin_axis[1] < 0.0
    assert forces["magnus"][2] > 0.0


def test_impossible_cor_and_negative_launch_speed_are_rejected() -> None:
    model = RigidBodyImpactModel()

    with pytest.raises((ContractViolationError, ValueError)):
        model.solve(_driver_pre_impact(), ImpactParameters(cor=1.01))

    with pytest.raises((ContractViolationError, ValueError)):
        BallFlightSimulator().simulate_trajectory(
            LaunchConditions(velocity=-1.0, launch_angle=math.radians(11.0)),
            max_time=0.1,
            dt=0.01,
        )
