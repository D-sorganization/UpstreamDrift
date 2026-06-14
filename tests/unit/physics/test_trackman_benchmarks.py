"""TrackMan-style ball-flight benchmarks for aerodynamic calibration."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from src.shared.python.physics.aerodynamics import LiftModel
from src.shared.python.physics.ball_flight_physics import (
    BallFlightSimulator,
    EnhancedBallFlightSimulator,
    LaunchConditions,
)
from src.shared.python.physics.ball_properties import BallProperties
from src.shared.python.physics.rust_kernel import is_rust_available

pytestmark = [pytest.mark.unit, pytest.mark.scientific]

YARDS_TO_METERS = 0.9144


def _launch(
    ball_speed_mph: float, launch_deg: float, spin_rpm: float
) -> LaunchConditions:
    return LaunchConditions(
        velocity=ball_speed_mph * 0.44704,
        launch_angle=math.radians(launch_deg),
        spin_rate=spin_rpm,
        spin_axis=np.array([0.0, -1.0, 0.0]),
    )


def _analysis(
    simulator_factory: Callable[[], BallFlightSimulator | EnhancedBallFlightSimulator],
    launch: LaunchConditions,
) -> dict[str, float]:
    simulator = simulator_factory()
    trajectory = simulator.simulate_trajectory(launch, max_time=12.0, dt=0.01)
    return simulator.analyze_trajectory(trajectory)


def _simulator_factories() -> list[
    tuple[str, Callable[[], BallFlightSimulator | EnhancedBallFlightSimulator]]
]:
    factories: list[
        tuple[str, Callable[[], BallFlightSimulator | EnhancedBallFlightSimulator]]
    ] = [("enhanced", EnhancedBallFlightSimulator)]
    if is_rust_available():
        factories.append(("rust", BallFlightSimulator))
    return factories


@pytest.mark.parametrize(("name", "factory"), _simulator_factories())
def test_driver_trackman_window(
    name: str,
    factory: Callable[[], BallFlightSimulator | EnhancedBallFlightSimulator],
) -> None:
    analysis = _analysis(factory, _launch(167.0, 10.9, 2686.0))
    carry_yd = analysis["carry_distance"] / YARDS_TO_METERS

    assert 238.0 <= carry_yd <= 290.0, f"{name} carry={carry_yd:.1f} yd"
    assert 25.0 <= analysis["max_height"] <= 40.0, (
        f"{name} apex={analysis['max_height']:.1f} m"
    )
    assert 5.5 <= analysis["flight_time"] <= 7.5, (
        f"{name} time={analysis['flight_time']:.2f} s"
    )


@pytest.mark.parametrize(("name", "factory"), _simulator_factories())
def test_iron_trackman_window(
    name: str,
    factory: Callable[[], BallFlightSimulator | EnhancedBallFlightSimulator],
) -> None:
    analysis = _analysis(factory, _launch(120.0, 16.3, 7097.0))
    carry_yd = analysis["carry_distance"] / YARDS_TO_METERS

    assert 148.0 <= carry_yd <= 182.0, f"{name} carry={carry_yd:.1f} yd"
    assert 24.0 <= analysis["max_height"] <= 38.0, (
        f"{name} apex={analysis['max_height']:.1f} m"
    )


@pytest.mark.skipif(
    not is_rust_available(),
    reason="upstream-physics Rust kernel not available in this lane",
)
def test_rust_and_enhanced_engines_agree_on_trackman_shots() -> None:
    for launch in (
        _launch(167.0, 10.9, 2686.0),
        _launch(120.0, 16.3, 7097.0),
    ):
        rust = _analysis(BallFlightSimulator, launch)
        enhanced = _analysis(EnhancedBallFlightSimulator, launch)

        assert rust["carry_distance"] == pytest.approx(
            enhanced["carry_distance"], rel=0.05
        )
        assert rust["max_height"] == pytest.approx(enhanced["max_height"], rel=0.10)


def test_aero_coefficients_match_trackman_calibration_band() -> None:
    ball = BallProperties()
    lift = LiftModel()

    assert 0.08 <= ball.calculate_cl(0.08) <= 0.16
    assert 0.08 <= lift._compute_lift_coefficient(0.08) <= 0.16
    assert 0.18 <= ball.calculate_cl(0.30) <= 0.28
    assert 0.18 <= lift._compute_lift_coefficient(0.30) <= 0.28
    assert 0.23 <= ball.calculate_cd(0.08) <= 0.29
