"""TrackMan-style ball-flight benchmarks for aerodynamic calibration."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.api.routes.ball_flight import router as ball_flight_router
from src.shared.python.physics.aerodynamics import (
    AerodynamicsConfig,
    DragModel,
    LiftModel,
)
from src.shared.python.physics.atmosphere import cd_dimpled_sphere
from src.shared.python.physics.ball_flight_physics import (
    BallFlightSimulator,
    EnhancedBallFlightSimulator,
    EnvironmentalConditions,
    LaunchConditions,
)
from src.shared.python.physics.ball_properties import BallProperties
from src.shared.python.physics.flight_models import (
    FlightModelRegistry,
    FlightModelType,
    FlightResult,
    UnifiedLaunchConditions,
)
from src.shared.python.physics.rust_kernel import is_rust_available

pytestmark = [pytest.mark.unit, pytest.mark.scientific]

YARDS_TO_METERS = 0.9144
GRAVITY = 9.80665


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


def _registry_launch(
    ball_speed_mph: float,
    launch_deg: float,
    spin_rpm: float,
    wind_speed_mps: float = 0.0,
    wind_direction_deg: float = 0.0,
) -> UnifiedLaunchConditions:
    return UnifiedLaunchConditions.from_imperial(
        ball_speed_mph=ball_speed_mph,
        launch_angle_deg=launch_deg,
        spin_rate_rpm=spin_rpm,
        wind_speed_mph=wind_speed_mps / 0.44704,
        wind_direction_deg=wind_direction_deg,
    )


def _registry_result(
    model_type: FlightModelType,
    launch: UnifiedLaunchConditions,
) -> FlightResult:
    return FlightModelRegistry.get_model(model_type).simulate(
        launch, max_time=12.0, dt=0.01
    )


def _simulator_factories() -> (
    list[tuple[str, Callable[[], BallFlightSimulator | EnhancedBallFlightSimulator]]]
):
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
    assert (
        25.0 <= analysis["max_height"] <= 40.0
    ), f"{name} apex={analysis['max_height']:.1f} m"
    assert (
        5.5 <= analysis["flight_time"] <= 7.5
    ), f"{name} time={analysis['flight_time']:.2f} s"


@pytest.mark.parametrize(("name", "factory"), _simulator_factories())
def test_iron_trackman_window(
    name: str,
    factory: Callable[[], BallFlightSimulator | EnhancedBallFlightSimulator],
) -> None:
    analysis = _analysis(factory, _launch(120.0, 16.3, 7097.0))
    carry_yd = analysis["carry_distance"] / YARDS_TO_METERS

    assert 148.0 <= carry_yd <= 182.0, f"{name} carry={carry_yd:.1f} yd"
    assert (
        24.0 <= analysis["max_height"] <= 38.0
    ), f"{name} apex={analysis['max_height']:.1f} m"


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


@pytest.mark.parametrize(
    ("shot", "launch", "carry_window_yd"),
    [
        ("driver", _registry_launch(167.0, 10.9, 2686.0), (250.0, 300.0)),
        ("7-iron", _registry_launch(120.0, 16.3, 7097.0), (165.0, 190.0)),
    ],
)
def test_registered_flight_models_match_trackman_carry_band(
    shot: str,
    launch: UnifiedLaunchConditions,
    carry_window_yd: tuple[float, float],
) -> None:
    carries: list[float] = []
    for model_type in FlightModelType:
        result = _registry_result(model_type, launch)
        carry_yd = result.carry_distance / YARDS_TO_METERS
        carries.append(carry_yd)
        assert (
            carry_window_yd[0] <= carry_yd <= carry_window_yd[1]
        ), f"{model_type.value} {shot} carry={carry_yd:.1f} yd"
        assert result.max_height > 0.0
        assert result.flight_time > 0.0

    mean_carry = sum(carries) / len(carries)
    for carry_yd in carries:
        assert carry_yd == pytest.approx(mean_carry, rel=0.10)


def test_rest_route_matches_trackman_driver_benchmark() -> None:
    app = FastAPI()
    app.include_router(ball_flight_router)
    client = TestClient(app)

    response = client.post(
        "/tools/ball-flight/simulate",
        json={
            "ball_speed_mps": 167.0 * 0.44704,
            "launch_angle_deg": 10.9,
            "spin_rate_rpm": 2686.0,
            "model_name": FlightModelType.WATERLOO_PENNER.value,
            "max_time_s": 12.0,
            "time_step_s": 0.01,
        },
    )

    assert response.status_code == 200
    summary = response.json()["summary"]
    carry_yd = summary["carry_m"] / YARDS_TO_METERS
    assert 250.0 <= carry_yd <= 300.0
    assert 24.0 <= summary["apex_m"] <= 37.0
    assert 5.5 <= summary["flight_time_s"] <= 7.5


def test_vacuum_trajectory_uses_air_density_not_drag_coefficient_clamp() -> None:
    launch = _launch(30.0 / 0.44704, 45.0, 0.0)
    environment = EnvironmentalConditions(air_density=0.0)
    simulator = EnhancedBallFlightSimulator(
        environment=environment,
        aero_config=AerodynamicsConfig(enabled=False),
    )

    trajectory = simulator.simulate_trajectory(launch, max_time=8.0, dt=0.002)
    analysis = simulator.analyze_trajectory(trajectory)
    expected_range = launch.velocity**2 * math.sin(2.0 * launch.launch_angle) / GRAVITY

    assert analysis["carry_distance"] == pytest.approx(expected_range, rel=0.005)
    assert cd_dimpled_sphere(1.0e5, base_cd=0.0) == 0.0
    np.testing.assert_allclose(
        DragModel(base_coefficient=0.0).calculate(np.array([50.0, 0.0, 0.0])),
        np.zeros(3),
    )
    with pytest.raises(ValueError):
        DragModel(base_coefficient=-0.01)


def test_five_meter_per_second_wind_changes_driver_carry_sensibly() -> None:
    calm = _registry_result(
        FlightModelType.WATERLOO_PENNER, _registry_launch(167.0, 10.9, 2686.0)
    )
    headwind = _registry_result(
        FlightModelType.WATERLOO_PENNER,
        _registry_launch(167.0, 10.9, 2686.0, wind_speed_mps=5.0),
    )
    tailwind = _registry_result(
        FlightModelType.WATERLOO_PENNER,
        _registry_launch(
            167.0, 10.9, 2686.0, wind_speed_mps=5.0, wind_direction_deg=180.0
        ),
    )

    headwind_loss_yd = (calm.carry_distance - headwind.carry_distance) / YARDS_TO_METERS
    tailwind_gain_yd = (tailwind.carry_distance - calm.carry_distance) / YARDS_TO_METERS
    assert 9.0 <= headwind_loss_yd <= 18.0
    assert 7.0 <= tailwind_gain_yd <= 14.0


def test_humid_air_is_less_dense_than_dry_air() -> None:
    dry = EnvironmentalConditions.from_altitude(
        altitude_m=0.0, temperature_c=30.0, relative_humidity=0.0
    )
    saturated = EnvironmentalConditions.from_altitude(
        altitude_m=0.0, temperature_c=30.0, relative_humidity=1.0
    )
    reduction = (dry.air_density - saturated.air_density) / dry.air_density

    assert saturated.air_density < dry.air_density
    assert 0.01 <= reduction <= 0.02
