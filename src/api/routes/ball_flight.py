"""Ball-flight simulator API routes.

Provides a headless REST endpoint for batch and UI clients that need golf-ball
flight trajectories without launching the GUI.

See issue #7218.
"""

from __future__ import annotations

import math

from fastapi import APIRouter
from pydantic import BaseModel, Field, model_validator

from src.api.middleware.error_handler import handle_api_errors
from src.shared.python.physics.flight_models import (
    FlightModelRegistry,
    FlightModelType,
    UnifiedLaunchConditions,
)

router = APIRouter(prefix="/tools/ball-flight", tags=["ball-flight"])


class BallFlightSimulationRequest(BaseModel):
    """Request to simulate a single ball-flight trajectory."""

    ball_speed_mps: float = Field(
        70.0, description="Initial ball speed [m/s]", gt=0.0, le=100.0
    )
    launch_angle_deg: float = Field(
        12.0, description="Vertical launch angle [deg]", ge=-10.0, le=80.0
    )
    azimuth_angle_deg: float = Field(
        0.0, description="Horizontal launch azimuth [deg]", ge=-90.0, le=90.0
    )
    spin_rate_rpm: float = Field(
        2600.0, description="Initial spin rate [rpm]", ge=0.0, le=15000.0
    )
    spin_axis_tilt_deg: float = Field(
        0.0, description="Spin-axis tilt [deg]", ge=-90.0, le=90.0
    )
    wind_speed_mps: float = Field(0.0, description="Wind speed [m/s]", ge=0.0, le=40.0)
    wind_direction_deg: float = Field(
        0.0, description="Wind direction [deg]", ge=-180.0, le=180.0
    )
    model_name: FlightModelType = Field(
        FlightModelType.WATERLOO_PENNER,
        description="Ball-flight model identifier",
    )
    max_time_s: float = Field(
        10.0, description="Maximum simulation time [s]", gt=0.0, le=30.0
    )
    time_step_s: float = Field(
        0.01, description="Returned trajectory sample interval [s]", gt=0.0, le=0.25
    )

    @model_validator(mode="after")
    def validate_time_step(self) -> BallFlightSimulationRequest:
        """Keep time discretization bounded by the requested integration window."""
        if self.time_step_s > self.max_time_s:
            raise ValueError("time_step_s must be less than or equal to max_time_s")
        return self

    def to_launch_conditions(self) -> UnifiedLaunchConditions:
        """Convert validated API fields into the existing physics model contract."""
        return UnifiedLaunchConditions(
            ball_speed=self.ball_speed_mps,
            launch_angle=math.radians(self.launch_angle_deg),
            azimuth_angle=math.radians(self.azimuth_angle_deg),
            spin_rate=self.spin_rate_rpm,
            spin_axis_angle=math.radians(self.spin_axis_tilt_deg),
            wind_speed=self.wind_speed_mps,
            wind_direction=math.radians(self.wind_direction_deg),
        )


class BallFlightTrajectorySample(BaseModel):
    """Single sampled trajectory point."""

    time_s: float
    position_m: list[float]
    velocity_mps: list[float]


class BallFlightSummary(BaseModel):
    """Scalar trajectory metrics."""

    carry_m: float
    apex_m: float
    flight_time_s: float
    landing_angle_deg: float
    lateral_deviation_m: float


class BallFlightSimulationResponse(BaseModel):
    """Response containing ball-flight trajectory and summary metrics."""

    model_name: str
    model_key: FlightModelType
    trajectory: list[BallFlightTrajectorySample]
    summary: BallFlightSummary


def _trajectory_samples(result_trajectory: list) -> list[BallFlightTrajectorySample]:
    """Serialize existing trajectory points without changing physics behavior."""
    return [
        BallFlightTrajectorySample(
            time_s=float(point.time),
            position_m=[float(value) for value in point.position.tolist()],
            velocity_mps=[float(value) for value in point.velocity.tolist()],
        )
        for point in result_trajectory
    ]


@router.post("/simulate", response_model=BallFlightSimulationResponse)
@handle_api_errors
async def simulate_ball_flight(
    payload: BallFlightSimulationRequest,
) -> BallFlightSimulationResponse:
    """Simulate ball flight for headless and batch clients.

    See issue #7218.
    """
    model = FlightModelRegistry.get_model(payload.model_name)
    result = model.simulate(
        payload.to_launch_conditions(),
        max_time=payload.max_time_s,
        dt=payload.time_step_s,
    )

    return BallFlightSimulationResponse(
        model_name=result.model_name,
        model_key=payload.model_name,
        trajectory=_trajectory_samples(result.trajectory),
        summary=BallFlightSummary(
            carry_m=float(result.carry_distance),
            apex_m=float(result.max_height),
            flight_time_s=float(result.flight_time),
            landing_angle_deg=float(result.landing_angle),
            lateral_deviation_m=float(result.lateral_deviation),
        ),
    )
