"""Putting Green Simulator API routes.

Provides REST endpoints for the putting green simulation tool page:
- Simulate putts with configurable parameters
- Read green contours and slope data
- Get aim-line assist calculations
- Scatter analysis for practice mode

See issue #1206
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/tools/putting-green", tags=["putting-green"])


# ── Request / Response Models ──


class PuttSimulationRequest(BaseModel):
    """Request to simulate a single putt."""

    ball_x: float = Field(5.0, description="Ball X position on green [m]")
    ball_y: float = Field(10.0, description="Ball Y position on green [m]")
    speed: float = Field(2.0, description="Stroke speed [m/s]", gt=0, le=10)
    direction_x: float = Field(0.0, description="Aim direction X component")
    direction_y: float = Field(1.0, description="Aim direction Y component")
    stimp_rating: float = Field(
        10.0, description="Green speed (Stimpmeter) [ft]", ge=6.0, le=15.0
    )
    green_width: float = Field(20.0, description="Green width [m]", gt=0)
    green_height: float = Field(20.0, description="Green height [m]", gt=0)
    hole_x: float = Field(10.0, description="Hole X position [m]")
    hole_y: float = Field(15.0, description="Hole Y position [m]")
    wind_speed: float = Field(0.0, description="Wind speed [m/s]", ge=0)
    wind_direction_x: float = Field(1.0, description="Wind direction X")
    wind_direction_y: float = Field(0.0, description="Wind direction Y")


class PuttSimulationResponse(BaseModel):
    """Response containing putt simulation results."""

    positions: list[list[float]]
    velocities: list[list[float]]
    times: list[float]
    holed: bool
    final_position: list[float]
    total_distance: float
    duration: float


class GreenReadingRequest(BaseModel):
    """Request for green reading between ball and target."""

    ball_x: float = Field(5.0, description="Ball X position [m]")
    ball_y: float = Field(5.0, description="Ball Y position [m]")
    target_x: float = Field(10.0, description="Target X position [m]")
    target_y: float = Field(15.0, description="Target Y position [m]")
    green_width: float = Field(20.0, description="Green width [m]", gt=0)
    green_height: float = Field(20.0, description="Green height [m]", gt=0)
    stimp_rating: float = Field(10.0, ge=6.0, le=15.0)


class GreenReadingResponse(BaseModel):
    """Response with green reading data."""

    distance: float
    total_break: float
    recommended_speed: float
    aim_point: list[float]
    elevations: list[float]
    slopes: list[list[float]]


class ScatterAnalysisRequest(BaseModel):
    """Request for scatter analysis (multiple putts with variance)."""

    ball_x: float = Field(5.0)
    ball_y: float = Field(10.0)
    speed: float = Field(2.0, gt=0, le=10)
    direction_x: float = Field(0.0)
    direction_y: float = Field(1.0)
    n_simulations: int = Field(10, ge=1, le=100)
    speed_variance: float = Field(0.1, ge=0)
    direction_variance_deg: float = Field(2.0, ge=0)
    green_width: float = Field(20.0, gt=0)
    green_height: float = Field(20.0, gt=0)
    stimp_rating: float = Field(10.0, ge=6.0, le=15.0)


class ScatterAnalysisResponse(BaseModel):
    """Response with scatter analysis results."""

    final_positions: list[list[float]]
    holed_count: int
    total_simulations: int
    average_distance_from_hole: float
    make_percentage: float


class GreenContourResponse(BaseModel):
    """Response with slope contour data for visualization."""

    width: float
    height: float
    grid_x: list[list[float]]
    grid_y: list[list[float]]
    elevations: list[list[float]]
    hole_position: list[float]


# ── Endpoints ──


@router.post("/simulate", response_model=PuttSimulationResponse)
async def simulate_putt(request: PuttSimulationRequest) -> PuttSimulationResponse:
    """Simulate a single putt with given parameters.

    See issue #1206
    """
    try:
        from src.engines.physics_engines.putting_green.python.green_surface import (
            GreenSurface,
        )
        from src.engines.physics_engines.putting_green.python.putter_stroke import (
            StrokeParameters,
        )
        from src.engines.physics_engines.putting_green.python.simulator import (
            PuttingGreenSimulator,
            SimulationConfig,
        )
        from src.engines.physics_engines.putting_green.python.turf_properties import (
            TurfProperties,
        )

        turf = TurfProperties(stimp_rating=request.stimp_rating)
        green = GreenSurface(
            width=request.green_width,
            height=request.green_height,
            turf=turf,
        )
        green.set_hole_position(np.array([request.hole_x, request.hole_y]))

        config = SimulationConfig(record_trajectory=True)
        sim = PuttingGreenSimulator(green=green, config=config)

        if request.wind_speed > 0:
            sim.set_wind(
                request.wind_speed,
                np.array([request.wind_direction_x, request.wind_direction_y]),
            )

        direction = np.array([request.direction_x, request.direction_y])
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm

        stroke = StrokeParameters(speed=request.speed, direction=direction)
        result = sim.simulate_putt(
            stroke, ball_position=np.array([request.ball_x, request.ball_y])
        )

        return PuttSimulationResponse(
            positions=result.positions.tolist(),
            velocities=result.velocities.tolist(),
            times=result.times.tolist(),
            holed=result.holed,
            final_position=result.final_position.tolist(),
            total_distance=result.total_distance,
            duration=result.duration,
        )

    except ImportError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/read-green", response_model=GreenReadingResponse)
async def read_green(request: GreenReadingRequest) -> GreenReadingResponse:
    """Read green between ball and target positions.

    See issue #1206
    """
    try:
        from src.engines.physics_engines.putting_green.python.green_surface import (
            GreenSurface,
        )
        from src.engines.physics_engines.putting_green.python.simulator import (
            PuttingGreenSimulator,
        )
        from src.engines.physics_engines.putting_green.python.turf_properties import (
            TurfProperties,
        )

        turf = TurfProperties(stimp_rating=request.stimp_rating)
        green = GreenSurface(
            width=request.green_width,
            height=request.green_height,
            turf=turf,
        )
        green.set_hole_position(np.array([request.target_x, request.target_y]))

        sim = PuttingGreenSimulator(green=green)
        reading = sim.read_green(
            np.array([request.ball_x, request.ball_y]),
            np.array([request.target_x, request.target_y]),
        )

        return GreenReadingResponse(
            distance=float(reading["distance"]),
            total_break=float(reading["total_break"]),
            recommended_speed=float(reading["recommended_speed"]),
            aim_point=reading["aim_point"].tolist(),
            elevations=[float(e) for e in reading["elevations"]],
            slopes=[s.tolist() for s in reading["slopes"]],
        )

    except ImportError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/scatter", response_model=ScatterAnalysisResponse)
async def scatter_analysis(
    request: ScatterAnalysisRequest,
) -> ScatterAnalysisResponse:
    """Run scatter analysis with multiple putts.

    See issue #1206
    """
    try:
        from src.engines.physics_engines.putting_green.python.green_surface import (
            GreenSurface,
        )
        from src.engines.physics_engines.putting_green.python.putter_stroke import (
            StrokeParameters,
        )
        from src.engines.physics_engines.putting_green.python.simulator import (
            PuttingGreenSimulator,
        )
        from src.engines.physics_engines.putting_green.python.turf_properties import (
            TurfProperties,
        )

        turf = TurfProperties(stimp_rating=request.stimp_rating)
        green = GreenSurface(
            width=request.green_width,
            height=request.green_height,
            turf=turf,
        )

        sim = PuttingGreenSimulator(green=green)

        direction = np.array([request.direction_x, request.direction_y])
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm

        stroke = StrokeParameters(speed=request.speed, direction=direction)
        results = sim.simulate_scatter(
            start_position=np.array([request.ball_x, request.ball_y]),
            stroke_params=stroke,
            n_simulations=request.n_simulations,
            speed_variance=request.speed_variance,
            direction_variance_deg=request.direction_variance_deg,
        )

        final_positions = [r.final_position.tolist() for r in results]
        holed_count = sum(1 for r in results if r.holed)
        hole_pos = green.hole_position
        avg_dist = float(
            np.mean([np.linalg.norm(r.final_position - hole_pos) for r in results])
        )

        return ScatterAnalysisResponse(
            final_positions=final_positions,
            holed_count=holed_count,
            total_simulations=len(results),
            average_distance_from_hole=avg_dist,
            make_percentage=(holed_count / len(results) * 100 if results else 0),
        )

    except ImportError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/contours", response_model=GreenContourResponse)
async def get_green_contours(
    width: float = 20.0,
    height: float = 20.0,
    resolution: int = 20,
    stimp_rating: float = 10.0,
) -> GreenContourResponse:
    """Get green elevation contour data for 2D visualization.

    See issue #1206
    """
    try:
        from src.engines.physics_engines.putting_green.python.green_surface import (
            GreenSurface,
        )
        from src.engines.physics_engines.putting_green.python.turf_properties import (
            TurfProperties,
        )

        turf = TurfProperties(stimp_rating=stimp_rating)
        green = GreenSurface(width=width, height=height, turf=turf)

        xs = np.linspace(0, width, resolution)
        ys = np.linspace(0, height, resolution)
        grid_x, grid_y = np.meshgrid(xs, ys)
        elevations = np.zeros_like(grid_x)

        for i in range(resolution):
            for j in range(resolution):
                elevations[i, j] = green.get_elevation(grid_x[i, j], grid_y[i, j])  # type: ignore[attr-defined]

        return GreenContourResponse(
            width=width,
            height=height,
            grid_x=grid_x.tolist(),
            grid_y=grid_y.tolist(),
            elevations=elevations.tolist(),
            hole_position=green.hole_position.tolist(),
        )

    except ImportError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
