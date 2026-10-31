"""Putting Green Simulator API routes.

Provides REST endpoints for the putting green simulation tool page:
- Simulate putts with configurable parameters
- Read green contours and slope data
- Get aim-line assist calculations
- Scatter analysis for practice mode

Roll-model provenance (ADR-0045 F1, issue #9343; migrated #14141 step 7a):
    Two roll models are preserved and both are reachable from this router, so
    every response that reports a roll-out result names the model that produced
    it. ``/read-green`` is now the only route running the UD engine
    (``ud-legacy-roll/1``); ``/simulate``, ``/scatter``, ``/contours`` and
    ``/simulate-3d`` all run the surface-aware model in
    ``src.shared.python.putting_dynamics``, which restates the Tools
    stimpmeter law (``usga-stimp-roll/1``). The two differ by the ~2.854
    roll-out ratio pinned in Tools#4819, so their numbers must never be
    compared without the names. ``roll_model`` is a required response field:
    a handler that fails to set it raises rather than emitting an unnamed
    result. ``/contours`` reports surface geometry only — no roll model is
    involved, so it carries no name (it now reads the shared flat height
    field, but a flat green's elevations are zero either way).

See issue #1206
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import numpy as np
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from src.api.middleware.error_handler import handle_api_errors
from src.api.rate_limit import get_limit, limiter
from src.shared.python.core.contracts import PostconditionError, precondition, require

if TYPE_CHECKING:
    from src.shared.python.putting_dynamics import HeightField, PutterState, SurfaceSpec

router = APIRouter(prefix="/tools/putting-green", tags=["putting-green"])


# -- Request / Response Models --


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
    wind_speed: float = Field(
        0.0,
        description=(
            "Wind speed [m/s]; must be 0 — the shared putting core has no wind model"
        ),
        ge=0,
    )
    wind_direction_x: float = Field(1.0, description="Wind direction X")
    wind_direction_y: float = Field(0.0, description="Wind direction Y")


class PuttSimulationResponse(BaseModel):
    """Response containing putt simulation results."""

    roll_model: str = Field(
        description="Roll model that produced this result (ADR-0045 F1)"
    )
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

    roll_model: str = Field(
        description="Roll model behind the recommended speed (ADR-0045 F1)"
    )
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

    roll_model: str = Field(
        description="Roll model that produced these results (ADR-0045 F1)"
    )
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


class Putt3DSimulationRequest(BaseModel):
    """Physics and visualization inputs for one three-dimensional putt."""

    putter_speed_mps: float = Field(1.8, gt=0.0, le=10.0)
    loft_deg: float = Field(3.0, ge=-6.0, le=10.0)
    head_mass_kg: float = Field(0.35, ge=0.1, le=1.0)
    head_moi_kg_m2: float = Field(4.5e-4, ge=1e-5, le=1e-2)
    coefficient_of_restitution: float = Field(0.78, gt=0.0, lt=1.0)
    hosel_toe_m: float = Field(0.0, ge=-0.08, le=0.08)
    hosel_forward_m: float = Field(0.0, ge=-0.05, le=0.05)
    impact_toe_m: float = Field(0.0, ge=-0.08, le=0.08)
    stimp_rating: float = Field(10.0, ge=6.0, le=15.0)
    grade_percent: float = Field(0.0, ge=0.0, le=10.0)
    downhill_aspect_deg: float = Field(0.0, ge=-180.0, le=180.0)
    grain_strength: float = Field(0.0, ge=0.0, lt=0.9)
    grain_direction_deg: float = Field(0.0, ge=-180.0, le=180.0)
    rolling_velocity_coefficient: float = Field(0.0, ge=0.0, le=1.0)
    bump_height_m: float = Field(0.0, ge=0.0, le=0.01)
    friction_variation: float = Field(0.0, ge=0.0, le=0.9)
    random_seed: int = Field(8345, ge=0, le=2**31 - 1)
    hole_x_m: float = Field(3.0, ge=-9.0, le=9.0)
    hole_y_m: float = Field(0.0, ge=-9.0, le=9.0)


class Putt3DSampleResponse(BaseModel):
    """One frame in the three-dimensional playback trajectory."""

    t_s: float
    x_m: float
    y_m: float
    z_m: float
    speed_mps: float
    spin_rad_s: float
    mode: Literal["airborne", "slide", "roll", "rest"]


class Putt3DCollisionResponse(BaseModel):
    """Impact quantities displayed beside the slow-motion collision."""

    ball_speed_mps: float
    putter_speed_before_mps: float
    putter_speed_after_mps: float
    launch_angle_deg: float
    spin_rad_s: float
    impulse_n_s: float
    contact_time_proxy_s: float
    kinetic_energy_loss_j: float
    face_twist_rad_s: float
    twist_moment_n_m_s: float


class Putt3DSurfaceResponse(BaseModel):
    """Surface geometry metadata required by the R3F scene."""

    width_m: float
    height_m: float
    grade_percent: float
    downhill_aspect_deg: float
    hole_x_m: float
    hole_y_m: float


class Putt3DSimulationResponse(BaseModel):
    """Complete deterministic playback payload for the R3F client."""

    roll_model: str = Field(
        description="Roll model that produced this playback (ADR-0045 F1)"
    )
    samples: list[Putt3DSampleResponse]
    collision: Putt3DCollisionResponse
    surface: Putt3DSurfaceResponse
    holed: bool
    total_distance_m: float
    duration_s: float
    skid_distance_m: float


# -- Endpoints --


def _build_flat_putting_surface(
    stimp_rating: float, green_width: float, green_height: float
) -> tuple[SurfaceSpec, float, float]:
    """Flat, homogeneous shared-core surface sized to cover one green.

    ``/simulate``, ``/scatter`` and ``/contours`` describe their green as a
    legacy flat surface spanning ``(0, 0)`` to ``(width, height)``; the
    shared :class:`~src.shared.python.putting_dynamics.HeightField` /
    :class:`~src.shared.python.putting_dynamics.FrictionField` grids are
    centred on the origin instead (see ``putting_dynamics/surfaces.py``).

    Returns the flat shared surface plus the ``(half_width, half_height)``
    translation that maps a green-local point into the surface's centred
    frame; callers translate in with :func:`_to_surface_frame`, simulate,
    and translate results back out with :func:`_from_surface_frame` so
    responses stay in the legacy green-local frame the existing schema and
    frontend already use.
    """
    from src.shared.python.putting_dynamics import SurfaceSpec

    extent_m = min(200.0, max(1.0, green_width, green_height))
    surface = SurfaceSpec.flat_uniform(stimp_ft=stimp_rating, extent_m=extent_m)
    return surface, green_width / 2.0, green_height / 2.0


def _to_surface_frame(
    x_m: float, y_m: float, half_width: float, half_height: float
) -> tuple[float, float]:
    """Green-local ``(0..width, 0..height)`` -> surface-centred coordinates."""
    return x_m - half_width, y_m - half_height


def _from_surface_frame(
    x_m: float, y_m: float, half_width: float, half_height: float
) -> tuple[float, float]:
    """Surface-centred coordinates -> green-local ``(0..width, 0..height)``."""
    return x_m + half_width, y_m + half_height


def _build_putting_3d_surface(
    payload: Putt3DSimulationRequest, extent_m: float = 20.0
) -> tuple[SurfaceSpec, HeightField, PutterState]:
    from src.shared.python.putting_dynamics import (
        FrictionField,
        FrictionParams,
        HeightField,
        PutterState,
        SurfaceSpec,
        bumpy_friction_field,
        bumpy_height_field,
        stimp_to_rolling_mu,
    )

    height = HeightField.planar(
        grade_percent=payload.grade_percent,
        aspect_deg=payload.downhill_aspect_deg,
        extent_m=extent_m,
    )
    if payload.bump_height_m > 0.0:
        height = bumpy_height_field(
            seed=payload.random_seed,
            amplitude_m=payload.bump_height_m,
            correlation_length_m=0.5,
            base=height,
        )

    friction_field = FrictionField.uniform(extent_m=extent_m)
    if payload.friction_variation > 0.0:
        friction_field = bumpy_friction_field(
            seed=payload.random_seed,
            amplitude=payload.friction_variation,
            correlation_length_m=0.5,
            base=friction_field,
        )
    friction = FrictionParams(
        mu_roll0=stimp_to_rolling_mu(payload.stimp_rating),
        k_v_per_mps=payload.rolling_velocity_coefficient,
        grain_strength=payload.grain_strength,
        grain_direction_rad=math.radians(payload.grain_direction_deg),
    )
    surface = SurfaceSpec(
        height=height,
        friction_field=friction_field,
        friction=friction,
    )
    putter = PutterState(
        head_mass_kg=payload.head_mass_kg,
        moi_kg_m2=payload.head_moi_kg_m2,
        loft_deg=payload.loft_deg,
        speed_mps=payload.putter_speed_mps,
        cor=payload.coefficient_of_restitution,
        hosel_toe_m=payload.hosel_toe_m,
        hosel_forward_m=payload.hosel_forward_m,
    )
    return surface, height, putter


@router.post("/simulate-3d", response_model=Putt3DSimulationResponse)
@limiter.limit(get_limit("API_LIMIT_PUTT_SIMULATE", "10/minute"))
@handle_api_errors
async def simulate_putt_3d(
    request: Request, payload: Putt3DSimulationRequest
) -> Putt3DSimulationResponse:
    """Run the canonical surface-aware strike model for R3F playback."""
    del request
    from src.engines.physics_engines.putting_green.python.ball_roll_physics import (
        USGA_STIMP_ROLL_MODEL,
    )
    from src.shared.python.putting_dynamics import simulate_strike

    extent_m = 20.0
    surface, height, putter = _build_putting_3d_surface(payload, extent_m)
    result = simulate_strike(
        putter,
        surface,
        impact_toe_m=payload.impact_toe_m,
        hole_x_m=payload.hole_x_m,
        hole_y_m=payload.hole_y_m,
    )
    collision = result.collision
    if collision is None:
        raise PostconditionError(
            "strike simulation must return a collision report",
            function_name="simulate_putt_3d",
            result=result,
        )

    samples = [
        Putt3DSampleResponse(
            t_s=sample.t_s,
            x_m=sample.x_m,
            y_m=sample.y_m,
            z_m=height.elevation(sample.x_m, sample.y_m) + sample.height_m,
            speed_mps=sample.speed_mps,
            spin_rad_s=sample.spin_rad_s,
            mode=sample.mode.value,
        )
        for sample in result.samples
    ]
    return Putt3DSimulationResponse(
        roll_model=USGA_STIMP_ROLL_MODEL,
        samples=samples,
        collision=Putt3DCollisionResponse(
            ball_speed_mps=collision.ball_speed_mps,
            putter_speed_before_mps=payload.putter_speed_mps,
            putter_speed_after_mps=(payload.putter_speed_mps - collision.putter_dv_mps),
            launch_angle_deg=collision.launch_angle_deg,
            spin_rad_s=collision.spin_rad_s,
            impulse_n_s=collision.impulse_n_s,
            contact_time_proxy_s=collision.contact_time_proxy_s,
            kinetic_energy_loss_j=collision.kinetic_energy_loss_j,
            face_twist_rad_s=collision.face_twist_rad_s,
            twist_moment_n_m_s=collision.twist_moment_n_m_s,
        ),
        surface=Putt3DSurfaceResponse(
            width_m=extent_m,
            height_m=extent_m,
            grade_percent=payload.grade_percent,
            downhill_aspect_deg=payload.downhill_aspect_deg,
            hole_x_m=payload.hole_x_m,
            hole_y_m=payload.hole_y_m,
        ),
        holed=result.holed,
        total_distance_m=result.total_distance_m,
        duration_s=result.time_s,
        skid_distance_m=result.skid_distance_m,
    )


@router.post("/simulate", response_model=PuttSimulationResponse)
@limiter.limit(get_limit("API_LIMIT_PUTT_SIMULATE", "10/minute"))
@precondition(
    lambda request, payload: payload.direction_x != 0.0 or payload.direction_y != 0.0,
    "Putt direction vector must not be zero",
)
@handle_api_errors
async def simulate_putt(
    request: Request, payload: PuttSimulationRequest
) -> PuttSimulationResponse:
    """Simulate a single putt on the shared surface-aware putting core.

    Runs ``usga-stimp-roll/1`` (epic #14141 step 7a); only ``/read-green``
    still runs the legacy UD engine. See issue #1206.
    """
    del request
    require(
        payload.wind_speed == 0.0,
        "Wind is not modelled by the shared putting core "
        "(src.shared.python.putting_dynamics); set wind_speed to 0",
        payload.wind_speed,
    )
    from src.engines.physics_engines.putting_green.python.ball_roll_physics import (
        USGA_STIMP_ROLL_MODEL,
    )
    from src.shared.python.core.physics_constants import GOLF_BALL_RADIUS_M
    from src.shared.python.putting_dynamics import BallState, simulate_ball

    surface, half_w, half_h = _build_flat_putting_surface(
        payload.stimp_rating, payload.green_width, payload.green_height
    )

    direction = np.array([payload.direction_x, payload.direction_y])
    norm = math.hypot(float(direction[0]), float(direction[1]))
    if norm > 0:
        direction = direction / norm
    dir_x, dir_y = float(direction[0]), float(direction[1])

    ball_x, ball_y = _to_surface_frame(payload.ball_x, payload.ball_y, half_w, half_h)
    hole_x, hole_y = _to_surface_frame(payload.hole_x, payload.hole_y, half_w, half_h)

    # The legacy engine runs `speed` through a putter-head-to-ball COR
    # transfer (PutterStroke.compute_launch_velocity) and gives the ball a
    # small loft-driven backspin before it skids into roll. This request
    # carries no putter mass/COR/loft inputs, so the shared core instead
    # treats `speed` as the ball's own ground speed with a pure-rolling
    # start (spin matched to the forward speed; no skid phase).
    ball = BallState(
        x_m=ball_x,
        y_m=ball_y,
        vx_mps=payload.speed * dir_x,
        vy_mps=payload.speed * dir_y,
        spin_rad_s=payload.speed / float(GOLF_BALL_RADIUS_M),
    )
    result = simulate_ball(ball, surface, hole_x_m=hole_x, hole_y_m=hole_y)

    positions = [
        list(_from_surface_frame(s.x_m, s.y_m, half_w, half_h)) for s in result.samples
    ]
    velocities = [[s.speed_mps * dir_x, s.speed_mps * dir_y] for s in result.samples]
    final_position = list(
        _from_surface_frame(result.rest_x_m, result.rest_y_m, half_w, half_h)
    )

    return PuttSimulationResponse(
        roll_model=USGA_STIMP_ROLL_MODEL,
        positions=positions,
        velocities=velocities,
        times=[s.t_s for s in result.samples],
        holed=result.holed,
        final_position=final_position,
        total_distance=result.total_distance_m,
        duration=result.time_s,
    )


@router.post("/read-green", response_model=GreenReadingResponse)
@handle_api_errors
async def read_green(request: GreenReadingRequest) -> GreenReadingResponse:
    """Read green between ball and target positions.

    See issue #1206
    """
    from src.engines.physics_engines.putting_green.python.green_surface import (
        GreenSurface,
    )
    from src.engines.physics_engines.putting_green.python.simulator import (
        ROLL_MODEL_FIELD,
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
        roll_model=str(reading[ROLL_MODEL_FIELD]),
        distance=float(reading["distance"]),
        total_break=float(reading["total_break"]),
        recommended_speed=float(reading["recommended_speed"]),
        aim_point=reading["aim_point"].tolist(),
        elevations=[float(e) for e in reading["elevations"]],
        slopes=[s.tolist() for s in reading["slopes"]],
    )


@router.post("/scatter", response_model=ScatterAnalysisResponse)
@precondition(
    lambda request: request.direction_x != 0.0 or request.direction_y != 0.0,
    "Scatter direction vector must not be zero",
)
@handle_api_errors
async def scatter_analysis(
    request: ScatterAnalysisRequest,
) -> ScatterAnalysisResponse:
    """Run scatter analysis with multiple putts on the shared putting core.

    Runs ``usga-stimp-roll/1`` (epic #14141 step 7a). See issue #1206.
    """
    from src.engines.physics_engines.putting_green.python.ball_roll_physics import (
        USGA_STIMP_ROLL_MODEL,
    )
    from src.shared.python.core.physics_constants import GOLF_BALL_RADIUS_M
    from src.shared.python.putting_dynamics import BallState, simulate_ball

    surface, half_w, half_h = _build_flat_putting_surface(
        request.stimp_rating, request.green_width, request.green_height
    )
    ball_radius_m = float(GOLF_BALL_RADIUS_M)

    direction = np.array([request.direction_x, request.direction_y])
    norm = math.hypot(float(direction[0]), float(direction[1]))
    if norm > 0:
        direction = direction / norm
    dir_x, dir_y = float(direction[0]), float(direction[1])

    ball_x, ball_y = _to_surface_frame(request.ball_x, request.ball_y, half_w, half_h)
    # Scatter keeps the legacy engine's default hole at the green centre
    # (GreenSurface puts it at width/2, height/2); the request carries no
    # hole position of its own.
    hole_x, hole_y = _to_surface_frame(
        request.green_width / 2.0, request.green_height / 2.0, half_w, half_h
    )

    # Same seed (0) the legacy `PuttingGreenSimulator` falls back to when
    # no `random_seed`/`rng` is supplied (see `_sim_core.py.__init__`), so
    # repeated calls stay deterministic.
    rng = np.random.default_rng(0)

    final_positions: list[list[float]] = []
    holed_count = 0
    for _ in range(request.n_simulations):
        speed = max(0.1, request.speed + rng.normal(0.0, request.speed_variance))
        angle_var = rng.normal(0.0, request.direction_variance_deg * math.pi / 180.0)
        cos_a, sin_a = math.cos(angle_var), math.sin(angle_var)
        putt_dir_x = cos_a * dir_x - sin_a * dir_y
        putt_dir_y = sin_a * dir_x + cos_a * dir_y

        ball = BallState(
            x_m=ball_x,
            y_m=ball_y,
            vx_mps=speed * putt_dir_x,
            vy_mps=speed * putt_dir_y,
            spin_rad_s=speed / ball_radius_m,
        )
        result = simulate_ball(ball, surface, hole_x_m=hole_x, hole_y_m=hole_y)
        final_positions.append(
            list(_from_surface_frame(result.rest_x_m, result.rest_y_m, half_w, half_h))
        )
        if result.holed:
            holed_count += 1

    if final_positions:
        hole_pos = np.array([request.green_width / 2.0, request.green_height / 2.0])
        # Vectorized sum of squares avoids repeated np.linalg.norm calls.
        diffs = np.array(final_positions, dtype=float) - hole_pos
        avg_dist = float(np.mean(np.sqrt(np.einsum("ij,ij->i", diffs, diffs))))
    else:
        avg_dist = float("nan")

    return ScatterAnalysisResponse(
        roll_model=USGA_STIMP_ROLL_MODEL,
        final_positions=final_positions,
        holed_count=holed_count,
        total_simulations=len(final_positions),
        average_distance_from_hole=avg_dist,
        make_percentage=(
            holed_count / len(final_positions) * 100 if final_positions else 0
        ),
    )


@router.get("/contours", response_model=GreenContourResponse)
@precondition(
    lambda width=20.0, height=20.0, resolution=20, stimp_rating=10.0: (
        width > 0 and height > 0 and resolution >= 2 and 6.0 <= stimp_rating <= 15.0
    ),
    "Green dimensions must be positive, resolution >= 2, and stimp_rating in [6, 15]",
)
@handle_api_errors
async def get_green_contours(
    width: float = 20.0,
    height: float = 20.0,
    resolution: int = 20,
    stimp_rating: float = 10.0,
) -> GreenContourResponse:
    """Get green elevation contour data for 2D visualization.

    Reads the shared flat :class:`~src.shared.python.putting_dynamics.
    HeightField` (epic #14141 step 7a); a flat green's elevations are
    zero everywhere, matching the legacy engine's default flat surface.
    No roll model is involved, so this response carries no ``roll_model``
    field. See issue #1206.
    """
    if not (width is not None):
        raise ValueError("width must be provided")

    surface, half_w, half_h = _build_flat_putting_surface(stimp_rating, width, height)

    xs = np.linspace(0, width, resolution)
    ys = np.linspace(0, height, resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    elevations = np.zeros_like(grid_x)

    for i in range(resolution):
        for j in range(resolution):
            sx, sy = _to_surface_frame(
                float(grid_x[i, j]), float(grid_y[i, j]), half_w, half_h
            )
            elevations[i, j] = surface.height.elevation(sx, sy)

    return GreenContourResponse(
        width=width,
        height=height,
        grid_x=grid_x.tolist(),
        grid_y=grid_y.tolist(),
        elevations=elevations.tolist(),
        hole_position=[width / 2.0, height / 2.0],
    )
