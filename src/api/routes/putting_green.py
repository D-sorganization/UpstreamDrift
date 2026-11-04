"""Putting Green Simulator API routes.

Provides REST endpoints for the putting green simulation tool page:
- Simulate putts with configurable parameters
- Read green contours and slope data
- Get aim-line assist calculations
- Scatter analysis for practice mode

Roll-model provenance (ADR-0045 F1, issue #9343; migrated #14141 steps 7a-7c):
    The legacy UD roll law (``ud-legacy-roll/1``) is retired (ADR-0058)
    and no route runs it: every route in this module — ``/simulate``, ``/scatter``,
    ``/contours``, ``/simulate-3d`` and, as of step 7c, ``/read-green`` — runs
    the surface-aware model in ``src.shared.python.putting_dynamics``, which
    restates the Tools stimpmeter law (``usga-stimp-roll/1``). ``roll_model``
    is a required response field: a handler that fails to set it raises
    rather than emitting an unnamed result. ``/contours`` reports surface
    geometry only — no roll model is involved, so it carries no name (it
    reads the shared flat height field, but a flat green's elevations are
    zero either way).

See issue #1206
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Literal, Protocol

import numpy as np
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field, field_validator

from src.api.middleware.error_handler import handle_api_errors
from src.api.rate_limit import get_limit, limiter
from src.shared.python.core.contracts import PostconditionError, precondition, require
from src.shared.python.putting_dynamics.phases import phase_marks
from src.shared.python.putting_dynamics.green_spec import (
    GREEN_PRESETS,
    GreenSpec,
    green_preset,
)
from src.shared.python.putting_dynamics.putter_presets import (
    BALANCE_PRESETS,
    balance_preset,
)
from src.shared.python.putting_dynamics.types import (
    TYPICAL_HEAD_MASS_KG,
    TYPICAL_HEAD_MOI_KG_M2,
    TYPICAL_LOFT_DEG,
)

if TYPE_CHECKING:
    from src.shared.python.putting_dynamics import (
        BallState,
        CollisionReport,
        HeightField,
        PutterState,
        SurfaceSpec,
        TrajectorySample,
    )

router = APIRouter(prefix="/tools/putting-green", tags=["putting-green"])

#: Putter head used when a request carries only a stroke speed (shared by
#: ``/simulate-3d`` defaults and the 2-D routes).
_DEFAULT_HEAD_MASS_KG = TYPICAL_HEAD_MASS_KG
_DEFAULT_HEAD_MOI_KG_M2 = TYPICAL_HEAD_MOI_KG_M2
_DEFAULT_LOFT_DEG = TYPICAL_LOFT_DEG


# -- Request / Response Models --


#: The shared core's preset names, so an unknown one is a 422 in body or query.
GreenPresetName = Enum(  # type: ignore[misc]
    "GreenPresetName", {key: key for key in GREEN_PRESETS}, type=str
)


class GreenShapeRequest(BaseModel):
    """The green's shape (#14463): a named preset, or slope, tier and bumps.

    A ``preset`` replaces every other field with that preset's shape.
    """

    preset: GreenPresetName | None = Field(None, description="GET /green-presets")
    grade_percent: float = Field(0.0, ge=0.0, le=10.0)
    aspect_deg: float = Field(0.0, ge=-360.0, le=360.0, description="Falls toward")
    bump_height_m: float = Field(0.0, ge=0.0, le=0.05)
    bump_length_m: float = Field(1.5, gt=0.0, le=20.0)
    tier_height_m: float = Field(0.0, ge=0.0, le=0.15)
    tier_width_m: float = Field(1.0, gt=0.0, le=5.0)
    seed: int = Field(0, ge=0)

    def spec(self) -> GreenSpec:
        """The shared core's description of this green."""
        if self.preset is not None:
            return green_preset(self.preset.value)
        return GreenSpec(**self.model_dump(exclude={"preset"}))


def _flat_green() -> GreenShapeRequest:
    return GreenShapeRequest.model_validate({})


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
    green: GreenShapeRequest = Field(default_factory=_flat_green)
    wind_speed: float = Field(0.0, description="Wind speed [m/s]", ge=0)
    wind_direction_x: float = Field(1.0, description="Wind direction X")
    wind_direction_y: float = Field(0.0, description="Wind direction Y")


class PhaseMarkResponse(BaseModel):
    """Where one run of a ball mode starts, for playback jumps (#14473)."""

    mode: Literal["airborne", "slide", "roll", "rest"]
    t_s: float
    index: int = Field(description="Index of the run's first sample")


def _phase_marks(samples: Sequence[TrajectorySample]) -> list[PhaseMarkResponse]:
    return [
        PhaseMarkResponse(mode=m.mode.value, t_s=m.t_s, index=m.index)
        for m in phase_marks(samples)
    ]


class PuttSimulationResponse(BaseModel):
    """Response containing putt simulation results."""

    roll_model: str = Field(
        description="Roll model that produced this result (ADR-0045 F1)"
    )
    positions: list[list[float]]
    velocities: list[list[float]]
    times: list[float]
    modes: list[Literal["airborne", "slide", "roll", "rest"]] = Field(
        default_factory=list, description="Ball mode at each sample (#14473)"
    )
    phases: list[PhaseMarkResponse] = Field(default_factory=list)
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
    green: GreenShapeRequest = Field(default_factory=_flat_green)


class GreenReadingResponse(BaseModel):
    """Response with green reading data."""

    roll_model: str = Field(
        description="Roll model behind the recommended speed (ADR-0045 F1)"
    )
    distance: float
    total_break: float
    recommended_speed: float
    aim_point: list[float]
    aim_heading_deg: float = Field(
        description="Putter heading that plays the read, CCW from +x [deg] (#14465)"
    )
    holed: bool = Field(description="Whether the solved putt drops")
    converged: bool = Field(description="Whether the heading solve converged")
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
    green: GreenShapeRequest = Field(default_factory=_flat_green)


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
    loft_deg: float = Field(_DEFAULT_LOFT_DEG, ge=-6.0, le=10.0)
    head_mass_kg: float = Field(_DEFAULT_HEAD_MASS_KG, ge=0.1, le=1.0)
    head_moi_kg_m2: float = Field(_DEFAULT_HEAD_MOI_KG_M2, ge=1e-5, le=1e-2)
    coefficient_of_restitution: float = Field(0.78, gt=0.0, lt=1.0)
    hosel_toe_m: float = Field(0.0, ge=-0.08, le=0.08)
    hosel_forward_m: float = Field(0.0, ge=-0.05, le=0.05)
    balance_type: str | None = Field(
        None,
        description=(
            "Named core putter head (see /putter-presets); when set it supplies "
            "head mass, MOI and hosel offsets, overriding those fields"
        ),
    )
    impact_toe_m: float = Field(
        0.0, ge=-0.08, le=0.08, description="Strike offset, toe + / heel - [m]"
    )
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

    @field_validator("balance_type")
    @classmethod
    def _known_balance_type(cls, value: str | None) -> str | None:
        if value is not None and value not in BALANCE_PRESETS:
            raise ValueError(f"balance_type must be one of {sorted(BALANCE_PRESETS)}")
        return value


class BalancePresetResponse(BaseModel):
    """One named putter head from the shared core (#14461)."""

    key: str
    name: str
    head_mass_kg: float
    moi_kg_m2: float
    hosel_toe_m: float
    hosel_forward_m: float


class BalancePresetsResponse(BaseModel):
    """Every core balance preset, for the balance-type selectors."""

    presets: list[BalancePresetResponse]


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
    phases: list[PhaseMarkResponse] = Field(default_factory=list)
    collision: Putt3DCollisionResponse
    surface: Putt3DSurfaceResponse
    holed: bool
    total_distance_m: float
    duration_s: float
    skid_distance_m: float


# -- Endpoints --


def _build_putting_surface(
    stimp_rating: float,
    green_width: float,
    green_height: float,
    green: GreenSpec | None = None,
) -> tuple[SurfaceSpec, float, float]:
    """Shared-core surface of ``green`` (flat by default) covering one green.

    ``/simulate``, ``/scatter``, ``/read-green`` and ``/contours`` describe
    their green as spanning ``(0, 0)`` to ``(width, height)``; the
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
    extent_m = min(200.0, max(1.0, green_width, green_height))
    spec = green if green is not None else GreenSpec()
    surface = spec.surface(stimp_ft=stimp_rating, extent_m=extent_m)
    return surface, green_width / 2.0, green_height / 2.0


class _GreenRequest(Protocol):
    """A body request that rolls on a green (/simulate, /scatter, /read-green)."""

    stimp_rating: float
    green_width: float
    green_height: float
    green: GreenShapeRequest


def _request_surface(request: _GreenRequest) -> tuple[SurfaceSpec, float, float]:
    """:func:`_build_putting_surface` for a body request's green (#14463)."""
    return _build_putting_surface(
        request.stimp_rating,
        request.green_width,
        request.green_height,
        request.green.spec(),
    )


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


def _default_putter(speed_mps: float) -> PutterState:
    """The default putter head swung at ``speed_mps``."""
    from src.shared.python.putting_dynamics import PutterState

    return PutterState.typical(speed_mps)


def _struck_ball(
    stroke_speed_mps: float, direction: tuple[float, float], x_m: float, y_m: float
) -> tuple[BallState, CollisionReport]:
    """Ball state after the default putter strikes it along ``direction``.

    ``speed`` in the 2-D requests is the putter-head speed, as in the legacy
    engine, so the ball leaves through the shared impact model
    (:func:`~src.shared.python.putting_dynamics.strike`) and skids into roll.
    """
    from src.shared.python.putting_dynamics import strike, struck_ball_state

    dir_x, dir_y = direction
    require(
        abs(math.hypot(dir_x, dir_y) - 1.0) < 1e-9, "direction must be a unit vector"
    )
    report = strike(_default_putter(stroke_speed_mps))
    ball = struck_ball_state(
        report, x_m=x_m, y_m=y_m, heading_rad=math.atan2(dir_y, dir_x)
    )
    return ball, report


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
    if payload.balance_type is not None:
        putter = balance_preset(payload.balance_type).putter(
            payload.putter_speed_mps,
            loft_deg=payload.loft_deg,
            cor=payload.coefficient_of_restitution,
        )
    else:
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
    from src.shared.python.putting_dynamics import USGA_STIMP_ROLL_MODEL
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
        phases=_phase_marks(result.samples),
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
    from src.shared.python.putting_dynamics import USGA_STIMP_ROLL_MODEL
    from src.shared.python.putting_dynamics import simulate_ball

    surface, half_w, half_h = _request_surface(payload)

    direction = np.array([payload.direction_x, payload.direction_y])
    norm = math.hypot(float(direction[0]), float(direction[1]))
    if norm > 0:
        direction = direction / norm
    dir_x, dir_y = float(direction[0]), float(direction[1])

    ball_x, ball_y = _to_surface_frame(payload.ball_x, payload.ball_y, half_w, half_h)
    hole_x, hole_y = _to_surface_frame(payload.hole_x, payload.hole_y, half_w, half_h)

    ball, report = _struck_ball(payload.speed, (dir_x, dir_y), ball_x, ball_y)
    result = simulate_ball(
        ball, surface, hole_x_m=hole_x, hole_y_m=hole_y, collision=report
    )

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
        modes=[s.mode.value for s in result.samples],
        phases=_phase_marks(result.samples),
        holed=result.holed,
        final_position=final_position,
        total_distance=result.total_distance_m,
        duration=result.time_s,
    )


@router.post("/read-green", response_model=GreenReadingResponse)
@handle_api_errors
async def read_green(request: GreenReadingRequest) -> GreenReadingResponse:
    """Read the green between the ball and the target by simulation.

    Runs ``usga-stimp-roll/1`` (epic #14141 step 7c):
    :func:`~src.shared.python.putting_dynamics.read_line` samples the
    elevation/slope profile and
    :func:`~src.shared.python.putting_dynamics.solve_aim` finds the
    putter-head speed and heading that hole (or just pass) the target by
    simulating the strike with :func:`~src.shared.python.putting_dynamics.
    simulate_strike`, rather than a separate closed-form heuristic, on the
    requested green (flat by default, #14463). See issue #1206.
    """
    from src.shared.python.putting_dynamics import USGA_STIMP_ROLL_MODEL
    from src.shared.python.putting_dynamics import read_line, solve_aim

    surface, half_w, half_h = _request_surface(request)
    ball_xy = _to_surface_frame(request.ball_x, request.ball_y, half_w, half_h)
    target_xy = _to_surface_frame(request.target_x, request.target_y, half_w, half_h)

    line = read_line(surface, ball_xy, target_xy)
    solution = solve_aim(surface, ball_xy, target_xy, _default_putter(1.0))
    aim_x, aim_y = _from_surface_frame(*solution.aim_point_xy, half_w, half_h)

    return GreenReadingResponse(
        roll_model=USGA_STIMP_ROLL_MODEL,
        distance=line.distance_m,
        total_break=abs(solution.break_m),
        recommended_speed=solution.putter_speed_mps,
        aim_point=[aim_x, aim_y],
        aim_heading_deg=math.degrees(solution.heading_rad),
        holed=solution.holed,
        converged=solution.converged,
        elevations=list(line.elevations_m),
        slopes=[list(s) for s in line.slopes],
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
    from src.shared.python.putting_dynamics import USGA_STIMP_ROLL_MODEL
    from src.shared.python.putting_dynamics import simulate_ball

    surface, half_w, half_h = _request_surface(request)

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

    # A fixed seed (0) keeps repeated calls deterministic.
    rng = np.random.default_rng(0)

    final_positions: list[list[float]] = []
    holed_count = 0
    for _ in range(request.n_simulations):
        speed = max(0.1, request.speed + rng.normal(0.0, request.speed_variance))
        angle_var = rng.normal(0.0, request.direction_variance_deg * math.pi / 180.0)
        cos_a, sin_a = math.cos(angle_var), math.sin(angle_var)
        putt_dir_x = cos_a * dir_x - sin_a * dir_y
        putt_dir_y = sin_a * dir_x + cos_a * dir_y

        ball, _ = _struck_ball(speed, (putt_dir_x, putt_dir_y), ball_x, ball_y)
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


@router.get("/putter-presets", response_model=BalancePresetsResponse)
@handle_api_errors
async def get_putter_presets() -> BalancePresetsResponse:
    """Named putter balance presets from the shared core (#14461)."""
    presets = [
        BalancePresetResponse(**dataclasses.asdict(p)) for p in BALANCE_PRESETS.values()
    ]
    return BalancePresetsResponse(presets=presets)


@router.get("/contours", response_model=GreenContourResponse)
@precondition(
    lambda green, width=20.0, height=20.0, resolution=20, stimp_rating=10.0: (
        width > 0 and height > 0 and resolution >= 2 and 6.0 <= stimp_rating <= 15.0
    ),
    "Green dimensions must be positive, resolution >= 2, and stimp_rating in [6, 15]",
)
@handle_api_errors
async def get_green_contours(
    green: Annotated[GreenShapeRequest, Depends()],
    width: float = 20.0,
    height: float = 20.0,
    resolution: int = 20,
    stimp_rating: float = 10.0,
) -> GreenContourResponse:
    """Get green elevation contour data for 2D visualization.

    Reads the shared :class:`~src.shared.python.putting_dynamics.
    HeightField` of the requested green (epic #14141 step 7a, #14463); the
    default flat green's elevations are zero everywhere.
    No roll model is involved, so this response carries no ``roll_model``
    field. See issue #1206.
    """
    if not (width is not None):
        raise ValueError("width must be provided")

    surface, half_w, half_h = _build_putting_surface(
        stimp_rating, width, height, green.spec()
    )

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


@router.get("/green-presets", response_model=dict[str, GreenShapeRequest])
@handle_api_errors
async def list_green_presets() -> dict[str, GreenShapeRequest]:
    """The shared core's named green shapes (#14463), by preset key."""
    return {
        key: GreenShapeRequest(**dataclasses.asdict(spec))
        for key, spec in GREEN_PRESETS.items()
    }
