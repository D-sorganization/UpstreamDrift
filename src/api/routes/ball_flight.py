"""Ball-flight simulator API routes.

Provides a headless REST endpoint for batch and UI clients that need golf-ball
flight trajectories without launching the GUI.

See issue #7218.
"""

from __future__ import annotations

import math
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field, model_validator

from src.api.middleware.error_handler import handle_api_errors
from src.shared.python.physics.flight_models import (
    FlightModelRegistry,
    FlightModelType,
    TrajectoryPoint,
    UnifiedLaunchConditions,
)
from src.api.routes._ball_flight_trajectory_import import (
    ImportedBallFlightTrajectory,
    import_trajectory_record,
    summarize_imported_trajectory,
)

router = APIRouter(prefix="/tools/ball-flight", tags=["ball-flight"])


class FlightModelInfo(BaseModel):
    """Metadata describing one registered ball-flight model."""

    key: FlightModelType
    name: str
    description: str
    reference: str


class FlightModelListResponse(BaseModel):
    """Enumeration of every flight model in :class:`FlightModelRegistry`."""

    models: list[FlightModelInfo]


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
    models: list[FlightModelType] | None = Field(
        None,
        description=(
            "Optional list of flight models for overlay comparison. When"
            " provided, the response carries one result per (deduplicated)"
            " model; the top-level trajectory/summary mirror the first entry"
            " for backwards compatibility."
        ),
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
        if self.models is not None and not self.models:
            raise ValueError("models must contain at least one flight model")
        return self

    def requested_models(self) -> list[FlightModelType]:
        """Return the deduplicated, order-preserving list of models to run."""
        requested = self.models or [self.model_name]
        return list(dict.fromkeys(requested))

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


class BallFlightModelResult(BaseModel):
    """Trajectory and summary metrics for a single flight model."""

    model_name: str
    model_key: FlightModelType
    #: Named coefficient values the model integrated with, so every returned
    #: trajectory is attributable to its coefficient set (issue #8978).
    coefficients: dict[str, float]
    trajectory: list[BallFlightTrajectorySample]
    summary: BallFlightSummary


class BallFlightSimulationResponse(BallFlightModelResult):
    """Response containing ball-flight trajectories and summary metrics.

    The top-level ``model_name``/``trajectory``/``summary`` fields describe
    the first requested model (backwards compatible with single-model
    clients); ``results`` carries one entry per requested model for
    overlay comparison (issue #7456).
    """

    results: list[BallFlightModelResult]


def _trajectory_samples(
    result_trajectory: list[TrajectoryPoint],
) -> list[BallFlightTrajectorySample]:
    """Serialize existing trajectory points without changing physics behavior."""
    return [
        BallFlightTrajectorySample(
            time_s=float(point.time),
            position_m=[float(value) for value in point.position.tolist()],
            velocity_mps=[float(value) for value in point.velocity.tolist()],
        )
        for point in result_trajectory
    ]


def _simulate_one(
    model_type: FlightModelType, payload: BallFlightSimulationRequest
) -> BallFlightModelResult:
    """Run a single flight model against the requested launch conditions."""
    model = FlightModelRegistry.get_model(model_type)
    result = model.simulate(
        payload.to_launch_conditions(),
        max_time=payload.max_time_s,
        dt=payload.time_step_s,
    )
    return BallFlightModelResult(
        model_name=result.model_name,
        model_key=model_type,
        coefficients={k: float(v) for k, v in result.coefficients.items()},
        trajectory=_trajectory_samples(result.trajectory),
        summary=BallFlightSummary(
            carry_m=float(result.carry_distance),
            apex_m=float(result.max_height),
            flight_time_s=float(result.flight_time),
            landing_angle_deg=float(result.landing_angle),
            lateral_deviation_m=float(result.lateral_deviation),
        ),
    )


@router.get("/models", response_model=FlightModelListResponse)
@handle_api_errors
async def list_flight_models() -> FlightModelListResponse:
    """Enumerate available flight models from the shared desktop registry.

    Single source of truth: the same :class:`FlightModelRegistry` the
    PyQt6 Shot Tracer iterates over (issue #7456).
    """
    return FlightModelListResponse(
        models=[
            FlightModelInfo(
                key=model_type,
                name=model.name,
                description=model.description,
                reference=model.reference,
            )
            for model_type in FlightModelType
            for model in (FlightModelRegistry.get_model(model_type),)
        ]
    )


@router.post("/simulate", response_model=BallFlightSimulationResponse)
@handle_api_errors
async def simulate_ball_flight(
    payload: BallFlightSimulationRequest,
) -> BallFlightSimulationResponse:
    """Simulate ball flight for headless and batch clients.

    Accepts either a single ``model_name`` (legacy) or a ``models`` list
    for multi-model overlay comparison. See issues #7218 and #7456.
    """
    results = [_simulate_one(mt, payload) for mt in payload.requested_models()]
    first = results[0]
    return BallFlightSimulationResponse(
        model_name=first.model_name,
        model_key=first.model_key,
        coefficients=first.coefficients,
        trajectory=first.trajectory,
        summary=first.summary,
        results=results,
    )


# =============================================================================
# Import — ADR-0047 H3 (issue #9352)
# =============================================================================


class ImportBallFlightTrajectoryRequest(BaseModel):
    """One ``swing_sim.ball_flight_trajectory/1`` record to import for overlay.

    ``record`` is accepted as an opaque JSON object rather than a typed
    model on purpose: the wire's own shape is validated by the vendored
    Tools reader (fail-closed), not re-declared here. Passing it through
    unmodified means every field the reader checks — unknown fields,
    missing fields, malformed provenance, non-monotone samples — is
    enforced exactly as the wire defines it, from either flight-model
    family (issue #9352, ADR-0047).
    """

    record: dict[str, Any] = Field(
        ...,
        description=(
            "A ball_flight_trajectory/1 JSON record produced by either "
            "flight-model family (ud.flight_models or swing_sim.flight)."
        ),
    )


class ImportedTrajectorySample(BaseModel):
    """Single imported trajectory sample, already in the plot frame."""

    time_s: float
    position_m: list[float]
    velocity_mps: list[float] | None = None


class ImportedBallFlightResponse(BaseModel):
    """An accepted import, in the same shape the page already plots.

    ``model_name`` and ``trajectory``/``summary`` mirror
    :class:`BallFlightModelResult` so the page can render an imported
    curve through the same 3D scene, profile charts, and metrics table
    as a computed one. ``model_family`` and ``parameter_digest`` carry
    the provenance the wire mandates; the page always labels an
    imported curve with both ``model_family`` and ``model_name``
    (ADR-0047), never ``model_name`` alone, so it is never confused
    with a computed curve from the UD registry.
    """

    model_name: str
    #: Stable, collision-free chart/legend key: computed curves key by a
    #: bare ``FlightModelType`` value (e.g. ``"waterloo_penner"``); this
    #: is always ``"<model_family>:<model_name>"`` instead.
    model_key: str
    model_family: str
    parameter_digest: str
    source_id: str
    frame_id: str
    trajectory: list[ImportedTrajectorySample]
    summary: BallFlightSummary


def _imported_response(
    imported: ImportedBallFlightTrajectory,
) -> ImportedBallFlightResponse:
    """Build the API response from a validated, frame-converted import."""
    trajectory = [
        ImportedTrajectorySample(
            time_s=sample.time_s,
            position_m=list(sample.position_m),
            velocity_mps=(
                list(sample.velocity_mps) if sample.velocity_mps is not None else None
            ),
        )
        for sample in imported.samples
    ]
    summary = summarize_imported_trajectory(imported)
    key_suffix = (
        f":{imported.source_id}"
        if imported.source_id
        else f":{imported.parameter_digest[:8]}"
        if imported.parameter_digest
        else ""
    )
    return ImportedBallFlightResponse(
        model_name=imported.model_name,
        model_key=f"{imported.model_family}:{imported.model_name}{key_suffix}",
        model_family=imported.model_family,
        parameter_digest=imported.parameter_digest,
        source_id=imported.source_id,
        frame_id=imported.frame_id,
        trajectory=trajectory,
        summary=BallFlightSummary(
            carry_m=summary.carry_m,
            apex_m=summary.apex_m,
            flight_time_s=summary.flight_time_s,
            landing_angle_deg=summary.landing_angle_deg,
            lateral_deviation_m=summary.lateral_deviation_m,
        ),
    )


@router.post("/import", response_model=ImportedBallFlightResponse)
@handle_api_errors
async def import_ball_flight_trajectory(
    payload: ImportBallFlightTrajectoryRequest,
) -> ImportedBallFlightResponse:
    """Import a ``ball_flight_trajectory/1`` record for overlay (ADR-0047 H3).

    Validated fail-closed through the vendored Tools reader — never
    reimplemented here. Records from either flight-model family are
    accepted; the record's declared frame is converted explicitly into
    the page's plot frame, and a frame this endpoint has not implemented
    is refused by name rather than silently mis-plotted. Every refusal
    (unknown/missing wire fields, malformed provenance, non-monotone
    samples, an unsupported frame, or an unresolvable Tools checkout)
    surfaces as a 400 whose ``detail`` is the specific reason. See
    issue #9352.
    """
    imported = import_trajectory_record(payload.record)
    return _imported_response(imported)
