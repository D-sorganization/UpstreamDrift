"""Shared biomechanical calculations for recorded and simulated trajectories."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from src.shared.python.analysis.biomechanics_display import prepare_biomechanics_plot
from src.shared.python.anatomical_angles import event_table as anatomical_events
from src.shared.python.anatomical_angles.wrist import wrist_forearm_angles
from src.shared.python.biomechanics.joint_conventions import (
    AnatomicalFrame,
    RotationConvention,
    matrix_to_orientations,
    orientations_to_matrix,
)

from ..dependencies import get_simulation_service

router = APIRouter(prefix="/biomechanics", tags=["biomechanics"])


class DisplayRequest(BaseModel):
    """Renderer options; science remains in the shared calculation layer."""

    model_config = ConfigDict(extra="forbid")
    result: dict[str, Any]
    selected: list[str] | None = None
    angle_unit: Literal["deg", "rad"] = "deg"


class ConversionRequest(BaseModel):
    """Explicit orientation representation and Euler conventions."""

    model_config = ConfigDict(extra="forbid")
    values: list[Any] = Field(max_length=100_000)
    source_representation: str
    target_representation: str
    source_sequence: str = "XYZ"
    target_sequence: str = "XYZ"
    source_degrees: bool = False
    target_degrees: bool = False


class SegmentFrames(BaseModel):
    """World pose series of one calibrated segment (rotations ``(N, 3, 3)``)."""

    model_config = ConfigDict(extra="forbid")
    rotation_world: list[Any] = Field(max_length=100_000)
    origin_world: list[Any] = Field(max_length=100_000)
    calibration_id: str


class ArmFrames(BaseModel):
    """Segment frames of one side; ``radius`` defaults to ``forearm``."""

    model_config = ConfigDict(extra="forbid")
    humerus: SegmentFrames
    forearm: SegmentFrames
    hand: SegmentFrames
    radius: SegmentFrames | None = None


class AnatomicalEventTableRequest(BaseModel):
    """Calibrated arm frames, swing-event frames and export provenance."""

    model_config = ConfigDict(extra="forbid")
    time_s: list[float] = Field(max_length=100_000)
    events: dict[str, int]
    handedness: str
    event_source: str
    receipt_path: str
    arms: dict[str, ArmFrames]
    format: Literal["json", "csv"] = "json"


def _frame(segment: SegmentFrames) -> AnatomicalFrame:
    return AnatomicalFrame(**segment.model_dump())


def _anatomical_event_table(
    payload: AnatomicalEventTableRequest,
) -> anatomical_events.AnatomicalEventTable:
    arms = {
        side: wrist_forearm_angles(
            _frame(arm.humerus),
            _frame(arm.forearm),
            _frame(arm.hand),
            side,
            receipt_path=payload.receipt_path,
            radius=None if arm.radius is None else _frame(arm.radius),
        )
        for side, arm in payload.arms.items()
    }
    return anatomical_events.anatomical_event_table(
        arms,
        payload.events,
        payload.time_s,
        handedness=payload.handedness,
        event_source=payload.event_source,
    )


def _compute_biomechanics(payload: dict[str, Any]) -> dict[str, Any]:
    from src.shared.python.biomechanics.golf_trajectory import (
        compute_golf_metrics,
        golf_trajectory_from_dict,
    )

    try:
        trajectory = golf_trajectory_from_dict(payload)
        return compute_golf_metrics(trajectory).to_dict()
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/compute")
def compute(payload: dict[str, Any]) -> dict[str, Any]:
    """Compute SI channels with definitions, availability and source provenance."""
    return _compute_biomechanics(payload)


@router.post("/display")
def display(payload: DisplayRequest) -> dict[str, Any]:
    """Flatten vector channels and convert display units without recomputation."""
    try:
        return prepare_biomechanics_plot(
            payload.result, selected=payload.selected, angle_unit=payload.angle_unit
        )
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/convert")
def convert(payload: ConversionRequest) -> dict[str, Any]:
    """Convert orientations with explicit singularity diagnostics."""
    try:
        source = RotationConvention(payload.source_sequence, payload.source_degrees)
        target = RotationConvention(payload.target_sequence, payload.target_degrees)
        matrices = orientations_to_matrix(
            payload.values, payload.source_representation, source
        )
        result = matrix_to_orientations(matrices, payload.target_representation, target)
        return {
            "values": result.values.tolist(),
            "singular": result.singular.tolist(),
            "representation": result.representation,
            "sequence": target.sequence,
            "degrees": target.degrees,
        }
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/anatomical-event-table", response_model=None)
def anatomical_event_table(
    payload: AnatomicalEventTableRequest,
) -> dict[str, Any] | PlainTextResponse:
    """Wrist and forearm angles at swing events, as the desktop export.

    Angles are computed from the posted calibrated frames by the shared
    ``anatomical_angles`` functions, so the JSON (``EVENT_TABLE_JSON_SCHEMA``)
    and CSV bodies equal the desktop export for the same bundle.
    """
    try:
        table = _anatomical_event_table(payload)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if payload.format == "csv":
        return PlainTextResponse(anatomical_events.to_csv(table), media_type="text/csv")
    return anatomical_events.to_json_dict(table)


@router.get("/results")
def session_results(service: Any = Depends(get_simulation_service)) -> dict[str, Any]:
    """Analyze the completed session's explicitly bound segment recording."""
    payload = service.get_biomechanics_payload()
    if payload is None:
        raise HTTPException(
            status_code=409,
            detail="No calibrated segment recording. Configure model bindings before recording, or import a canonical trajectory.",
        )
    return _compute_biomechanics(payload)


@router.post("/bindings")
def configure_binding(
    payload: dict[str, Any], service: Any = Depends(get_simulation_service)
) -> dict[str, str]:
    """Apply an explicit anatomical model binding to the next simulation."""
    try:
        service.configure_biomechanics(payload)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"status": "configured", "applies_to": "next_simulation"}
