"""Shared biomechanical calculations for recorded and simulated trajectories."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.shared.python.analysis.biomechanics_display import prepare_biomechanics_plot
from src.shared.python.biomechanics.joint_conventions import (
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
