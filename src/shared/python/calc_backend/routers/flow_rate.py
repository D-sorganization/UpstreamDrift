"""Flow rate converter router.  See issue #608.

DRY: reuses the canonical conversion tables from
upstream_drift_tools.calculators.conversion.flow_rate_converter
instead of duplicating conversion factors.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from upstream_drift_tools.calculators.conversion.flow_rate_converter import (
    MASS_FLOW_CONVERSIONS,
    MOLAR_FLOW_CONVERSIONS,
    VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S,
)

from ..contracts.flow_rate import FlowRateConvertRequest, FlowRateConvertResponse

router = APIRouter(prefix="/api/calc/flow-rate", tags=["flow-rate"])

# Re-use canonical tables from flow_rate_converter (single source of truth)
_TABLES: dict[str, dict[str, float]] = {
    "mass": MASS_FLOW_CONVERSIONS,
    "molar": MOLAR_FLOW_CONVERSIONS,
    "volumetric": VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S,
}


@router.post("", response_model=FlowRateConvertResponse)
def convert_flow_rate(request: FlowRateConvertRequest) -> FlowRateConvertResponse:
    """Convert a flow rate value between compatible units."""
    table = _TABLES.get(request.category)
    if table is None:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown category '{request.category}'. "
            f"Must be one of: {', '.join(_TABLES)}",
        )

    if request.from_unit not in table:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown from_unit '{request.from_unit}' for category "
            f"'{request.category}'. Valid units: {', '.join(table)}",
        )
    if request.to_unit not in table:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown to_unit '{request.to_unit}' for category "
            f"'{request.category}'. Valid units: {', '.join(table)}",
        )

    # Convert: source -> base SI -> target
    base = request.value * table[request.from_unit]
    result = base / table[request.to_unit]

    return FlowRateConvertResponse(
        result=result,
        from_unit=request.from_unit,
        to_unit=request.to_unit,
        category=request.category,
    )
