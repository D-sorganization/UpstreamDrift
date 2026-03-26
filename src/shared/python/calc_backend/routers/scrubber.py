"""Scrubber calculator router.  See issue #613."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from ..contracts.scrubber import ScrubberRequest, ScrubberResponse

router = APIRouter(prefix="/api/calc/scrubber", tags=["scrubber"])


def _as_float(value: Any, field_name: str) -> float:
    """Convert calculator outputs to float for strict response contracts."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid numeric value for {field_name}: {value}",
            ) from exc
    raise HTTPException(
        status_code=422, detail=f"Invalid type for {field_name}: {type(value).__name__}"
    )


@router.post("", response_model=ScrubberResponse)
def calculate_scrubber(request: ScrubberRequest) -> ScrubberResponse:
    """Calculate packed-bed scrubber column sizing and caustic requirements."""
    from upstream_drift_tools.process_calculators.scrubber_calculator import (
        PACKING_DATABASE,
        WATER_DENSITY,
        WATER_VISCOSITY,
        calculate_caustic_requirement,
        calculate_column_diameter,
        calculate_flooding_velocity,
        calculate_gas_density,
        calculate_gas_viscosity,
    )

    packing = PACKING_DATABASE.get(request.packing_type)
    if packing is None:
        available = ", ".join(sorted(PACKING_DATABASE.keys()))
        raise HTTPException(
            status_code=422,
            detail=f"Unknown packing type '{request.packing_type}'. Available: {available}",
        )

    try:
        gas_density = calculate_gas_density(
            request.gas_temperature_k,
            request.gas_pressure_pa,
            request.gas_molecular_weight,
        )
        # Gas viscosity calculated for reference but not directly needed by
        # the column-sizing functions called below.
        _ = calculate_gas_viscosity(request.gas_temperature_k, request.gas_molecular_weight)

        # Liquid mass flux (assume per unit area ~ liquid_flow / column area, start with 1 m2)
        liquid_mass_flux = request.liquid_flow_kg_hr / 3600.0  # kg/s per m2 placeholder

        flooding_velocity = calculate_flooding_velocity(
            liquid_mass_flux=liquid_mass_flux,
            gas_density=gas_density,
            liquid_density=WATER_DENSITY,
            packing=packing,
            liquid_viscosity=WATER_VISCOSITY,
        )

        column_result = calculate_column_diameter(
            gas_flow_kg_hr=request.gas_flow_kg_hr,
            gas_density=gas_density,
            flooding_velocity=flooding_velocity,
            percent_of_flood=request.percent_of_flood,
        )

        caustic_result = calculate_caustic_requirement(
            acid_gas_removed=request.acid_gas_removed_kg_hr,
            caustic_concentration=request.caustic_concentration_pct,
        )
    except (ValueError, ZeroDivisionError, OverflowError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ScrubberResponse(
        gas_density_kg_m3=gas_density,
        flooding_velocity_m_s=flooding_velocity,
        design_velocity_m_s=_as_float(
            column_result.get("design_velocity_m_s", 0.0), "design_velocity_m_s"
        ),
        column_diameter_m=_as_float(column_result.get("diameter_m", 0.0), "diameter_m"),
        column_diameter_ft=_as_float(column_result.get("diameter_ft", 0.0), "diameter_ft"),
        cross_section_m2=_as_float(column_result.get("cross_section_m2", 0.0), "cross_section_m2"),
        caustic_requirement=caustic_result,
    )
