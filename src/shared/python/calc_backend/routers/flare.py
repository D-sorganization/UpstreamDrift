"""Flare calculator router.  See issue #613."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..contracts.flare import (
    FlareDesignOut,
    FlareRequest,
    FlareResponse,
    RadiationZonesOut,
)

router = APIRouter(prefix="/api/calc/flare", tags=["flare"])


@router.post("", response_model=FlareResponse)
def calculate_flare(request: FlareRequest) -> FlareResponse:
    """Calculate flare size, radiation zones, and combustion efficiency."""
    from upstream_drift_tools.process_calculators import FlareCalculator

    calc = FlareCalculator()

    try:
        design = calc.calculate_flare_size(
            total_flow=request.total_flow_kg_hr,
            gas_composition=request.gas_composition,
            temperature=request.temperature_k,
            pressure=request.pressure_bar,
        )
        zones = calc.calculate_radiation_zones(design)
        efficiency = calc.calculate_combustion_efficiency(
            gas_composition=request.gas_composition,
            temperature=request.temperature_k,
            pressure=request.pressure_bar,
        )
    except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return FlareResponse(
        design=FlareDesignOut(
            height_m=design.height,
            diameter_m=design.diameter,
            exit_velocity_m_s=design.exit_velocity,
            heat_release_kw=design.heat_release,
            radiation_intensity_kw_m2=design.radiation_intensity,
        ),
        radiation_zones=RadiationZonesOut(
            lethal_m=zones["lethal"],
            damage_m=zones["damage"],
            safe_m=zones["safe"],
            comfort_m=zones["comfort"],
        ),
        combustion_efficiency=efficiency,
    )
