"""Baghouse calculator router.  See issue #613."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..contracts.baghouse import BaghouseRequest, BaghouseResponse

router = APIRouter(prefix="/api/calc/baghouse", tags=["baghouse"])


@router.post("", response_model=BaghouseResponse)
def calculate_baghouse(request: BaghouseRequest) -> BaghouseResponse:
    """Calculate baghouse filter performance."""
    from upstream_drift_tools.process_calculators import BaghouseCalculator

    calc = BaghouseCalculator()

    try:
        result = calc.calculate(
            gas_flow_kg_s=request.gas_flow_kg_s,
            inlet_temp_k=request.inlet_temp_k,
            pressure_pa=request.pressure_pa,
            composition=request.composition,
            solid_carbon_in_kg_hr=request.solid_carbon_in_kg_hr,
            ash_in_kg_hr=request.ash_in_kg_hr,
            carbon_removal_efficiency=request.carbon_removal_efficiency,
            ash_removal_efficiency=request.ash_removal_efficiency,
            heat_loss_w=request.heat_loss_w,
            drum_volume_m3=request.drum_volume_m3,
            solid_density_kg_m3=request.solid_density_kg_m3,
            bag_area_ft2=request.bag_area_ft2,
        )
    except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return BaghouseResponse(
        carbon_removed_rate_kg_hr=result.carbon_removed_rate,
        ash_removed_rate_kg_hr=result.ash_removed_rate,
        total_solids_removed_rate_kg_hr=result.total_solids_removed_rate,
        drum_fill_time_hours=result.drum_fill_time_hours,
        drum_fill_time_days=result.drum_fill_time_days,
        carbon_only_fill_time_hours=result.carbon_only_fill_time_hours,
        ash_only_fill_time_hours=result.ash_only_fill_time_hours,
        clean_gas_flow_rate_kg_hr=result.clean_gas_flow_rate,
        flow_acfm=result.flow_acfm,
        flow_scfm=result.flow_scfm,
        air_to_cloth_ratio=result.air_to_cloth_ratio,
        outlet_temperature_c=result.outlet_temperature_c,
        ash_stream_composition=result.ash_stream_composition,
        removal_efficiency=result.removal_efficiency,
    )
