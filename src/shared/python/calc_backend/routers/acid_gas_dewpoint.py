"""Acid gas dewpoint calculator router.  See issue #613."""

from __future__ import annotations

import math

from fastapi import APIRouter, HTTPException

from ..contracts.acid_gas_dewpoint import (
    AcidGasDewpointRequest,
    AcidGasDewpointResponse,
    DewpointComponentOut,
)

router = APIRouter(prefix="/api/calc/acid-gas-dewpoint", tags=["acid-gas-dewpoint"])


def _safe_float(value: float) -> float | None:
    """Convert NaN/Inf to None for JSON serialisation."""
    if value is None or math.isnan(value) or math.isinf(value):
        return None
    return value


@router.post("", response_model=AcidGasDewpointResponse)
def calculate_acid_gas_dewpoint(
    request: AcidGasDewpointRequest,
) -> AcidGasDewpointResponse:
    """Calculate acid gas dewpoint for a mixture."""
    from upstream_drift_tools.process_calculators.acid_gas_dewpoint_calculator import (
        AcidGasComposition,
        AcidGasDewpointCalculator,
    )

    calc = AcidGasDewpointCalculator()
    composition = AcidGasComposition(
        h2o=request.h2o_fraction,
        hf=request.hf_fraction,
        hcl=request.hcl_fraction,
        h2s=request.h2s_fraction,
    )

    try:
        result = calc.calculate_dewpoint_mixture(
            temperature_c=request.temperature_c,
            pressure_bar=request.pressure_bar,
            composition=composition,
            method=request.method,
        )
    except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    components = {
        "H2O": DewpointComponentOut(
            dewpoint_c=_safe_float(result.h2o_dewpoint_c),
            vapor_pressure_pa=result.h2o_vapor_pressure_pa,
            partial_pressure_pa=result.h2o_partial_pressure_pa,
        ),
        "HF": DewpointComponentOut(
            dewpoint_c=_safe_float(result.hf_dewpoint_c),
            vapor_pressure_pa=result.hf_vapor_pressure_pa,
            partial_pressure_pa=result.hf_partial_pressure_pa,
        ),
        "HCl": DewpointComponentOut(
            dewpoint_c=_safe_float(result.hcl_dewpoint_c),
            vapor_pressure_pa=result.hcl_vapor_pressure_pa,
            partial_pressure_pa=result.hcl_partial_pressure_pa,
        ),
        "H2S": DewpointComponentOut(
            dewpoint_c=_safe_float(result.h2s_dewpoint_c),
            vapor_pressure_pa=result.h2s_vapor_pressure_pa,
            partial_pressure_pa=result.h2s_partial_pressure_pa,
        ),
    }

    return AcidGasDewpointResponse(
        overall_dewpoint_c=_safe_float(result.overall_dewpoint_c),
        limiting_component=result.limiting_component,
        dewpoint_margin_c=_safe_float(result.dewpoint_margin_c),
        condensation_risk=result.condensation_risk,
        components=components,
        warnings=result.warnings,
        calculation_method=result.calculation_method,
    )
