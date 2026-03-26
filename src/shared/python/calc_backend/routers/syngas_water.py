"""Syngas water calculator router.  See issue #608."""

from __future__ import annotations

import math

from fastapi import APIRouter

from ..contracts.syngas_water import (
    CondensationRiskOut,
    SyngasWaterRequest,
    SyngasWaterResponse,
    WaterContentOut,
)

router = APIRouter(prefix="/api/calc/syngas-water", tags=["syngas-water"])


def _sanitize(v: float) -> float:
    """Replace NaN / Inf with 0.0 to avoid JSON serialization errors."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return 0.0
    return float(v)


@router.post("", response_model=SyngasWaterResponse)
def calculate_syngas_water(request: SyngasWaterRequest) -> SyngasWaterResponse:
    """Calculate water content and condensation risk in syngas."""
    try:
        from upstream_drift_tools.process_calculators.syngas_water_calculator import (
            SyngasWaterCalculator,
            estimate_condensation_risk,
        )

        calculator = SyngasWaterCalculator()
        result = calculator.calculate_water_content(
            request.temperature_c,
            request.pressure_bar,
            request.composition_key,
            request.method,
        )

        risk = estimate_condensation_risk(request.temperature_c, request.pressure_bar)
    except ImportError:
        # Fallback: basic calculation without the full calculator
        return _fallback_calculate(request)
    return SyngasWaterResponse(
        water_content=WaterContentOut(
            mole_fraction_water=_sanitize(result.mole_fraction_water),
            water_content_mg_per_nm3=_sanitize(result.water_content_mg_per_nm3),
            water_content_ppmv=_sanitize(result.water_content_ppmv),
            water_content_g_per_m3=_sanitize(result.water_content_g_per_m3),
            water_content_lb_per_mmscf=_sanitize(result.water_content_lb_per_mmscf),
            vapor_pressure_bar=_sanitize(result.vapor_pressure_bar),
            dew_point_c=_sanitize(result.dew_point_c),
        ),
        risk_assessment=CondensationRiskOut(
            temperature_margin_c=_sanitize(float(risk["temperature_margin_c"])),
            condensation_risk=str(risk["condensation_risk"]),
            recommended_temperature_c=_sanitize(float(risk["recommended_temperature_c"])),
        ),
    )


def _fallback_calculate(request: SyngasWaterRequest) -> SyngasWaterResponse:
    """Basic fallback when full calculator is not available."""
    import math

    # Antoine equation for water vapor pressure
    A, B, C = 8.07131, 1730.63, 233.426
    log_p = A - B / (C + request.temperature_c)
    vp_bar = math.pow(10, log_p) * 133.322 / 1e5

    # Simple water mole fraction from composition preset
    water_pct_map = {
        "typical_syngas": 10.0,
        "biomass_syngas": 20.0,
        "coal_syngas": 5.0,
        "natural_gas_reforming": 30.0,
    }
    water_pct = water_pct_map.get(request.composition_key, 10.0)
    mole_frac = min(water_pct / 100.0, vp_bar / request.pressure_bar)

    ppmv = mole_frac * 1e6
    mg_nm3 = mole_frac * (18.015 / 0.022414) * 1000
    g_m3 = (
        mole_frac
        * (18.015 * request.pressure_bar * 1e5)
        / (8.314 * (request.temperature_c + 273.15))
    )
    lb_mmscf = ppmv * 18.015 / (385.5 * 453.592) * 1e6

    # Dew point via inverse Antoine
    pp = mole_frac * request.pressure_bar
    dew_c = B / (A - math.log10(max(pp * 1e5 / 133.322, 1e-10))) - C if pp > 0 else -273.15

    margin = request.temperature_c - dew_c
    if margin < 0:
        risk_str = "Critical - Condensation Occurring"
    elif margin < 5:
        risk_str = "High"
    elif margin < 15:
        risk_str = "Medium"
    else:
        risk_str = "Low"

    return SyngasWaterResponse(
        water_content=WaterContentOut(
            mole_fraction_water=mole_frac,
            water_content_mg_per_nm3=mg_nm3,
            water_content_ppmv=ppmv,
            water_content_g_per_m3=g_m3,
            water_content_lb_per_mmscf=lb_mmscf,
            vapor_pressure_bar=vp_bar,
            dew_point_c=dew_c,
        ),
        risk_assessment=CondensationRiskOut(
            temperature_margin_c=margin,
            condensation_risk=risk_str,
            recommended_temperature_c=dew_c + 15,
        ),
    )
