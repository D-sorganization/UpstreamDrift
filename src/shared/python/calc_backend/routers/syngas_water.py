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
        from shared.python.sidekick.process_calculators.syngas_water_calculator import (
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
            recommended_temperature_c=_sanitize(
                float(risk["recommended_temperature_c"])
            ),
        ),
    )


def _fallback_calculate(request: SyngasWaterRequest) -> SyngasWaterResponse:
    """Basic fallback when full calculator is not available.

    All physical constants are imported from the canonical shared modules so
    the fallback path cannot silently drift from the authoritative values
    (issue #3678).  The Antoine vapor-pressure and inverse-dewpoint formulas
    delegate to the shared :mod:`water_vapor_pressure` kernel.
    """
    from shared.python.sidekick.process_calculators.constants import (
        ANTOINE_WATER_A,
        ANTOINE_WATER_B,
        ANTOINE_WATER_C,
        CELSIUS_TO_KELVIN_OFFSET,
        MW_WATER_GMOL,
    )
    from shared.python.sidekick.process_calculators.water_vapor_pressure import (
        antoine_pressure_pa,
        antoine_temperature_c,
    )
    from shared.python.sidekick.utils.unit_constants import (
        MOLAR_VOLUME_STP_OLD,
        R_UNIVERSAL,
    )

    A, B, C = ANTOINE_WATER_A, ANTOINE_WATER_B, ANTOINE_WATER_C
    r_gas = R_UNIVERSAL
    c_to_k = CELSIUS_TO_KELVIN_OFFSET
    mw_water_g_mol = MW_WATER_GMOL

    # lb/mmscf conversion: standard molar volume at 60°F/14.696 psia (385.5
    # scf/lb-mol) and pounds-to-grams (453.592 g/lb).
    _scf_per_lbmol_60f = 385.5
    _grams_per_pound = 453.592

    # Antoine equation for water vapor pressure (shared kernel returns Pa).
    vp_bar = antoine_pressure_pa(A, B, C, request.temperature_c) / 1e5

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
    mg_nm3 = mole_frac * (mw_water_g_mol / MOLAR_VOLUME_STP_OLD) * 1000
    g_m3 = (
        mole_frac
        * (mw_water_g_mol * request.pressure_bar * 1e5)
        / (r_gas * (request.temperature_c + c_to_k))
    )
    lb_mmscf = ppmv * mw_water_g_mol / (_scf_per_lbmol_60f * _grams_per_pound) * 1e6

    # Dew point via inverse Antoine (shared kernel takes Pa).
    pp = mole_frac * request.pressure_bar
    dew_c = antoine_temperature_c(A, B, C, max(pp * 1e5, 1e-10)) if pp > 0 else -c_to_k

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
