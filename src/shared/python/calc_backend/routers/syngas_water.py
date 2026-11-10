"""Syngas water calculator router.  See issue #608."""

from __future__ import annotations

import math

from fastapi import APIRouter, HTTPException

from ..contracts.syngas_water import (
    CondensationRiskOut,
    SyngasCalculationStatus,
    SyngasNumericalFailureError,
    SyngasWaterRequest,
    SyngasWaterResponse,
    WaterContentOut,
)

router = APIRouter(prefix="/api/calc/syngas-water", tags=["syngas-water"])


def _require_finite(v: float, field_name: str = "") -> float:
    """Validate that a value is finite, raising SyngasNumericalFailureError if not."""
    if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
        val = float(v) if isinstance(v, (int, float)) else float("nan")
        raise SyngasNumericalFailureError(field_name, val)
    return float(v)


_sanitize = _require_finite


@router.post("", response_model=SyngasWaterResponse)
def calculate_syngas_water(request: SyngasWaterRequest) -> SyngasWaterResponse:
    """Calculate water content and condensation risk in syngas."""
    try:
        try:
            from src.shared.python.sidekick.process_calculators.syngas_water_calculator import (
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

            risk = estimate_condensation_risk(
                request.temperature_c, request.pressure_bar
            )
        except ImportError:
            # Fallback: basic calculation without the full calculator
            return _fallback_calculate(request)

        _s = _sanitize
        r = result
        return SyngasWaterResponse(
            status=SyngasCalculationStatus.SUCCEEDED,
            water_content=WaterContentOut(
                mole_fraction_water=_s(r.mole_fraction_water, "mole_fraction_water"),
                water_content_mg_per_nm3=_s(
                    r.water_content_mg_per_nm3, "water_content_mg_per_nm3"
                ),
                water_content_ppmv=_s(r.water_content_ppmv, "water_content_ppmv"),
                water_content_g_per_m3=_s(
                    r.water_content_g_per_m3, "water_content_g_per_m3"
                ),
                water_content_lb_per_mmscf=_s(
                    r.water_content_lb_per_mmscf, "water_content_lb_per_mmscf"
                ),
                vapor_pressure_bar=_s(r.vapor_pressure_bar, "vapor_pressure_bar"),
                dew_point_c=_s(r.dew_point_c, "dew_point_c"),
                standard_basis=getattr(r, "standard_basis", "60F_14.696psia"),
            ),
            risk_assessment=CondensationRiskOut(
                temperature_margin_c=_s(
                    float(risk["temperature_margin_c"]), "temperature_margin_c"
                ),
                condensation_risk=str(risk["condensation_risk"]),
                recommended_temperature_c=_s(
                    float(risk["recommended_temperature_c"]),
                    "recommended_temperature_c",
                ),
            ),
        )
    except SyngasNumericalFailureError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "status": exc.status.value,
                "code": "NON_FINITE_OUTPUT",
                "field": exc.field_name,
                "message": str(exc),
            },
        ) from exc


def _fallback_calculate(request: SyngasWaterRequest) -> SyngasWaterResponse:
    """Basic fallback when full calculator is not available.

    All physical constants are imported from the canonical shared modules so
    the fallback path cannot silently drift from the authoritative values
    (issue #3678).  The Antoine vapor-pressure and inverse-dewpoint formulas
    delegate to the shared :mod:`water_vapor_pressure` kernel.
    """
    from src.shared.python.sidekick.process_calculators.constants import (
        ANTOINE_WATER_A,
        ANTOINE_WATER_B,
        ANTOINE_WATER_C,
        CELSIUS_TO_KELVIN_OFFSET,
        MW_WATER_GMOL,
    )
    from src.shared.python.sidekick.process_calculators.syngas_water_calculator import (
        water_content_units,
    )
    from src.shared.python.sidekick.process_calculators.water_vapor_pressure import (
        antoine_pressure_pa,
        antoine_temperature_c,
    )

    A, B, C = ANTOINE_WATER_A, ANTOINE_WATER_B, ANTOINE_WATER_C
    c_to_k = CELSIUS_TO_KELVIN_OFFSET
    mw_water_g_mol = MW_WATER_GMOL

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

    t_k = request.temperature_c + c_to_k
    p_pa = request.pressure_bar * 1e5
    units = water_content_units(
        y_water=mole_frac,
        p_pa=p_pa,
        t_k=t_k,
        mw_water=mw_water_g_mol,
    )

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

    _s = _sanitize
    return SyngasWaterResponse(
        status=SyngasCalculationStatus.SUCCEEDED,
        water_content=WaterContentOut(
            mole_fraction_water=_s(mole_frac, "mole_fraction_water"),
            water_content_mg_per_nm3=_s(units.mg_nm3, "water_content_mg_per_nm3"),
            water_content_ppmv=_s(units.ppmv, "water_content_ppmv"),
            water_content_g_per_m3=_s(units.g_m3, "water_content_g_per_m3"),
            water_content_lb_per_mmscf=_s(units.lb_mmscf, "water_content_lb_per_mmscf"),
            vapor_pressure_bar=_s(vp_bar, "vapor_pressure_bar"),
            dew_point_c=_s(dew_c, "dew_point_c"),
            standard_basis=units.standard_basis,
        ),
        risk_assessment=CondensationRiskOut(
            temperature_margin_c=_s(margin, "temperature_margin_c"),
            condensation_risk=risk_str,
            recommended_temperature_c=_s(dew_c + 15, "recommended_temperature_c"),
        ),
    )
