"""Focused unit tests for syngas water router unit conversions (#6004, #14934)."""

from __future__ import annotations

import pytest
from src.shared.python.calc_backend.contracts.syngas_water import (
    SyngasWaterRequest,
)
from src.shared.python.calc_backend.routers.syngas_water import (
    _fallback_calculate,
    calculate_syngas_water,
)
from src.shared.python.sidekick.process_calculators.syngas_water_calculator import (
    water_content_units,
)

pytestmark = pytest.mark.unit


@pytest.mark.unit
def test_saturated_syngas_at_40c_1atm_returns_51_g_per_m3() -> None:
    """Saturated syngas at 40 °C, 1.01325 bar: g/m³ is 51.1 ± 1%, mg/Nm³ == g/m³(0 °C, 1 atm) * 1000."""
    req = SyngasWaterRequest(
        temperature_c=40.0, pressure_bar=1.01325, composition_key="typical_syngas"
    )
    # Test router fallback
    resp_fallback = _fallback_calculate(req)
    assert 50.589 <= resp_fallback.water_content.water_content_g_per_m3 <= 51.611, (
        f"Expected ~51.1 g/m³ (±1%), got {resp_fallback.water_content.water_content_g_per_m3}"
    )

    # Test main router endpoint
    resp_main = calculate_syngas_water(req)
    assert 50.589 <= resp_main.water_content.water_content_g_per_m3 <= 51.611, (
        f"Expected ~51.1 g/m³ (±1%), got {resp_main.water_content.water_content_g_per_m3}"
    )

    # Cross-unit identity: mg/Nm³ == g/m³(0 °C, 1 atm) * 1000
    y_water = resp_fallback.water_content.mole_fraction_water
    units_normal = water_content_units(
        y_water=y_water,
        p_pa=101325.0,
        t_k=273.15,
    )
    assert float(units_normal.g_m3) * 1000.0 == pytest.approx(
        float(units_normal.mg_nm3), rel=1e-6
    )
    assert float(resp_fallback.water_content.water_content_mg_per_nm3) == pytest.approx(
        float(units_normal.mg_nm3), rel=1e-6
    )


@pytest.mark.unit
def test_thousand_ppmv_gives_47_47_lb_per_mmscf_in_router_fallback() -> None:
    """1000 ppmv (y_water = 0.001) gives lb_mmscf == 47.47 ± 0.1 (60F_14.696psia basis)."""
    from src.shared.python.sidekick.process_calculators.constants import (
        ANTOINE_WATER_A,
        ANTOINE_WATER_B,
        ANTOINE_WATER_C,
    )
    from src.shared.python.sidekick.process_calculators.water_vapor_pressure import (
        antoine_pressure_pa,
    )

    # Choose pressure so that mole_frac = vp_bar / p_bar = 0.001
    vp_pa = antoine_pressure_pa(ANTOINE_WATER_A, ANTOINE_WATER_B, ANTOINE_WATER_C, 25.0)
    p_bar = (vp_pa / 1e5) / 0.001

    req = SyngasWaterRequest(
        temperature_c=25.0, pressure_bar=p_bar, composition_key="typical_syngas"
    )
    resp = _fallback_calculate(req)
    assert resp.water_content.water_content_ppmv == pytest.approx(1000.0, rel=1e-3)
    assert 47.37 <= resp.water_content.water_content_lb_per_mmscf <= 47.57, (
        f"Expected 47.47 ± 0.1 lb/MMscf, got {resp.water_content.water_content_lb_per_mmscf}"
    )


@pytest.mark.unit
def test_standard_basis_field_present_on_syngas_water_response() -> None:
    """standard_basis field is present on SyngasWaterResponse and defaults to 60F_14.696psia."""
    req = SyngasWaterRequest(temperature_c=40.0, pressure_bar=1.0)
    resp_fallback = _fallback_calculate(req)
    assert resp_fallback.water_content.standard_basis == "60F_14.696psia"

    resp_main = calculate_syngas_water(req)
    assert resp_main.water_content.standard_basis == "60F_14.696psia"
