# ruff: noqa: E501
import numpy as np
import pytest
from upstream_drift_tools.process_calculators.syngas_water_calculator import (
    SyngasComposition,
    SyngasWaterCalculator,
    WaterContentResult,
    estimate_condensation_risk,
    quick_water_content,
)
from upstream_drift_tools.process_calculators.water_vapor_pressure_calculator import (
    WaterVaporPressureCalculator,
)


@pytest.fixture
def calc() -> SyngasWaterCalculator:
    return SyngasWaterCalculator()


def test_syngas_composition_normalization() -> None:
    comp = SyngasComposition(h2=1.0, co=1.0)
    norm = comp.normalize()
    assert norm.h2 == 0.5
    assert norm.co == 0.5
    assert norm.total == 1.0

    comp_zero = SyngasComposition(name="Zero")
    norm_zero = comp_zero.normalize()
    assert norm_zero.total == 0.0
    assert norm_zero.name == "Zero"


def test_syngas_composition_to_dict() -> None:
    comp = SyngasComposition(h2=0.5, co=0.5)
    d = comp.to_dict()
    assert d["H2"] == 0.5
    assert d["Other"] == 0.0


def test_calculate_vapor_pressure_methods(calc: SyngasWaterCalculator) -> None:
    vp_antoine, m_antoine = calc.calculate_vapor_pressure(100.0, "antoine")
    assert vp_antoine > 0

    vp_buck, m_buck = calc.calculate_vapor_pressure(50.0, "buck")
    assert vp_buck > 0

    vp_magnus, m_magnus = calc.calculate_vapor_pressure(20.0, "magnus")
    assert vp_magnus > 0

    vp_iapws, m_iapws = calc.calculate_vapor_pressure(200.0, "iapws")
    assert vp_iapws > 0

    with pytest.raises(ValueError):
        calc.calculate_vapor_pressure(500.0, "iapws")


def test_calculate_vapor_pressure_auto(calc: SyngasWaterCalculator) -> None:
    _, method = calc.calculate_vapor_pressure(50.0, method="auto")
    assert "Magnus" in method
    _, method = calc.calculate_vapor_pressure(-10.0, method="auto")
    assert "Buck" in method
    _, method = calc.calculate_vapor_pressure(200.0, method="auto")
    assert "IAPWS-IF97" in method
    _, method = calc.calculate_vapor_pressure(400.0, method="auto")
    assert "Antoine" in method


def test_magnus_equation_out_of_bounds(calc: SyngasWaterCalculator) -> None:
    with pytest.raises(ValueError, match="Magnus equation valid"):
        calc.calculate_vapor_pressure(-5.0, "magnus")
    with pytest.raises(ValueError, match="Magnus equation valid"):
        calc.calculate_vapor_pressure(105.0, "magnus")


def test_water_vapor_pressure_wrapper() -> None:
    wrapper = WaterVaporPressureCalculator()
    vp = wrapper.calculate_vapor_pressure(100.0, "antoine")
    assert vp > 0


def test_vapor_pressure_fast(calc: SyngasWaterCalculator) -> None:
    vp = calc.vapor_pressure_fast(373.15)
    assert not np.isnan(vp)
    assert vp > 0

    calc2 = SyngasWaterCalculator()
    if hasattr(calc2, "vapor_pressure_table"):
        del calc2.vapor_pressure_table
    vp2 = calc2.vapor_pressure_fast(300.0)
    assert vp2 > 0


def test_calculate_dew_point(calc: SyngasWaterCalculator) -> None:
    vp, _ = calc.calculate_vapor_pressure(20.0, "buck")
    dp = calc.calculate_dew_point(vp, 101325.0)
    assert dp == pytest.approx(20.0, abs=0.1)

    from unittest.mock import patch

    with patch.object(calc, "_buck_equation", return_value=1000.0):
        # Initial guess will evaluate to 1.0 kPa, diff won't change
        # dp_dT will calculate as 0
        dp = calc.calculate_dew_point(2000.0, 101325.0)
        assert dp == 20.0  # Initial guess is returned because it breaks


def test_calculate_water_content(calc: SyngasWaterCalculator) -> None:
    res = calc.calculate_water_content(50.0, 1.0)
    assert isinstance(res, WaterContentResult)
    assert res.mole_fraction_water > 0
    assert res.water_content_ppmv > 0
    assert res.dew_point_margin_c >= 0

    d = res.to_dict()
    assert "timestamp" in d
    assert d["results"]["water_mole_fraction"] == res.mole_fraction_water

    # Test custom comp and missing name
    comp = SyngasComposition(h2=1.0)  # no name
    res2 = calc.calculate_water_content(50.0, 1.0, gas_composition=comp)
    assert res2.gas_composition == "custom"

    # Test condensation
    res3 = calc.calculate_water_content(90.0, 0.5)
    assert len(res3.warnings) > 0
    assert "condensation will occur" in res3.warnings[0]


def test_generate_water_content_curve(calc: SyngasWaterCalculator) -> None:
    df = calc.generate_water_content_curve(1.0, (20, 50), 3)
    assert len(df) == 3
    assert "water_mole_fraction" in df.columns


def test_quick_functions() -> None:
    res = quick_water_content(50.0, 1.0)
    assert "water_content_ppmv" in res
    assert res["mole_fraction"] > 0

    risk = estimate_condensation_risk(20.0, 1.0)
    assert "condensation_risk" in risk
    assert isinstance(risk["condensation_occurring"], bool)

    from unittest.mock import patch

    with patch(
        "upstream_drift_tools.process_calculators.syngas_water_calculator.SyngasWaterCalculator.calculate_water_content"
    ) as mock_calc:
        content_res = WaterContentResult(
            temperature_c=25.0,
            temperature_k=298.15,
            pressure_bar=1.0,
            pressure_pa=1e5,
            gas_composition="mock",
            vapor_pressure_pa=1000,
            vapor_pressure_bar=0.01,
            saturation_temperature_c=25.0,
            mole_fraction_water=0.01,
            mass_fraction_water=0.01,
            water_content_g_per_m3=1,
            water_content_mg_per_nm3=1,
            water_content_ppmv=10,
            water_content_lb_per_mmscf=1,
            dew_point_c=25.0,
            dew_point_margin_c=-1.0,
            relative_humidity=100.0,
            calculation_method="mock",
        )
        mock_calc.return_value = content_res
        risk1 = estimate_condensation_risk(25.0, 1.0)
        assert risk1["condensation_risk"] == "Critical - Condensation occurring"

        content_res.dew_point_margin_c = 5.0
        risk2 = estimate_condensation_risk(25.0, 1.0, safety_margin_c=10.0)
        assert risk2["condensation_risk"] == "High"

        content_res.dew_point_margin_c = 15.0
        risk3 = estimate_condensation_risk(25.0, 1.0, safety_margin_c=10.0)
        assert risk3["condensation_risk"] == "Medium"
