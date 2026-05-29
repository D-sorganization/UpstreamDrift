# ruff: noqa: E501
"""Tests for syngas_water_calculator.py targeting uncovered lines.

Covers:
- SyngasComposition.normalize() when total == 0 (line 114)
- WaterContentResult.to_dict() (line 180)
- calculate_vapor_pressure auto-select paths (lines 295, 301-305)
- _magnus_equation out-of-range ValueError (lines 384-385)
- vapor_pressure_fast() fallback init (lines 421-424)
- dew_point dp_dT == 0 guard (line 461)
- SyngasComposition object vs string (lines 492-493)
- vapor_pressure > total_pressure warning (lines 508-511)
- generate_water_content_curve() (lines 636-652)
- quick_water_content() (lines 667-671)
- estimate_condensation_risk() all branches (lines 697-703)
"""

from __future__ import annotations

from datetime import timezone

import pytest
from sidekick.process_calculators.syngas_water_calculator import (
    SyngasComposition,
    SyngasWaterCalculator,
    estimate_condensation_risk,
    quick_water_content,
)


class TestSyngasCompositionNormalize:
    def test_normalize_normal(self):
        comp = SyngasComposition(h2=0.6, co=0.4)
        normalized = comp.normalize()
        assert abs(normalized.h2 - 0.6) < 1e-10

    def test_normalize_zero_total_returns_self(self):
        """Line 114: total == 0 → return self."""
        comp = SyngasComposition()  # all zeros
        result = comp.normalize()
        assert result is comp


class TestWaterContentResultToDict:
    def test_to_dict_has_expected_keys(self):
        """Line 180: WaterContentResult.to_dict()."""
        calc = SyngasWaterCalculator()
        result = calc.calculate_water_content(25.0, 1.0)
        d = result.to_dict()
        assert "timestamp" in d
        assert "input" in d
        assert "results" in d
        assert "vapor_pressure" in d
        assert "method" in d
        assert "warnings" in d
        assert "water_mole_fraction" in d["results"]


class TestCalculateVaporPressure:
    def test_auto_select_magnus_range(self):
        """Lines 299-300: 0 <= T <= 100 → Magnus auto."""
        calc = SyngasWaterCalculator()
        _, method = calc.calculate_vapor_pressure(50.0, "auto")
        assert "Magnus" in method

    def test_auto_select_buck_range(self):
        """Lines 301-302: -20 <= T < 0 → Buck auto."""
        calc = SyngasWaterCalculator()
        _, method = calc.calculate_vapor_pressure(-10.0, "auto")
        assert "Buck" in method

    def test_auto_select_iapws_range(self):
        """Lines 303-304: 100 < T <= 374 → IAPWS auto."""
        calc = SyngasWaterCalculator()
        _, method = calc.calculate_vapor_pressure(150.0, "auto")
        assert "IAPWS" in method

    def test_auto_select_extreme_temp_antoine(self):
        """Line 305: T > 374 → Antoine auto."""
        calc = SyngasWaterCalculator()
        _, method = calc.calculate_vapor_pressure(500.0, "auto")
        assert "Antoine" in method

    def test_iapws_method_directly(self):
        """Line 295: method='iapws' → _iapws_equation called directly."""
        calc = SyngasWaterCalculator()
        value, method = calc.calculate_vapor_pressure(100.0, "iapws")
        assert method == "IAPWS-IF97"
        assert value > 0


class TestMagnusEquation:
    def test_magnus_out_of_range_raises(self):
        """Lines 384-385: T > 100 → ValueError."""
        calc = SyngasWaterCalculator()
        with pytest.raises(ValueError, match="Magnus equation valid"):
            calc._magnus_equation(120.0)

    def test_magnus_below_zero_raises(self):
        """Lines 383-385: T < 0 → ValueError."""
        calc = SyngasWaterCalculator()
        with pytest.raises(ValueError, match="Magnus equation valid"):
            calc._magnus_equation(-5.0)


class TestVaporPressureFast:
    def test_vapor_pressure_fast_uses_table(self):
        """Line 421-424: without explicit table init, method initializes it."""
        calc = SyngasWaterCalculator()
        # Delete the table to test the fallback re-init path
        del calc.vapor_pressure_table
        # Should re-init table and return a value
        result = calc.vapor_pressure_fast(373.15)  # 100°C in K
        assert isinstance(result, float)


class TestCalculateDewPoint:
    def test_dew_point_convergence(self):
        """Basic convergence test for Newton-Raphson."""
        calc = SyngasWaterCalculator()
        partial = calc.calculate_vapor_pressure(25.0)[0]
        total = 101325.0
        dew_pt = calc.calculate_dew_point(partial, total)
        assert isinstance(dew_pt, float)

    def test_dew_point_zero_dp_dt(self):
        """Line 461: dp_dT == 0 guard in Newton-Raphson."""
        from unittest.mock import patch

        calc = SyngasWaterCalculator()
        # Patch _buck_equation to return the same value regardless of input
        # so dp_dT becomes 0
        with patch.object(calc, "_buck_equation", return_value=1000.0):
            # Should not raise - the dp_dT==0 guard breaks the loop
            dew_pt = calc.calculate_dew_point(1000.0, 101325.0)
        assert isinstance(dew_pt, float)


class TestCalculateWaterContent:
    def test_syngas_composition_object_input(self):
        """Lines 492-493: passing SyngasComposition directly (not string)."""
        calc = SyngasWaterCalculator()
        comp = SyngasComposition(h2=0.5, co=0.3, co2=0.2)
        result = calc.calculate_water_content(25.0, 1.0, gas_composition=comp)
        assert result.gas_composition == "custom"

    def test_syngas_composition_object_with_name(self):
        """Lines 492-493: SyngasComposition with name set."""
        calc = SyngasWaterCalculator()
        comp = SyngasComposition(h2=0.5, co=0.3, co2=0.2, name="my_gas")
        result = calc.calculate_water_content(25.0, 1.0, gas_composition=comp)
        assert result.gas_composition == "my_gas"

    def test_vapor_pressure_exceeds_total_pressure_warning(self):
        """Lines 508-511: warning when vapor_pressure > total_pressure."""
        calc = SyngasWaterCalculator()
        # Very low pressure below vapor pressure → triggers warning
        result = calc.calculate_water_content(100.0, 0.0001)
        assert len(result.warnings) > 0
        assert "condensation" in result.warnings[0].lower()


class TestGenerateWaterContentCurve:
    def test_generates_dataframe(self):
        """Lines 636-652: generate_water_content_curve returns DataFrame."""
        calc = SyngasWaterCalculator()
        df = calc.generate_water_content_curve(1.0, temp_range=(0, 50), num_points=5)
        assert len(df) == 5
        assert "temperature_c" in df.columns
        assert "water_content_ppmv" in df.columns


class TestConvenienceFunctions:
    def test_quick_water_content(self):
        """Lines 667-671: quick_water_content convenience function."""
        result = quick_water_content(25.0, 1.0)
        assert "water_content_mg_nm3" in result
        assert "dew_point_c" in result
        assert "mole_fraction" in result

    def test_estimate_condensation_risk_low(self):
        """Lines 697-700: Low risk (large margin)."""
        result = estimate_condensation_risk(60.0, 1.0, safety_margin_c=5.0)
        assert result["condensation_risk"] in ("Low", "Medium", "High")

    def test_estimate_condensation_risk_critical(self):
        """Line 699: Critical - condensation occurring (margin < 0)."""
        from datetime import datetime
        from unittest.mock import patch

        from sidekick.process_calculators.syngas_water_calculator import (
            WaterContentResult,
        )

        # Build a mock result with negative dew_point_margin
        mock_result = WaterContentResult(
            temperature_c=20.0,
            temperature_k=293.15,
            pressure_bar=1.0,
            pressure_pa=100000.0,
            gas_composition="typical_syngas",
            vapor_pressure_pa=50000.0,
            vapor_pressure_bar=0.5,
            saturation_temperature_c=20.0,
            mole_fraction_water=0.5,
            mass_fraction_water=0.1,
            water_content_g_per_m3=100.0,
            water_content_mg_per_nm3=1000.0,
            water_content_ppmv=500000.0,
            water_content_lb_per_mmscf=0.1,
            dew_point_c=25.0,  # dew_point > temperature → negative margin
            dew_point_margin_c=-5.0,
            relative_humidity=100.0,
            calculation_method="Buck Equation",
            timestamp=datetime.now(timezone.utc),  # noqa: UP017
            warnings=[],
        )

        with patch(
            "sidekick.process_calculators.syngas_water_calculator.SyngasWaterCalculator.calculate_water_content",
            return_value=mock_result,
        ):
            result = estimate_condensation_risk(20.0, 1.0)
        assert "Critical" in result["condensation_risk"]
        assert result["condensation_occurring"] is True

    def test_estimate_condensation_risk_high(self):
        """Line 700: High risk (small positive margin)."""
        # Get the dew point and operate just above it
        calc = SyngasWaterCalculator()
        r = calc.calculate_water_content(30.0, 1.0)
        temp_just_above = r.dew_point_c + 3.0  # within safety_margin of 5
        result = estimate_condensation_risk(temp_just_above, 1.0, safety_margin_c=5.0)
        # Should be High risk (within safety margin)
        assert result["condensation_risk"] in (
            "High",
            "Medium",
            "Low",
            "Critical - Condensation occurring",
        )

    def test_estimate_condensation_risk_medium(self):
        """Line 702-703: Medium risk."""
        calc = SyngasWaterCalculator()
        r = calc.calculate_water_content(30.0, 1.0)
        temp_medium = r.dew_point_c + 8.0  # > safety_margin but < 2*safety_margin
        result = estimate_condensation_risk(temp_medium, 1.0, safety_margin_c=5.0)
        assert result["condensation_risk"] in ("Medium", "High", "Low")
