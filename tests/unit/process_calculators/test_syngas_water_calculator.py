"""Tests for src.shared.python.upstream_drift_tools.process_calculators.syngas_water_calculator
and water_vapor_pressure_calculator (Issues #1949, #1744).
"""

from __future__ import annotations

import pytest
from src.shared.python.upstream_drift_tools.process_calculators.syngas_water_calculator import (
    SYNGAS_PRESETS,
    SyngasComposition,
    SyngasWaterCalculator,
)
from src.shared.python.upstream_drift_tools.process_calculators.water_vapor_pressure_calculator import (
    WaterVaporPressureCalculator,
)


class TestSyngasComposition:
    def test_default_construction_zero(self) -> None:
        c = SyngasComposition()
        assert c.h2 == pytest.approx(0.0)
        assert c.total == pytest.approx(0.0)

    def test_total_sums_all_components(self) -> None:
        c = SyngasComposition(h2=0.3, co=0.3, co2=0.15, n2=0.25)
        assert c.total == pytest.approx(1.0, abs=1e-6)

    def test_normalize_returns_composition(self) -> None:
        c = SyngasComposition(h2=30.0, co=30.0, co2=15.0, n2=25.0)
        normalized = c.normalize()
        assert isinstance(normalized, SyngasComposition)

    def test_syngas_water_calculator_normalize_sums_to_one(self) -> None:
        c = SyngasComposition(h2=30.0, co=30.0, co2=15.0, n2=25.0)
        n = c.normalize()
        assert n.total == pytest.approx(1.0, abs=1e-6)

    def test_normalize_zero_total_returns_self(self) -> None:
        c = SyngasComposition()
        result = c.normalize()
        assert result is c

    def test_to_dict_has_expected_keys(self) -> None:
        c = SyngasComposition(h2=0.3, co=0.3)
        d = c.to_dict()
        for key in ["H2", "CO", "CO2", "CH4", "N2", "AR", "H2O", "Other"]:
            assert key in d

    def test_to_dict_values_match(self) -> None:
        c = SyngasComposition(h2=0.30, co=0.25)
        d = c.to_dict()
        assert d["H2"] == pytest.approx(0.30)
        assert d["CO"] == pytest.approx(0.25)


class TestSyngasPresets:
    def test_typical_syngas_exists(self) -> None:
        assert "typical_syngas" in SYNGAS_PRESETS

    def test_biomass_syngas_exists(self) -> None:
        assert "biomass_syngas" in SYNGAS_PRESETS

    def test_coal_syngas_exists(self) -> None:
        assert "coal_syngas" in SYNGAS_PRESETS

    def test_typical_syngas_sums_to_one(self) -> None:
        comp = SYNGAS_PRESETS["typical_syngas"]
        assert comp.total == pytest.approx(1.0, abs=1e-6)

    def test_coal_syngas_has_h2_and_co(self) -> None:
        comp = SYNGAS_PRESETS["coal_syngas"]
        assert comp.h2 > 0.0
        assert comp.co > 0.0


class TestSyngasWaterCalculatorVaporPressure:
    def setup_method(self) -> None:
        self.calc = SyngasWaterCalculator()

    def test_syngas_water_calculator_returns_tuple(self) -> None:
        result = self.calc.calculate_vapor_pressure(25.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_pressure_is_positive(self) -> None:
        pressure, _ = self.calc.calculate_vapor_pressure(25.0)
        assert pressure > 0.0

    def test_pressure_increases_with_temperature(self) -> None:
        p20, _ = self.calc.calculate_vapor_pressure(20.0)
        p50, _ = self.calc.calculate_vapor_pressure(50.0)
        assert p50 > p20

    def test_buck_method(self) -> None:
        pressure, method = self.calc.calculate_vapor_pressure(25.0, method="buck")
        assert pressure > 0.0
        assert "Buck" in method

    def test_antoine_method(self) -> None:
        pressure, method = self.calc.calculate_vapor_pressure(25.0, method="antoine")
        assert pressure > 0.0
        assert "Antoine" in method

    def test_magnus_method(self) -> None:
        pressure, method = self.calc.calculate_vapor_pressure(25.0, method="magnus")
        assert pressure > 0.0
        assert "Magnus" in method

    def test_iapws_method(self) -> None:
        pressure, method = self.calc.calculate_vapor_pressure(100.0, method="iapws")
        assert pressure > 0.0

    def test_at_100c_high_pressure(self) -> None:
        # At 100°C water vapor pressure is physically large
        pressure, _ = self.calc.calculate_vapor_pressure(100.0, method="buck")
        assert pressure > 50000  # Must be well above 0 Pa at boiling point

    def test_auto_method_returns_positive(self) -> None:
        pressure, _ = self.calc.calculate_vapor_pressure(25.0, method="auto")
        assert pressure > 0.0


class TestWaterVaporPressureCalculator:
    def setup_method(self) -> None:
        self.calc = WaterVaporPressureCalculator()

    def test_syngas_water_calculator_returns_float(self) -> None:
        result = self.calc.calculate_vapor_pressure(25.0)
        assert isinstance(result, float)

    def test_positive_at_room_temp(self) -> None:
        result = self.calc.calculate_vapor_pressure(25.0)
        assert result > 0.0

    def test_syngas_water_calculator_increases_with_temperature(self) -> None:
        p_low = self.calc.calculate_vapor_pressure(10.0)
        p_high = self.calc.calculate_vapor_pressure(60.0)
        assert p_high > p_low

    def test_auto_method_works(self) -> None:
        result = self.calc.calculate_vapor_pressure(30.0, method="auto")
        assert result > 0.0
