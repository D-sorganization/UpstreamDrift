"""Tests for sidekick.process_calculators.flare_calculator (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from sidekick.process_calculators.flare_calculator import (
    GAS_PROPERTIES,
    FlareCalculator,
    FlareDesign,
)

_SYNGAS = {"H2": 30.0, "CO": 40.0, "CO2": 20.0, "N2": 10.0}


class TestGasProperties:
    def test_has_hydrogen(self) -> None:
        assert "H2" in GAS_PROPERTIES

    def test_has_carbon_monoxide(self) -> None:
        assert "CO" in GAS_PROPERTIES

    def test_flare_calculator_has_required_fields(self) -> None:
        for gas, props in GAS_PROPERTIES.items():
            assert "mw" in props, f"{gas} missing 'mw'"
            assert "hv" in props, f"{gas} missing 'hv'"
            assert "cp" in props, f"{gas} missing 'cp'"

    def test_positive_molecular_weights(self) -> None:
        for gas, props in GAS_PROPERTIES.items():
            assert props["mw"] > 0, f"{gas} MW should be positive"


class TestFlareDesign:
    def test_flare_calculator_construction(self) -> None:
        design = FlareDesign(
            height=10.0,
            diameter=0.5,
            exit_velocity=50.0,
            heat_release=1000.0,
            radiation_intensity=1.6,
        )
        assert design.height == pytest.approx(10.0)
        assert design.diameter == pytest.approx(0.5)


class TestFlareCalculator:
    def test_flare_calculator_construction(self) -> None:
        calc = FlareCalculator()
        assert calc is not None

    def test_returns_flare_design(self) -> None:
        calc = FlareCalculator()
        result = calc.calculate_flare_size(
            total_flow=1000.0,
            gas_composition=_SYNGAS,
            temperature=700.0,
            pressure=1.5,
        )
        assert isinstance(result, FlareDesign)

    def test_positive_height(self) -> None:
        calc = FlareCalculator()
        result = calc.calculate_flare_size(1000.0, _SYNGAS, 700.0, 1.5)
        assert result.height > 0.0

    def test_positive_diameter(self) -> None:
        calc = FlareCalculator()
        result = calc.calculate_flare_size(1000.0, _SYNGAS, 700.0, 1.5)
        assert result.diameter > 0.0

    def test_positive_heat_release(self) -> None:
        calc = FlareCalculator()
        result = calc.calculate_flare_size(1000.0, _SYNGAS, 700.0, 1.5)
        assert result.heat_release > 0.0

    def test_positive_radiation_intensity(self) -> None:
        calc = FlareCalculator()
        result = calc.calculate_flare_size(1000.0, _SYNGAS, 700.0, 1.5)
        assert result.radiation_intensity >= 0.0

    def test_higher_flow_larger_diameter(self) -> None:
        calc = FlareCalculator()
        low = calc.calculate_flare_size(500.0, _SYNGAS, 700.0, 1.5)
        high = calc.calculate_flare_size(2000.0, _SYNGAS, 700.0, 1.5)
        assert high.diameter >= low.diameter

    def test_higher_flow_more_heat(self) -> None:
        calc = FlareCalculator()
        low = calc.calculate_flare_size(500.0, _SYNGAS, 700.0, 1.5)
        high = calc.calculate_flare_size(2000.0, _SYNGAS, 700.0, 1.5)
        assert high.heat_release > low.heat_release

    def test_zero_flow_raises(self) -> None:
        calc = FlareCalculator()
        with pytest.raises((AssertionError, ValueError, ZeroDivisionError)):
            calc.calculate_flare_size(0.0, _SYNGAS, 700.0, 1.5)

    def test_inert_only_gas_has_zero_heat(self) -> None:
        calc = FlareCalculator()
        inert = {"N2": 50.0, "CO2": 50.0}
        result = calc.calculate_flare_size(1000.0, inert, 700.0, 1.5)
        assert result.heat_release == pytest.approx(0.0)
