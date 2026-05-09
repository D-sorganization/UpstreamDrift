"""Tests for src.shared.python.upstream_drift_tools.process_calculators.flare_calculator (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from src.shared.python.upstream_drift_tools.process_calculators.flare_calculator import (
    GAS_PROPERTIES,
    FlareCalculator,
    FlareDesign,
)

# ---------------------------------------------------------------------------
# Constants / GAS_PROPERTIES
# ---------------------------------------------------------------------------


class TestGasProperties:
    def test_h2_present(self) -> None:
        assert "H2" in GAS_PROPERTIES

    def test_ch4_heating_value_positive(self) -> None:
        assert GAS_PROPERTIES["CH4"]["hv"] > 0

    def test_n2_heating_value_zero(self) -> None:
        assert GAS_PROPERTIES["N2"]["hv"] == 0

    def test_all_entries_have_required_keys(self) -> None:
        for gas, props in GAS_PROPERTIES.items():
            assert "mw" in props, f"{gas} missing mw"
            assert "hv" in props, f"{gas} missing hv"
            assert "cp" in props, f"{gas} missing cp"


# ---------------------------------------------------------------------------
# FlareCalculator.calculate_flare_size
# ---------------------------------------------------------------------------

_SYNGAS = {"H2": 50.0, "CO": 30.0, "CO2": 10.0, "N2": 10.0}


class TestCalculateFlareSize:
    _CALC = FlareCalculator()

    def test_returns_flare_design(self) -> None:
        result = self._CALC.calculate_flare_size(1000.0, _SYNGAS, 400.0, 1.0)
        assert isinstance(result, FlareDesign)

    def test_height_at_least_minimum(self) -> None:
        result = self._CALC.calculate_flare_size(1000.0, _SYNGAS, 400.0, 1.0)
        assert result.height >= 10.0

    def test_diameter_non_negative(self) -> None:
        result = self._CALC.calculate_flare_size(1000.0, _SYNGAS, 400.0, 1.0)
        assert result.diameter >= 0.0

    def test_larger_flow_larger_heat_release(self) -> None:
        low = self._CALC.calculate_flare_size(500.0, _SYNGAS, 400.0, 1.0)
        high = self._CALC.calculate_flare_size(2000.0, _SYNGAS, 400.0, 1.0)
        assert high.heat_release > low.heat_release

    def test_larger_flow_larger_diameter(self) -> None:
        low = self._CALC.calculate_flare_size(500.0, _SYNGAS, 400.0, 1.0)
        high = self._CALC.calculate_flare_size(5000.0, _SYNGAS, 400.0, 1.0)
        assert high.diameter > low.diameter

    def test_inert_gas_zero_heat_release(self) -> None:
        result = self._CALC.calculate_flare_size(1000.0, {"N2": 100.0}, 300.0, 1.0)
        assert result.heat_release == 0.0

    def test_zero_flow_raises(self) -> None:
        with pytest.raises(AssertionError):
            self._CALC.calculate_flare_size(0.0, _SYNGAS, 400.0, 1.0)

    def test_negative_flow_raises(self) -> None:
        with pytest.raises(AssertionError):
            self._CALC.calculate_flare_size(-100.0, _SYNGAS, 400.0, 1.0)

    def test_flare_calculator_negative_temperature_raises(self) -> None:
        with pytest.raises(AssertionError):
            self._CALC.calculate_flare_size(1000.0, _SYNGAS, -10.0, 1.0)

    def test_empty_composition_raises(self) -> None:
        with pytest.raises(AssertionError):
            self._CALC.calculate_flare_size(1000.0, {}, 400.0, 1.0)


# ---------------------------------------------------------------------------
# FlareCalculator.calculate_radiation_zones
# ---------------------------------------------------------------------------


class TestCalculateRadiationZones:
    _CALC = FlareCalculator()

    def _make_design(self, heat_release: float = 5000.0) -> FlareDesign:
        return FlareDesign(
            height=20.0,
            diameter=0.5,
            exit_velocity=170.0,
            heat_release=heat_release,
            radiation_intensity=1.6,
        )

    def test_returns_dict_with_zones(self) -> None:
        zones = self._CALC.calculate_radiation_zones(self._make_design())
        assert "lethal" in zones
        assert "damage" in zones
        assert "safe" in zones
        assert "comfort" in zones

    def test_zones_are_non_negative(self) -> None:
        zones = self._CALC.calculate_radiation_zones(self._make_design())
        for zone, dist in zones.items():
            assert dist >= 0.0, f"{zone} distance should be non-negative"

    def test_lethal_closer_than_damage(self) -> None:
        zones = self._CALC.calculate_radiation_zones(self._make_design())
        assert zones["lethal"] < zones["damage"]

    def test_damage_closer_than_safe(self) -> None:
        zones = self._CALC.calculate_radiation_zones(self._make_design())
        assert zones["damage"] < zones["safe"]

    def test_safe_closer_than_comfort(self) -> None:
        zones = self._CALC.calculate_radiation_zones(self._make_design())
        assert zones["safe"] < zones["comfort"]

    def test_larger_heat_release_larger_zones(self) -> None:
        small = self._CALC.calculate_radiation_zones(self._make_design(1000.0))
        large = self._CALC.calculate_radiation_zones(self._make_design(10000.0))
        assert large["safe"] > small["safe"]


# ---------------------------------------------------------------------------
# FlareCalculator.calculate_combustion_efficiency
# ---------------------------------------------------------------------------


class TestCalculateCombustionEfficiency:
    _CALC = FlareCalculator()

    def test_base_efficiency_range(self) -> None:
        eff = self._CALC.calculate_combustion_efficiency({"CH4": 100.0}, 400.0, 1.0)
        assert 0.95 <= eff <= 1.0

    def test_high_h2_boosts_efficiency(self) -> None:
        base = self._CALC.calculate_combustion_efficiency({"N2": 100.0}, 400.0, 1.0)
        h2_rich = self._CALC.calculate_combustion_efficiency({"H2": 100.0}, 400.0, 1.0)
        assert h2_rich >= base

    def test_high_h2s_penalizes_efficiency(self) -> None:
        base = self._CALC.calculate_combustion_efficiency({"N2": 100.0}, 400.0, 1.0)
        h2s_rich = self._CALC.calculate_combustion_efficiency(
            {"H2S": 100.0}, 400.0, 1.0
        )
        assert h2s_rich <= base

    def test_efficiency_within_bounds(self) -> None:
        eff = self._CALC.calculate_combustion_efficiency(_SYNGAS, 350.0, 2.0)
        assert 0.95 <= eff <= 0.999
