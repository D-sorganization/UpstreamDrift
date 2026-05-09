"""Tests for src.shared.python.upstream_drift_tools.process_calculators.baghouse_calculator (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from src.shared.python.upstream_drift_tools.process_calculators.baghouse_calculator import (
    BaghouseCalculator,
    BaghouseResult,
)

_SYNGAS = {"H2": 0.4, "CO": 0.3, "CO2": 0.1, "N2": 0.2}

_BASE_KWARGS = {
    "gas_flow_kg_s": 1.0,
    "inlet_temp_k": 450.0,
    "pressure_pa": 101325.0,
    "composition": _SYNGAS,
    "solid_carbon_in_kg_hr": 50.0,
    "ash_in_kg_hr": 20.0,
    "carbon_removal_efficiency": 0.99,
    "ash_removal_efficiency": 0.99,
    "heat_loss_w": 1000.0,
    "drum_volume_m3": 1.0,
    "solid_density_kg_m3": 500.0,
    "bag_area_ft2": 1000.0,
}


class TestBaghouseCalculator:
    _CALC = BaghouseCalculator()

    def test_returns_baghouse_result(self) -> None:
        result = self._CALC.calculate(**_BASE_KWARGS)
        assert isinstance(result, BaghouseResult)

    def test_carbon_removed_positive(self) -> None:
        result = self._CALC.calculate(**_BASE_KWARGS)
        assert result.carbon_removed_rate > 0.0

    def test_ash_removed_positive(self) -> None:
        result = self._CALC.calculate(**_BASE_KWARGS)
        assert result.ash_removed_rate > 0.0

    def test_total_solids_is_sum(self) -> None:
        result = self._CALC.calculate(**_BASE_KWARGS)
        expected = result.carbon_removed_rate + result.ash_removed_rate
        assert abs(result.total_solids_removed_rate - expected) < 1e-6

    def test_flow_acfm_positive(self) -> None:
        result = self._CALC.calculate(**_BASE_KWARGS)
        assert result.flow_acfm > 0.0

    def test_air_to_cloth_positive(self) -> None:
        result = self._CALC.calculate(**_BASE_KWARGS)
        assert result.air_to_cloth_ratio > 0.0

    def test_removal_efficiency_in_result(self) -> None:
        result = self._CALC.calculate(**_BASE_KWARGS)
        assert isinstance(result.removal_efficiency, dict)

    def test_zero_gas_flow_raises(self) -> None:
        kwargs = {**_BASE_KWARGS, "gas_flow_kg_s": 0.0}
        with pytest.raises(AssertionError):
            self._CALC.calculate(**kwargs)

    def test_baghouse_calculator_negative_temperature_raises(self) -> None:
        kwargs = {**_BASE_KWARGS, "inlet_temp_k": -100.0}
        with pytest.raises(AssertionError):
            self._CALC.calculate(**kwargs)

    def test_efficiency_above_one_raises(self) -> None:
        kwargs = {**_BASE_KWARGS, "carbon_removal_efficiency": 1.5}
        with pytest.raises(AssertionError):
            self._CALC.calculate(**kwargs)

    def test_zero_efficiency_no_removal(self) -> None:
        kwargs = {
            **_BASE_KWARGS,
            "carbon_removal_efficiency": 0.0,
            "ash_removal_efficiency": 0.0,
        }
        result = self._CALC.calculate(**kwargs)
        assert result.carbon_removed_rate == 0.0
        assert result.ash_removed_rate == 0.0
