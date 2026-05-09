"""Tests for upstream_drift_tools.process_calculators.baghouse_calculator (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from src.shared.python.upstream_drift_tools.process_calculators.baghouse_calculator import (
    BaghouseCalculator,
    BaghouseResult,
)

_SYNGAS = {"CO": 0.4, "H2": 0.3, "CO2": 0.2, "H2O": 0.05, "N2": 0.05}


def _run(**kwargs) -> BaghouseResult:
    defaults = {
        "gas_flow_kg_s": 1.0,
        "inlet_temp_k": 700.0,
        "pressure_pa": 101325.0,
        "composition": _SYNGAS,
        "solid_carbon_in_kg_hr": 50.0,
        "ash_in_kg_hr": 20.0,
        "carbon_removal_efficiency": 0.99,
        "ash_removal_efficiency": 0.99,
        "heat_loss_w": 5000.0,
        "drum_volume_m3": 0.5,
        "solid_density_kg_m3": 500.0,
        "bag_area_ft2": 1000.0,
    }
    defaults.update(kwargs)
    calc = BaghouseCalculator()
    return calc.calculate(**defaults)


class TestBaghouseCalculator:
    def test_baghouse_calculator_construction(self) -> None:
        calc = BaghouseCalculator()
        assert calc is not None

    def test_returns_baghouse_result(self) -> None:
        result = _run()
        assert isinstance(result, BaghouseResult)

    def test_positive_carbon_removed(self) -> None:
        result = _run()
        assert result.carbon_removed_rate >= 0.0

    def test_positive_ash_removed(self) -> None:
        result = _run()
        assert result.ash_removed_rate >= 0.0

    def test_total_solids_equals_sum(self) -> None:
        result = _run()
        assert result.total_solids_removed_rate == pytest.approx(
            result.carbon_removed_rate + result.ash_removed_rate, rel=1e-6
        )

    def test_positive_drum_fill_time(self) -> None:
        result = _run()
        assert result.drum_fill_time_hours > 0.0

    def test_positive_air_to_cloth_ratio(self) -> None:
        result = _run()
        assert result.air_to_cloth_ratio > 0.0

    def test_removal_efficiency_dict(self) -> None:
        result = _run()
        assert isinstance(result.removal_efficiency, dict)

    def test_ash_stream_composition_dict(self) -> None:
        result = _run()
        assert isinstance(result.ash_stream_composition, dict)

    def test_high_efficiency_removes_most_solids(self) -> None:
        result = _run(carbon_removal_efficiency=0.999, ash_removal_efficiency=0.999)
        # Should remove most of the 50 kg/hr carbon
        assert result.carbon_removed_rate > 40.0

    def test_zero_solids_has_zero_removed(self) -> None:
        result = _run(solid_carbon_in_kg_hr=0.0, ash_in_kg_hr=0.0)
        assert result.carbon_removed_rate == pytest.approx(0.0)
        assert result.ash_removed_rate == pytest.approx(0.0)

    def test_outlet_temperature_present(self) -> None:
        result = _run()
        assert isinstance(result.outlet_temperature_c, float)

    def test_positive_flow_acfm(self) -> None:
        result = _run()
        assert result.flow_acfm > 0.0

    def test_positive_flow_scfm(self) -> None:
        result = _run()
        assert result.flow_scfm > 0.0
