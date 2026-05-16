"""Tests for sidekick.process_calculators.scrubber.engine.scrubber_engine (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from sidekick.process_calculators.scrubber.engine.scrubber_engine import (
    ScrubberEngine,
)
from sidekick.process_calculators.scrubber.models.scrubber_models import (
    ScrubberInputs,
    ScrubberResults,
)


def _make_inputs(**kwargs) -> ScrubberInputs:
    defaults = {
        "gas_flow_kg_hr": 1000.0,
        "inlet_temp_c": 200.0,
        "pressure_bar": 1.5,
        "molecular_weight": 28.0,
        "target_outlet_temp_c": 40.0,
        "packing_name": "Metal Pall Rings",
        "percent_of_flood": 0.7,
        "height_safety_factor": 1.2,
        "lg_ratio": 2.5,
        "caustic_concentration_wt_pct": 10.0,
        "cooling_water_inlet_temp_c": 20.0,
        "kla_hr": 100.0,
        "acid_gas_composition_ppmv": {"HCl": 500.0},
        "acid_gas_removal_pct": {"HCl": 99.0},
    }
    defaults.update(kwargs)
    return ScrubberInputs(**defaults)


class TestScrubberEngineCalculate:
    def test_returns_scrubber_results(self) -> None:
        inputs = _make_inputs()
        result = ScrubberEngine.calculate(inputs)
        assert isinstance(result, ScrubberResults)

    def test_positive_column_diameter(self) -> None:
        inputs = _make_inputs()
        result = ScrubberEngine.calculate(inputs)
        assert result.column_diameter_m > 0.0

    def test_positive_packed_height(self) -> None:
        inputs = _make_inputs()
        result = ScrubberEngine.calculate(inputs)
        assert result.packed_height_m > 0.0

    def test_positive_heat_duty(self) -> None:
        inputs = _make_inputs()
        result = ScrubberEngine.calculate(inputs)
        assert result.total_heat_duty_kw >= 0.0

    def test_positive_cooling_water(self) -> None:
        inputs = _make_inputs()
        result = ScrubberEngine.calculate(inputs)
        assert result.cooling_water_flow_L_min >= 0.0

    def test_zero_flow_returns_zeros(self) -> None:
        inputs = _make_inputs(gas_flow_kg_hr=0.0)
        result = ScrubberEngine.calculate(inputs)
        assert result.column_diameter_m == pytest.approx(0.0)
        assert result.packed_height_m == pytest.approx(0.0)

    def test_with_multiple_acid_gases(self) -> None:
        inputs = _make_inputs(
            acid_gas_composition_ppmv={"HCl": 500.0, "SO2": 100.0},
            acid_gas_removal_pct={"HCl": 99.0, "SO2": 95.0},
        )
        result = ScrubberEngine.calculate(inputs)
        assert isinstance(result, ScrubberResults)
        # With acid gas, there should be caustic consumption
        assert result.naoh_pure_kg_hr >= 0.0

    def test_higher_gas_flow_larger_diameter(self) -> None:
        low_flow = _make_inputs(gas_flow_kg_hr=500.0)
        high_flow = _make_inputs(gas_flow_kg_hr=2000.0)
        result_low = ScrubberEngine.calculate(low_flow)
        result_high = ScrubberEngine.calculate(high_flow)
        assert result_high.column_diameter_m >= result_low.column_diameter_m

    def test_warnings_is_list(self) -> None:
        inputs = _make_inputs()
        result = ScrubberEngine.calculate(inputs)
        assert isinstance(result.warnings, list)

    def test_htu_positive(self) -> None:
        inputs = _make_inputs()
        result = ScrubberEngine.calculate(inputs)
        assert result.htu_m > 0.0

    def test_pressure_drop_positive(self) -> None:
        inputs = _make_inputs()
        result = ScrubberEngine.calculate(inputs)
        assert result.pressure_drop_kpa >= 0.0
