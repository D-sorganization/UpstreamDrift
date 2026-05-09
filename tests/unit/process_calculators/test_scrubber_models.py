"""Tests for upstream_drift_tools.process_calculators.scrubber.models (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from src.shared.python.upstream_drift_tools.process_calculators.scrubber.models.scrubber_models import (
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
        "packing_name": "pall_ring_25mm",
        "percent_of_flood": 0.7,
        "height_safety_factor": 1.2,
        "lg_ratio": 2.5,
        "caustic_concentration_wt_pct": 10.0,
        "cooling_water_inlet_temp_c": 20.0,
        "kla_hr": 100.0,
    }
    defaults.update(kwargs)
    return ScrubberInputs(**defaults)


def _make_results(**kwargs) -> ScrubberResults:
    defaults = {
        "column_diameter_m": 1.2,
        "packed_height_m": 5.0,
        "pressure_drop_kpa": 2.5,
        "naoh_pure_kg_hr": 50.0,
        "naoh_solution_L_hr": 500.0,
        "total_heat_duty_kw": 100.0,
        "cooling_water_flow_L_min": 200.0,
        "gas_density_kg_m3": 1.1,
        "flooding_velocity_m_s": 2.5,
        "htu_m": 0.8,
        "max_ntu": 3.0,
    }
    defaults.update(kwargs)
    return ScrubberResults(**defaults)


class TestScrubberInputs:
    def test_scrubber_models_construction(self) -> None:
        inputs = _make_inputs()
        assert inputs.gas_flow_kg_hr == pytest.approx(1000.0)

    def test_frozen_immutable(self) -> None:
        inputs = _make_inputs()
        with pytest.raises((AttributeError, TypeError)):
            inputs.gas_flow_kg_hr = 999.0  # type: ignore[misc]

    def test_default_acid_gas_composition_empty(self) -> None:
        inputs = _make_inputs()
        assert inputs.acid_gas_composition_ppmv == {}

    def test_default_acid_gas_removal_empty(self) -> None:
        inputs = _make_inputs()
        assert inputs.acid_gas_removal_pct == {}

    def test_custom_acid_gas_composition(self) -> None:
        inputs = _make_inputs(acid_gas_composition_ppmv={"H2S": 500.0, "CO2": 1000.0})
        assert inputs.acid_gas_composition_ppmv["H2S"] == pytest.approx(500.0)

    def test_all_fields_stored(self) -> None:
        inputs = _make_inputs(
            packing_name="raschig_ring",
            percent_of_flood=0.8,
        )
        assert inputs.packing_name == "raschig_ring"
        assert inputs.percent_of_flood == pytest.approx(0.8)


class TestScrubberResults:
    def test_scrubber_models_construction(self) -> None:
        results = _make_results()
        assert results.column_diameter_m == pytest.approx(1.2)

    def test_frozen_immutable(self) -> None:
        results = _make_results()
        with pytest.raises((AttributeError, TypeError)):
            results.column_diameter_m = 2.0  # type: ignore[misc]

    def test_scrubber_models_default_warnings_empty(self) -> None:
        results = _make_results()
        assert results.warnings == []

    def test_default_acid_gas_details_empty(self) -> None:
        results = _make_results()
        assert results.acid_gas_details == []

    def test_custom_warnings(self) -> None:
        results = _make_results(warnings=["High pressure drop"])
        assert "High pressure drop" in results.warnings

    def test_all_required_fields(self) -> None:
        results = _make_results()
        assert results.packed_height_m == pytest.approx(5.0)
        assert results.pressure_drop_kpa == pytest.approx(2.5)
        assert results.htu_m == pytest.approx(0.8)
        assert results.max_ntu == pytest.approx(3.0)
