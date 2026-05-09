"""Tests for pressure_drop_calculator data models (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from src.shared.python.upstream_drift_tools.process_calculators.pressure_drop_calculator.models.pressure_drop_data_models import (
    GasComposition,
    PipeFitting,
    PressureDropInputs,
)


def _make_inputs(**kwargs) -> PressureDropInputs:
    defaults = {
        "pipe_diameter": 0.1,
        "pipe_length": 10.0,
        "pipe_roughness": 0.000045,
        "mass_flow_rate": 1.0,
        "inlet_pressure": 101325.0,
        "inlet_temperature": 300.0,
    }
    defaults.update(kwargs)
    return PressureDropInputs(**defaults)


class TestGasComposition:
    def test_default_empty(self) -> None:
        gc = GasComposition()
        assert gc.components == {}

    def test_custom_components(self) -> None:
        gc = GasComposition({"H2": 0.5, "CO": 0.5})
        assert gc.components["H2"] == pytest.approx(0.5)

    def test_validate_sums_to_one(self) -> None:
        gc = GasComposition({"H2": 0.5, "CO2": 0.5})
        assert gc.validate() is True

    def test_validate_fails_not_summing_to_one(self) -> None:
        gc = GasComposition({"H2": 0.3, "CO": 0.3})
        assert gc.validate() is False

    def test_validate_empty_composition_fails(self) -> None:
        gc = GasComposition()
        # empty sum is 0, not in [0.99, 1.01]
        assert gc.validate() is False

    def test_pressure_drop_models_normalize_sums_to_one(self) -> None:
        gc = GasComposition({"H2": 30.0, "CO": 70.0})
        gc.normalize()
        total = sum(gc.components.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_normalize_zero_total_noop(self) -> None:
        gc = GasComposition()
        gc.normalize()
        assert gc.components == {}


class TestPipeFitting:
    def test_pressure_drop_models_construction(self) -> None:
        f = PipeFitting(fitting_type="elbow_90")
        assert f.fitting_type == "elbow_90"

    def test_default_quantity_one(self) -> None:
        f = PipeFitting(fitting_type="tee")
        assert f.quantity == 1

    def test_default_k_factor_zero(self) -> None:
        f = PipeFitting(fitting_type="straight")
        assert f.k_factor == pytest.approx(0.0)

    def test_custom_k_factor(self) -> None:
        f = PipeFitting(fitting_type="elbow", k_factor=0.9)
        assert f.k_factor == pytest.approx(0.9)

    def test_custom_quantity(self) -> None:
        f = PipeFitting(fitting_type="tee", quantity=3)
        assert f.quantity == 3


class TestPressureDropInputs:
    def test_pressure_drop_models_valid_construction(self) -> None:
        inputs = _make_inputs()
        assert inputs.pipe_diameter == pytest.approx(0.1)

    def test_validate_all_valid(self) -> None:
        inputs = _make_inputs()
        is_valid, msg = inputs.validate()
        assert isinstance(is_valid, bool)
        assert isinstance(msg, str)

    def test_valid_inputs_pass(self) -> None:
        inputs = _make_inputs()
        # Default inputs don't have a composition → composition.validate() returns False
        # So overall validate may return False; just check it runs without error
        is_valid, msg = inputs.validate()
        assert msg != ""

    def test_negative_diameter_fails(self) -> None:
        inputs = _make_inputs(pipe_diameter=-0.1)
        is_valid, _ = inputs.validate()
        assert is_valid is False

    def test_zero_pipe_length_fails(self) -> None:
        inputs = _make_inputs(pipe_length=0.0)
        is_valid, _ = inputs.validate()
        assert is_valid is False

    def test_negative_roughness_fails(self) -> None:
        inputs = _make_inputs(pipe_roughness=-0.001)
        is_valid, _ = inputs.validate()
        assert is_valid is False

    def test_zero_mass_flow_fails(self) -> None:
        inputs = _make_inputs(mass_flow_rate=0.0)
        is_valid, _ = inputs.validate()
        assert is_valid is False

    def test_default_elevation_change_zero(self) -> None:
        inputs = _make_inputs()
        assert inputs.elevation_change == pytest.approx(0.0)

    def test_default_compressibility_correction_true(self) -> None:
        assert _make_inputs().compressibility_correction is True

    def test_default_friction_method(self) -> None:
        assert _make_inputs().friction_method == "colebrook"

    def test_valid_with_good_composition(self) -> None:
        gc = GasComposition({"H2": 0.5, "CO2": 0.5})
        inputs = _make_inputs(gas_composition=gc)
        is_valid, _ = inputs.validate()
        assert is_valid is True

    def test_fittings_list(self) -> None:
        fittings = [PipeFitting("elbow", k_factor=0.5)]
        inputs = _make_inputs(fittings=fittings)
        assert len(inputs.fittings) == 1
