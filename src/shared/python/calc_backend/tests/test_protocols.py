"""Tests for calc_backend protocols (structural typing) and ODE Pydantic contracts."""

from __future__ import annotations

from typing import Any

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Protocol structural typing tests
# ──────────────────────────────────────────────────────────────────────────────


class TestProtocols:
    """Test that the calc_backend protocols work correctly."""

    def test_calculation_engine_protocol(self) -> Any:
        from calc_backend.protocols import CalculationEngine
        from pydantic import BaseModel

        class DummyRequest(BaseModel):
            x: float

        class DummyResponse(BaseModel):
            result: float

        class DummyEngine:
            def calculate(self, request: DummyRequest) -> DummyResponse:
                return DummyResponse(result=request.x * 2)

        engine = DummyEngine()
        assert isinstance(engine, CalculationEngine)
        result = engine.calculate(DummyRequest(x=5.0))
        assert result.result == 10.0

    def test_validation_mixin_protocol(self) -> Any:
        from calc_backend.protocols import ValidationMixin

        class DummyValidator:
            def validate_inputs(self, request) -> None:
                if request.get("x", 0) < 0:
                    raise ValueError("x must be positive")

        v = DummyValidator()
        assert isinstance(v, ValidationMixin)
        v.validate_inputs({"x": 1.0})
        with pytest.raises(ValueError):
            v.validate_inputs({"x": -1.0})

    def test_expression_evaluator_protocol(self) -> Any:
        from calc_backend.protocols import ExpressionEvaluator

        class DummyEval:
            def evaluate(self, expression: str, namespace: dict) -> float:
                return eval(expression, {}, namespace)  # noqa: S307  # nosec B307

            def validate(self, expression: str) -> bool:
                return True

        e = DummyEval()
        assert isinstance(e, ExpressionEvaluator)
        assert e.evaluate("x + 1", {"x": 5}) == 6
        assert e.validate("x + 1") is True


# ──────────────────────────────────────────────────────────────────────────────
# ODE contracts tests
# ──────────────────────────────────────────────────────────────────────────────


class TestODEContracts:
    """Test Pydantic contracts for ODE solver."""

    def test_ode_request_defaults(self) -> Any:
        from calc_backend.contracts.ode_solver import ODESolverRequest

        req = ODESolverRequest(
            derivatives={"y": "-y"},
            initial_conditions={"y": 1.0},
        )
        assert req.t_start == 0.0
        assert req.t_end == 20.0
        assert req.num_points == 100

    def test_ode_response_defaults(self) -> Any:
        from calc_backend.contracts.ode_solver import (
            ODESolverResponse,
            ODEVariableSummary,
        )

        resp = ODESolverResponse(
            times=[0.0, 1.0],
            solutions={"y": [1.0, 0.9]},
            variable_summaries=[
                ODEVariableSummary(
                    name="y",
                    initial_value=1.0,
                    final_value=0.9,
                    min_value=0.9,
                    max_value=1.0,
                )
            ],
        )
        assert resp.success is True
        assert "computed" in resp.message

    def test_thermal_profile_request_defaults(self) -> Any:
        from calc_backend.contracts.thermal_profile import ThermalProfileRequest

        req = ThermalProfileRequest(
            thermal_mass_j_per_k=1000.0, heat_loss_coeff_w_per_k=5.0
        )
        assert req.power_w == 5000.0
        assert req.power_profile == "constant"
        assert req.num_points == 100

    def test_flow_rate_contracts(self) -> Any:
        from calc_backend.contracts.flow_rate import FlowRateConvertRequest

        req = FlowRateConvertRequest(value=10.0, from_unit="kg/s", to_unit="lb/s")
        assert req.category == "mass"

    def test_pressure_drop_contracts(self) -> Any:
        from calc_backend.contracts.pressure_drop import PressureDropRequest

        req = PressureDropRequest(
            pipe_diameter_m=0.1,
            pipe_length_m=100.0,
            flow_rate_kg_s=1.0,
            temperature_k=300.0,
            pressure_pa=101325.0,
            molecular_weight_kg_mol=0.029,
        )
        assert req.roughness_m == pytest.approx(0.000045, rel=1e-5)

    def test_wgs_contracts(self) -> Any:
        from calc_backend.contracts.wgs_reactor import WGSReactorRequest

        req = WGSReactorRequest(
            inlet_composition={"CO": 20.0, "H2": 40.0, "CO2": 10.0, "H2O": 30.0},
            temperature_k=700.0,
            pressure_bar=10.0,
        )
        assert req.steam_ratio == 2.0
        assert req.feed_rate_kmol_hr == 0.0

    def test_syngas_water_contracts(self) -> Any:
        from calc_backend.contracts.syngas_water import SyngasWaterRequest

        req = SyngasWaterRequest(temperature_c=50.0, pressure_bar=10.0)
        assert req.composition_key == "typical_syngas"
        assert req.method == "auto"

    def test_acid_gas_contracts(self) -> Any:
        from calc_backend.contracts.acid_gas_dewpoint import AcidGasDewpointRequest

        req = AcidGasDewpointRequest(temperature_c=150.0, pressure_bar=1.0)
        assert req.method == "antoine"
        assert req.h2o_fraction == 0.0

    def test_rotation_contract_validate_twist(self) -> Any:
        """Test that the model validator fires on bad twist_frame_conversion."""
        import pydantic
        from calc_backend.contracts.rotation_converter import (
            ReferenceFrameConversionRequest,
        )

        with pytest.raises(pydantic.ValidationError):
            # Missing twist/transform → validator raises ValueError
            ReferenceFrameConversionRequest(operation="twist_frame_conversion")

    def test_rotation_contract_validate_homogeneous(self) -> Any:
        import pydantic
        from calc_backend.contracts.rotation_converter import (
            ReferenceFrameConversionRequest,
        )

        with pytest.raises(pydantic.ValidationError):
            # Missing rotation_matrix/translation
            ReferenceFrameConversionRequest(operation="homogeneous_transform")

    def test_rotation_contract_validate_so3(self) -> Any:
        import pydantic
        from calc_backend.contracts.rotation_converter import (
            ReferenceFrameConversionRequest,
        )

        with pytest.raises(pydantic.ValidationError):
            # Need exactly one of so3_vector, so3_matrix, rotation_matrix
            ReferenceFrameConversionRequest(operation="so3_so3_maps")

    def test_rotation_contract_so3_valid(self) -> Any:
        from calc_backend.contracts.rotation_converter import (
            ReferenceFrameConversionRequest,
        )

        req = ReferenceFrameConversionRequest(
            operation="so3_so3_maps",
            so3_vector=[0.1, 0.2, 0.3],
        )
        assert req.operation == "so3_so3_maps"

    def test_scrubber_contracts(self) -> Any:
        from calc_backend.contracts.scrubber import ScrubberRequest

        req = ScrubberRequest(
            gas_flow_kg_hr=10000.0,
            gas_temperature_k=400.0,
            gas_pressure_pa=101325.0,
            gas_molecular_weight=28.0,
            liquid_flow_kg_hr=5000.0,
        )
        assert req.packing_type == "Metal Pall Rings"
        assert req.percent_of_flood == pytest.approx(70.0)

    def test_flare_contracts(self) -> Any:
        from calc_backend.contracts.flare import FlareRequest

        req = FlareRequest(
            total_flow_kg_hr=10000.0,
            gas_composition={"H2": 50.0, "CO": 50.0},
            temperature_k=400.0,
            pressure_bar=1.5,
        )
        assert req.total_flow_kg_hr == 10000.0

    def test_baghouse_contracts(self) -> Any:
        from calc_backend.contracts.baghouse import BaghouseRequest

        req = BaghouseRequest(
            gas_flow_kg_s=5.0,
            inlet_temp_k=450.0,
            pressure_pa=101325.0,
            composition={"N2": 0.7},
            solid_carbon_in_kg_hr=50.0,
            ash_in_kg_hr=30.0,
            carbon_removal_efficiency=0.95,
            ash_removal_efficiency=0.99,
            heat_loss_w=0.0,
            drum_volume_m3=0.5,
            solid_density_kg_m3=1500.0,
            bag_area_ft2=1000.0,
        )
        assert req.gas_flow_kg_s == 5.0
