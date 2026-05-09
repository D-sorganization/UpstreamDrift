"""Tests for src.shared.python.calc_backend.contracts.acid_gas_dewpoint (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from src.shared.python.calc_backend.contracts.acid_gas_dewpoint import (
    AcidGasDewpointRequest,
    AcidGasDewpointResponse,
    DewpointComponentOut,
)


class TestAcidGasDewpointRequest:
    def test_contracts_acid_gas_dewpoint_valid_construction(self) -> None:
        req = AcidGasDewpointRequest(temperature_c=150.0, pressure_bar=1.0)
        assert isinstance(req, AcidGasDewpointRequest)

    def test_contracts_acid_gas_dewpoint_temperature_stored(self) -> None:
        req = AcidGasDewpointRequest(temperature_c=200.0, pressure_bar=2.0)
        assert req.temperature_c == pytest.approx(200.0)

    def test_default_fractions_zero(self) -> None:
        req = AcidGasDewpointRequest(temperature_c=100.0, pressure_bar=1.0)
        assert req.h2o_fraction == pytest.approx(0.0)
        assert req.hf_fraction == pytest.approx(0.0)
        assert req.hcl_fraction == pytest.approx(0.0)
        assert req.h2s_fraction == pytest.approx(0.0)

    def test_default_method(self) -> None:
        req = AcidGasDewpointRequest(temperature_c=100.0, pressure_bar=1.0)
        assert req.method == "antoine"

    def test_contracts_acid_gas_dewpoint_zero_pressure_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AcidGasDewpointRequest(temperature_c=100.0, pressure_bar=0.0)

    def test_fraction_over_1_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AcidGasDewpointRequest(
                temperature_c=100.0, pressure_bar=1.0, h2o_fraction=1.5
            )

    def test_fraction_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AcidGasDewpointRequest(
                temperature_c=100.0, pressure_bar=1.0, hcl_fraction=-0.01
            )

    def test_custom_fractions(self) -> None:
        req = AcidGasDewpointRequest(
            temperature_c=150.0, pressure_bar=1.0, h2o_fraction=0.1, hcl_fraction=0.005
        )
        assert req.h2o_fraction == pytest.approx(0.1)
        assert req.hcl_fraction == pytest.approx(0.005)

    def test_negative_temperature_allowed(self) -> None:
        # Temperature can be negative (e.g., cold pipe below freezing)
        req = AcidGasDewpointRequest(temperature_c=-10.0, pressure_bar=1.0)
        assert req.temperature_c == pytest.approx(-10.0)


class TestDewpointComponentOut:
    def test_construction_with_dewpoint(self) -> None:
        comp = DewpointComponentOut(
            dewpoint_c=45.0,
            vapor_pressure_pa=9600.0,
            partial_pressure_pa=1000.0,
        )
        assert isinstance(comp, DewpointComponentOut)

    def test_construction_no_dewpoint(self) -> None:
        comp = DewpointComponentOut(
            dewpoint_c=None,
            vapor_pressure_pa=0.5,
            partial_pressure_pa=0.0,
        )
        assert comp.dewpoint_c is None

    def test_vapor_pressure_stored(self) -> None:
        comp = DewpointComponentOut(
            dewpoint_c=60.0, vapor_pressure_pa=20000.0, partial_pressure_pa=2000.0
        )
        assert comp.vapor_pressure_pa == pytest.approx(20000.0)


class TestAcidGasDewpointResponse:
    def _make_response(self) -> AcidGasDewpointResponse:
        return AcidGasDewpointResponse(
            overall_dewpoint_c=45.0,
            limiting_component="H2O",
            dewpoint_margin_c=105.0,
            condensation_risk="Low",
            components={
                "H2O": DewpointComponentOut(
                    dewpoint_c=45.0,
                    vapor_pressure_pa=9600.0,
                    partial_pressure_pa=1000.0,
                )
            },
            calculation_method="antoine",
        )

    def test_contracts_acid_gas_dewpoint_construction(self) -> None:
        resp = self._make_response()
        assert isinstance(resp, AcidGasDewpointResponse)

    def test_limiting_component_stored(self) -> None:
        resp = self._make_response()
        assert resp.limiting_component == "H2O"

    def test_condensation_risk_stored(self) -> None:
        resp = self._make_response()
        assert resp.condensation_risk == "Low"

    def test_contracts_acid_gas_dewpoint_default_warnings_empty(self) -> None:
        resp = self._make_response()
        assert resp.warnings == []

    def test_overall_dewpoint_none_allowed(self) -> None:
        resp = AcidGasDewpointResponse(
            overall_dewpoint_c=None,
            limiting_component="none",
            dewpoint_margin_c=None,
            condensation_risk="None",
            components={},
            calculation_method="antoine",
        )
        assert resp.overall_dewpoint_c is None
