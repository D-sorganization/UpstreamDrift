"""Tests for src.shared.python.calc_backend.contracts.baghouse (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from src.shared.python.calc_backend.contracts.baghouse import (
    BaghouseRequest,
    BaghouseResponse,
)


def _valid_request(**kwargs) -> BaghouseRequest:
    defaults = {
        "gas_flow_kg_s": 1.0,
        "inlet_temp_k": 500.0,
        "pressure_pa": 101325.0,
        "composition": {"N2": 0.7, "CO2": 0.15, "H2O": 0.1, "CO": 0.05},
    }
    defaults.update(kwargs)
    return BaghouseRequest(**defaults)


class TestBaghouseRequest:
    def test_contracts_baghouse_valid_construction(self) -> None:
        req = _valid_request()
        assert isinstance(req, BaghouseRequest)

    def test_gas_flow_stored(self) -> None:
        req = _valid_request(gas_flow_kg_s=2.5)
        assert req.gas_flow_kg_s == pytest.approx(2.5)

    def test_default_solid_carbon_zero(self) -> None:
        req = _valid_request()
        assert req.solid_carbon_in_kg_hr == pytest.approx(0.0)

    def test_default_ash_zero(self) -> None:
        req = _valid_request()
        assert req.ash_in_kg_hr == pytest.approx(0.0)

    def test_default_carbon_removal_efficiency(self) -> None:
        req = _valid_request()
        assert req.carbon_removal_efficiency == pytest.approx(0.99)

    def test_default_ash_removal_efficiency(self) -> None:
        req = _valid_request()
        assert req.ash_removal_efficiency == pytest.approx(0.999)

    def test_default_drum_volume(self) -> None:
        req = _valid_request()
        assert req.drum_volume_m3 == pytest.approx(2.0)

    def test_zero_gas_flow_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_request(gas_flow_kg_s=0.0)

    def test_contracts_baghouse_zero_temperature_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_request(inlet_temp_k=0.0)

    def test_contracts_baghouse_zero_pressure_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_request(pressure_pa=0.0)

    def test_carbon_efficiency_over_1_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_request(carbon_removal_efficiency=1.5)

    def test_custom_composition(self) -> None:
        comp = {"N2": 0.9, "CO2": 0.1}
        req = _valid_request(composition=comp)
        assert req.composition["N2"] == pytest.approx(0.9)

    def test_heat_loss_default_zero(self) -> None:
        req = _valid_request()
        assert req.heat_loss_w == pytest.approx(0.0)


class TestBaghouseResponse:
    def _make_response(self) -> BaghouseResponse:
        return BaghouseResponse(
            carbon_removed_rate_kg_hr=50.0,
            ash_removed_rate_kg_hr=30.0,
            total_solids_removed_rate_kg_hr=80.0,
            drum_fill_time_hours=25.0,
            drum_fill_time_days=1.04,
            carbon_only_fill_time_hours=40.0,
            ash_only_fill_time_hours=66.7,
            clean_gas_flow_rate_kg_hr=3600.0,
            flow_acfm=2000.0,
            flow_scfm=1800.0,
            air_to_cloth_ratio=1.8,
            outlet_temperature_c=227.0,
            ash_stream_composition={"C": 0.62, "ash": 0.38},
            removal_efficiency={"carbon": 0.99, "ash": 0.999},
        )

    def test_contracts_baghouse_construction(self) -> None:
        resp = self._make_response()
        assert isinstance(resp, BaghouseResponse)

    def test_drum_fill_time_stored(self) -> None:
        resp = self._make_response()
        assert resp.drum_fill_time_hours == pytest.approx(25.0)

    def test_removal_efficiency_stored(self) -> None:
        resp = self._make_response()
        assert resp.removal_efficiency["carbon"] == pytest.approx(0.99)

    def test_outlet_temperature_stored(self) -> None:
        resp = self._make_response()
        assert resp.outlet_temperature_c == pytest.approx(227.0)
