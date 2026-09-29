"""Tests for src.shared.python.calc_backend.contracts.scrubber (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from src.shared.python.calc_backend.contracts.scrubber import (
    ScrubberRequest,
    ScrubberResponse,
)


def _valid_request(**kwargs) -> ScrubberRequest:
    defaults = {
        "gas_flow_kg_hr": 1000.0,
        "gas_temperature_k": 400.0,
        "gas_pressure_pa": 101325.0,
        "gas_molecular_weight": 28.0,
        "liquid_flow_kg_hr": 500.0,
    }
    defaults.update(kwargs)
    return ScrubberRequest(**defaults)


class TestScrubberRequest:
    def test_contracts_scrubber_valid_construction(self) -> None:
        req = _valid_request()
        assert isinstance(req, ScrubberRequest)

    def test_gas_flow_stored(self) -> None:
        req = _valid_request(gas_flow_kg_hr=2000.0)
        assert req.gas_flow_kg_hr == pytest.approx(2000.0)

    def test_default_packing_type(self) -> None:
        req = _valid_request()
        assert "Pall" in req.packing_type or len(req.packing_type) > 0

    def test_default_percent_of_flood(self) -> None:
        req = _valid_request()
        assert req.percent_of_flood == pytest.approx(70.0)

    def test_default_caustic_concentration(self) -> None:
        req = _valid_request()
        assert req.caustic_concentration_pct == pytest.approx(10.0)

    def test_default_acid_gas_removed_empty(self) -> None:
        req = _valid_request()
        assert req.acid_gas_removed_kg_hr == {}

    def test_zero_gas_flow_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_request(gas_flow_kg_hr=0.0)

    def test_zero_liquid_flow_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_request(liquid_flow_kg_hr=0.0)

    def test_contracts_scrubber_zero_temperature_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_request(gas_temperature_k=0.0)

    def test_percent_over_100_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_request(percent_of_flood=110.0)

    def test_caustic_over_50_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_request(caustic_concentration_pct=55.0)

    def test_acid_gas_removed_custom(self) -> None:
        req = _valid_request(acid_gas_removed_kg_hr={"HCl": 5.0, "SO2": 3.0})
        assert req.acid_gas_removed_kg_hr["HCl"] == pytest.approx(5.0)


class TestScrubberResponse:
    def _make_response(self) -> ScrubberResponse:
        return ScrubberResponse(
            gas_density_kg_m3=0.8,
            flooding_velocity_m_s=3.0,
            design_velocity_m_s=2.1,
            column_diameter_m=1.2,
            column_diameter_ft=3.94,
            cross_section_m2=1.13,
            caustic_requirement={"NaOH": 10.0, "H2O": 90.0},
        )

    def test_contracts_scrubber_construction(self) -> None:
        resp = self._make_response()
        assert isinstance(resp, ScrubberResponse)

    def test_column_diameter_stored(self) -> None:
        resp = self._make_response()
        assert resp.column_diameter_m == pytest.approx(1.2)

    def test_caustic_requirement_stored(self) -> None:
        resp = self._make_response()
        assert "NaOH" in resp.caustic_requirement
