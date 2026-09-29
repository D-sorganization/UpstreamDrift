"""Tests for src.shared.python.calc_backend.contracts.pressure_drop (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from src.shared.python.calc_backend.contracts.pressure_drop import (
    PressureDropRequest,
    PressureDropResponse,
)


def _valid_request(**kwargs) -> PressureDropRequest:
    defaults = {
        "pipe_diameter_m": 0.1,
        "pipe_length_m": 10.0,
        "flow_rate_kg_s": 0.5,
        "temperature_k": 300.0,
        "pressure_pa": 101325.0,
        "molecular_weight_kg_mol": 0.029,
    }
    defaults.update(kwargs)
    return PressureDropRequest(**defaults)


class TestPressureDropRequest:
    def test_contracts_pressure_drop_valid_construction(self) -> None:
        req = _valid_request()
        assert isinstance(req, PressureDropRequest)

    def test_diameter_stored(self) -> None:
        req = _valid_request(pipe_diameter_m=0.05)
        assert req.pipe_diameter_m == pytest.approx(0.05)

    def test_default_roughness(self) -> None:
        req = _valid_request()
        assert req.roughness_m == pytest.approx(0.000045)

    def test_zero_diameter_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_request(pipe_diameter_m=0.0)

    def test_negative_diameter_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_request(pipe_diameter_m=-0.1)

    def test_zero_length_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_request(pipe_length_m=0.0)

    def test_zero_flow_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_request(flow_rate_kg_s=0.0)

    def test_contracts_pressure_drop_zero_temperature_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_request(temperature_k=0.0)

    def test_contracts_pressure_drop_zero_pressure_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_request(pressure_pa=0.0)

    def test_zero_molecular_weight_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_request(molecular_weight_kg_mol=0.0)

    def test_roughness_zero_allowed(self) -> None:
        req = _valid_request(roughness_m=0.0)
        assert req.roughness_m == 0.0


class TestPressureDropResponse:
    def _make_response(self) -> PressureDropResponse:
        return PressureDropResponse(
            pressure_drop_pa=1500.0,
            reynolds_number=50000.0,
            friction_factor=0.02,
            velocity_m_s=10.0,
            flow_regime="Turbulent",
            density_kg_m3=1.2,
            viscosity_pa_s=1.8e-5,
        )

    def test_contracts_pressure_drop_construction(self) -> None:
        resp = self._make_response()
        assert isinstance(resp, PressureDropResponse)

    def test_flow_regime_stored(self) -> None:
        resp = self._make_response()
        assert resp.flow_regime == "Turbulent"

    def test_pressure_drop_stored(self) -> None:
        resp = self._make_response()
        assert resp.pressure_drop_pa == pytest.approx(1500.0)

    def test_reynolds_number_stored(self) -> None:
        resp = self._make_response()
        assert resp.reynolds_number == pytest.approx(50000.0)
