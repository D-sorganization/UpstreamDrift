"""Tests for src.shared.python.calc_backend.contracts.flare (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from src.shared.python.calc_backend.contracts.flare import (
    FlareDesignOut,
    FlareRequest,
    FlareResponse,
    RadiationZonesOut,
)


def _valid_request(**kwargs) -> FlareRequest:
    defaults = {
        "total_flow_kg_hr": 1000.0,
        "gas_composition": {"H2": 10.0, "CO": 30.0, "CH4": 20.0, "N2": 40.0},
        "temperature_k": 400.0,
        "pressure_bar": 1.2,
    }
    defaults.update(kwargs)
    return FlareRequest(**defaults)


class TestFlareRequest:
    def test_contracts_flare_valid_construction(self) -> None:
        req = _valid_request()
        assert isinstance(req, FlareRequest)

    def test_flow_stored(self) -> None:
        req = _valid_request(total_flow_kg_hr=2000.0)
        assert req.total_flow_kg_hr == pytest.approx(2000.0)

    def test_contracts_flare_temperature_stored(self) -> None:
        req = _valid_request(temperature_k=500.0)
        assert req.temperature_k == pytest.approx(500.0)

    def test_contracts_flare_pressure_stored(self) -> None:
        req = _valid_request(pressure_bar=1.5)
        assert req.pressure_bar == pytest.approx(1.5)

    def test_gas_composition_stored(self) -> None:
        comp = {"H2": 50.0, "N2": 50.0}
        req = _valid_request(gas_composition=comp)
        assert req.gas_composition["H2"] == pytest.approx(50.0)

    def test_zero_flow_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_request(total_flow_kg_hr=0.0)

    def test_contracts_flare_zero_temperature_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_request(temperature_k=0.0)

    def test_contracts_flare_zero_pressure_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_request(pressure_bar=0.0)

    def test_negative_flow_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_request(total_flow_kg_hr=-100.0)


class TestFlareDesignOut:
    def _make_design(self) -> FlareDesignOut:
        return FlareDesignOut(
            height_m=30.0,
            diameter_m=0.5,
            exit_velocity_m_s=20.0,
            heat_release_kw=5000.0,
            radiation_intensity_kw_m2=1.6,
        )

    def test_contracts_flare_construction(self) -> None:
        design = self._make_design()
        assert isinstance(design, FlareDesignOut)

    def test_height_stored(self) -> None:
        design = self._make_design()
        assert design.height_m == pytest.approx(30.0)

    def test_heat_release_stored(self) -> None:
        design = self._make_design()
        assert design.heat_release_kw == pytest.approx(5000.0)


class TestRadiationZonesOut:
    def _make_zones(self) -> RadiationZonesOut:
        return RadiationZonesOut(
            lethal_m=15.0,
            damage_m=30.0,
            safe_m=100.0,
            comfort_m=200.0,
        )

    def test_contracts_flare_construction(self) -> None:
        zones = self._make_zones()
        assert isinstance(zones, RadiationZonesOut)

    def test_zones_ordered(self) -> None:
        zones = self._make_zones()
        # Lethal zone is closest, comfort zone is farthest
        assert zones.lethal_m < zones.damage_m
        assert zones.damage_m < zones.safe_m
        assert zones.safe_m < zones.comfort_m


class TestFlareResponse:
    def test_contracts_flare_construction(self) -> None:
        resp = FlareResponse(
            design=FlareDesignOut(
                height_m=30.0,
                diameter_m=0.5,
                exit_velocity_m_s=20.0,
                heat_release_kw=5000.0,
                radiation_intensity_kw_m2=1.6,
            ),
            radiation_zones=RadiationZonesOut(
                lethal_m=15.0, damage_m=30.0, safe_m=100.0, comfort_m=200.0
            ),
            combustion_efficiency=0.98,
        )
        assert isinstance(resp, FlareResponse)

    def test_combustion_efficiency_stored(self) -> None:
        resp = FlareResponse(
            design=FlareDesignOut(
                height_m=30.0,
                diameter_m=0.5,
                exit_velocity_m_s=20.0,
                heat_release_kw=5000.0,
                radiation_intensity_kw_m2=1.6,
            ),
            radiation_zones=RadiationZonesOut(
                lethal_m=15.0, damage_m=30.0, safe_m=100.0, comfort_m=200.0
            ),
            combustion_efficiency=0.98,
        )
        assert resp.combustion_efficiency == pytest.approx(0.98)
