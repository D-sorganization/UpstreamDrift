"""Tests for src.shared.python.calc_backend.contracts.wgs_reactor (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from src.shared.python.calc_backend.contracts.wgs_reactor import (
    WGSEquilibriumOut,
    WGSReactorRequest,
    WGSReactorResponse,
    WGSSizingOut,
)


def _valid_request(**kwargs) -> WGSReactorRequest:
    defaults = {
        "inlet_composition": {"CO": 25.0, "H2": 30.0, "CO2": 10.0, "H2O": 35.0},
        "temperature_k": 600.0,
        "pressure_bar": 5.0,
    }
    defaults.update(kwargs)
    return WGSReactorRequest(**defaults)


class TestWGSReactorRequest:
    def test_contracts_wgs_reactor_valid_construction(self) -> None:
        req = _valid_request()
        assert isinstance(req, WGSReactorRequest)

    def test_contracts_wgs_reactor_temperature_stored(self) -> None:
        req = _valid_request(temperature_k=700.0)
        assert req.temperature_k == pytest.approx(700.0)

    def test_default_steam_ratio(self) -> None:
        req = _valid_request()
        assert req.steam_ratio == pytest.approx(2.0)

    def test_default_catalyst_type(self) -> None:
        req = _valid_request()
        assert req.catalyst_type == "HTS"

    def test_default_feed_rate(self) -> None:
        req = _valid_request()
        assert req.feed_rate_kmol_hr == 0.0

    def test_contracts_wgs_reactor_zero_temperature_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_request(temperature_k=0.0)

    def test_contracts_wgs_reactor_zero_pressure_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_request(pressure_bar=0.0)

    def test_zero_steam_ratio_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_request(steam_ratio=0.0)

    def test_inlet_composition_stored(self) -> None:
        req = _valid_request()
        assert "CO" in req.inlet_composition

    def test_custom_feed_rate(self) -> None:
        req = _valid_request(feed_rate_kmol_hr=100.0)
        assert req.feed_rate_kmol_hr == pytest.approx(100.0)


class TestWGSEquilibriumOut:
    def _make_equilibrium(self) -> WGSEquilibriumOut:
        return WGSEquilibriumOut(
            conversion_pct=85.0,
            composition={"CO": 3.0, "H2": 55.0, "CO2": 22.0, "H2O": 20.0},
            h2_co_ratio=18.3,
            equilibrium_constant=12.5,
            heat_released_kj=-41.2,
        )

    def test_contracts_wgs_reactor_construction(self) -> None:
        eq = self._make_equilibrium()
        assert isinstance(eq, WGSEquilibriumOut)

    def test_conversion_stored(self) -> None:
        eq = self._make_equilibrium()
        assert eq.conversion_pct == pytest.approx(85.0)

    def test_h2_co_ratio_stored(self) -> None:
        eq = self._make_equilibrium()
        assert eq.h2_co_ratio == pytest.approx(18.3)


class TestWGSReactorResponse:
    def _make_equilibrium(self) -> WGSEquilibriumOut:
        return WGSEquilibriumOut(
            conversion_pct=85.0,
            composition={"CO": 3.0, "H2": 55.0, "CO2": 22.0, "H2O": 20.0},
            h2_co_ratio=18.3,
            equilibrium_constant=12.5,
            heat_released_kj=-41.2,
        )

    def test_construction_no_sizing(self) -> None:
        resp = WGSReactorResponse(equilibrium=self._make_equilibrium())
        assert isinstance(resp, WGSReactorResponse)

    def test_sizing_defaults_to_none(self) -> None:
        resp = WGSReactorResponse(equilibrium=self._make_equilibrium())
        assert resp.sizing is None

    def test_with_sizing(self) -> None:
        sizing = WGSSizingOut(
            reactor_volume_m3=5.0,
            catalyst_volume_m3=3.5,
            diameter_m=1.5,
            length_m=3.0,
            heat_duty_kw=500.0,
            ghsv=2000.0,
        )
        resp = WGSReactorResponse(equilibrium=self._make_equilibrium(), sizing=sizing)
        assert resp.sizing is not None
        assert resp.sizing.reactor_volume_m3 == pytest.approx(5.0)
