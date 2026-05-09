"""Tests for src.shared.python.calc_backend.contracts.syngas_water (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from src.shared.python.calc_backend.contracts.syngas_water import (
    CondensationRiskOut,
    SyngasWaterRequest,
    SyngasWaterResponse,
    WaterContentOut,
)


class TestSyngasWaterRequest:
    def test_contracts_syngas_water_valid_construction(self) -> None:
        req = SyngasWaterRequest(temperature_c=25.0, pressure_bar=1.0)
        assert isinstance(req, SyngasWaterRequest)

    def test_contracts_syngas_water_temperature_stored(self) -> None:
        req = SyngasWaterRequest(temperature_c=100.0, pressure_bar=5.0)
        assert req.temperature_c == pytest.approx(100.0)

    def test_contracts_syngas_water_pressure_stored(self) -> None:
        req = SyngasWaterRequest(temperature_c=50.0, pressure_bar=2.5)
        assert req.pressure_bar == pytest.approx(2.5)

    def test_default_composition_key(self) -> None:
        req = SyngasWaterRequest(temperature_c=25.0, pressure_bar=1.0)
        assert req.composition_key == "typical_syngas"

    def test_default_method(self) -> None:
        req = SyngasWaterRequest(temperature_c=25.0, pressure_bar=1.0)
        assert req.method == "auto"

    def test_temperature_too_low_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SyngasWaterRequest(temperature_c=-100.0, pressure_bar=1.0)

    def test_temperature_too_high_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SyngasWaterRequest(temperature_c=500.0, pressure_bar=1.0)

    def test_contracts_syngas_water_zero_pressure_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SyngasWaterRequest(temperature_c=25.0, pressure_bar=0.0)

    def test_pressure_over_500_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SyngasWaterRequest(temperature_c=25.0, pressure_bar=600.0)

    def test_custom_method(self) -> None:
        req = SyngasWaterRequest(temperature_c=25.0, pressure_bar=1.0, method="antoine")
        assert req.method == "antoine"


class TestWaterContentOut:
    def _make_water_content(self) -> WaterContentOut:
        return WaterContentOut(
            mole_fraction_water=0.03,
            water_content_mg_per_nm3=24000.0,
            water_content_ppmv=30000.0,
            water_content_g_per_m3=23.0,
            water_content_lb_per_mmscf=1500.0,
            vapor_pressure_bar=0.03,
            dew_point_c=25.0,
        )

    def test_contracts_syngas_water_construction(self) -> None:
        wc = self._make_water_content()
        assert isinstance(wc, WaterContentOut)

    def test_mole_fraction_stored(self) -> None:
        wc = self._make_water_content()
        assert wc.mole_fraction_water == pytest.approx(0.03)

    def test_dew_point_stored(self) -> None:
        wc = self._make_water_content()
        assert wc.dew_point_c == pytest.approx(25.0)


class TestCondensationRiskOut:
    def test_contracts_syngas_water_construction(self) -> None:
        risk = CondensationRiskOut(
            temperature_margin_c=20.0,
            condensation_risk="Low",
            recommended_temperature_c=50.0,
        )
        assert isinstance(risk, CondensationRiskOut)

    def test_risk_label_stored(self) -> None:
        risk = CondensationRiskOut(
            temperature_margin_c=-5.0,
            condensation_risk="High",
            recommended_temperature_c=80.0,
        )
        assert risk.condensation_risk == "High"


class TestSyngasWaterResponse:
    def test_contracts_syngas_water_construction(self) -> None:
        resp = SyngasWaterResponse(
            water_content=WaterContentOut(
                mole_fraction_water=0.03,
                water_content_mg_per_nm3=24000.0,
                water_content_ppmv=30000.0,
                water_content_g_per_m3=23.0,
                water_content_lb_per_mmscf=1500.0,
                vapor_pressure_bar=0.03,
                dew_point_c=25.0,
            ),
            risk_assessment=CondensationRiskOut(
                temperature_margin_c=20.0,
                condensation_risk="Low",
                recommended_temperature_c=50.0,
            ),
        )
        assert isinstance(resp, SyngasWaterResponse)
