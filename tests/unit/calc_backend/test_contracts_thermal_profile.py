"""Tests for src.shared.python.calc_backend.contracts.thermal_profile (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from src.shared.python.calc_backend.contracts.thermal_profile import (
    ThermalProfileDataPoint,
    ThermalProfileRequest,
    ThermalProfileResponse,
)


class TestThermalProfileRequest:
    def _valid_request(self, **kwargs) -> ThermalProfileRequest:
        defaults = {
            "thermal_mass_j_per_k": 5000.0,
            "heat_loss_coeff_w_per_k": 10.0,
        }
        defaults.update(kwargs)
        return ThermalProfileRequest(**defaults)

    def test_contracts_thermal_profile_valid_construction(self) -> None:
        req = self._valid_request()
        assert isinstance(req, ThermalProfileRequest)

    def test_default_initial_temp(self) -> None:
        req = self._valid_request()
        assert req.initial_temp_c == pytest.approx(25.0)

    def test_default_ambient_temp(self) -> None:
        req = self._valid_request()
        assert req.ambient_temp_c == pytest.approx(25.0)

    def test_default_power(self) -> None:
        req = self._valid_request()
        assert req.power_w == pytest.approx(5000.0)

    def test_default_power_profile(self) -> None:
        req = self._valid_request()
        assert req.power_profile == "constant"

    def test_default_t_end(self) -> None:
        req = self._valid_request()
        assert req.t_end_s == pytest.approx(3600.0)

    def test_contracts_thermal_profile_default_num_points(self) -> None:
        req = self._valid_request()
        assert req.num_points == 100

    def test_zero_thermal_mass_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._valid_request(thermal_mass_j_per_k=0.0)

    def test_negative_heat_loss_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._valid_request(heat_loss_coeff_w_per_k=-1.0)

    def test_zero_t_end_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._valid_request(t_end_s=0.0)

    def test_num_points_below_10_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._valid_request(num_points=5)

    def test_custom_initial_temp(self) -> None:
        req = self._valid_request(initial_temp_c=100.0)
        assert req.initial_temp_c == pytest.approx(100.0)

    def test_heat_loss_zero_allowed(self) -> None:
        req = self._valid_request(heat_loss_coeff_w_per_k=0.0)
        assert req.heat_loss_coeff_w_per_k == pytest.approx(0.0)


class TestThermalProfileDataPoint:
    def test_contracts_thermal_profile_construction(self) -> None:
        dp = ThermalProfileDataPoint(
            time_s=0.0, temperature_c=25.0, power_w=5000.0, heat_loss_w=0.0
        )
        assert isinstance(dp, ThermalProfileDataPoint)

    def test_values_stored(self) -> None:
        dp = ThermalProfileDataPoint(
            time_s=100.0, temperature_c=45.0, power_w=5000.0, heat_loss_w=200.0
        )
        assert dp.time_s == pytest.approx(100.0)
        assert dp.temperature_c == pytest.approx(45.0)
        assert dp.heat_loss_w == pytest.approx(200.0)


class TestThermalProfileResponse:
    def _make_response(self) -> ThermalProfileResponse:
        data = [
            ThermalProfileDataPoint(
                time_s=float(i * 36),
                temperature_c=25.0 + i,
                power_w=5000.0,
                heat_loss_w=i * 10.0,
            )
            for i in range(5)
        ]
        return ThermalProfileResponse(
            data=data,
            final_temp_c=29.0,
            max_temp_c=29.0,
            min_temp_c=25.0,
            temp_change_c=4.0,
        )

    def test_contracts_thermal_profile_construction(self) -> None:
        resp = self._make_response()
        assert isinstance(resp, ThermalProfileResponse)

    def test_data_length(self) -> None:
        resp = self._make_response()
        assert len(resp.data) == 5

    def test_steady_state_defaults_none(self) -> None:
        resp = self._make_response()
        assert resp.steady_state_temp_c is None

    def test_time_constant_defaults_none(self) -> None:
        resp = self._make_response()
        assert resp.time_constant_s is None

    def test_with_optional_fields(self) -> None:
        data = [
            ThermalProfileDataPoint(
                time_s=0.0, temperature_c=25.0, power_w=5000.0, heat_loss_w=0.0
            )
        ]
        resp = ThermalProfileResponse(
            data=data,
            final_temp_c=525.0,
            max_temp_c=525.0,
            min_temp_c=25.0,
            temp_change_c=500.0,
            steady_state_temp_c=525.0,
            time_constant_s=500.0,
        )
        assert resp.steady_state_temp_c == pytest.approx(525.0)
        assert resp.time_constant_s == pytest.approx(500.0)
