"""Tests for sidekick.calculators.conversion.core (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from sidekick.calculators.conversion.core import (
    convert_temperature,
    convert_via_table,
    scfm_to_standard_m3_per_hour,
    standard_m3_per_hour_to_scfm,
)
from sidekick.calculators.conversion.tables import (
    StandardCondition,
)

# ---------------------------------------------------------------------------
# convert_temperature
# ---------------------------------------------------------------------------


class TestConvertTemperature:
    def test_same_unit_identity(self) -> None:
        assert convert_temperature(100.0, "C", "C") == 100.0

    def test_celsius_to_kelvin(self) -> None:
        result = convert_temperature(0.0, "C", "K")
        assert abs(result - 273.15) < 0.01

    def test_kelvin_to_celsius(self) -> None:
        result = convert_temperature(273.15, "K", "C")
        assert abs(result - 0.0) < 0.01

    def test_fahrenheit_to_celsius(self) -> None:
        result = convert_temperature(32.0, "F", "C")
        assert abs(result - 0.0) < 0.01

    def test_celsius_to_fahrenheit(self) -> None:
        result = convert_temperature(100.0, "C", "F")
        assert abs(result - 212.0) < 0.01

    def test_kelvin_to_rankine(self) -> None:
        result = convert_temperature(273.15, "K", "R")
        # 273.15 K * 1.8 = 491.67 R
        assert abs(result - 491.67) < 0.1

    def test_conversion_core_unknown_from_unit_raises(self) -> None:
        with pytest.raises(ValueError):
            convert_temperature(100.0, "X", "K")

    def test_conversion_core_unknown_to_unit_raises(self) -> None:
        with pytest.raises(ValueError):
            convert_temperature(100.0, "K", "Z")


# ---------------------------------------------------------------------------
# convert_via_table
# ---------------------------------------------------------------------------


class TestConvertViaTable:
    _TABLE = {"m": 1.0, "km": 1000.0, "cm": 0.01}

    def test_same_unit_identity(self) -> None:
        result = convert_via_table(5.0, "m", "m", self._TABLE)
        assert result == 5.0

    def test_m_to_km(self) -> None:
        result = convert_via_table(1000.0, "m", "km", self._TABLE)
        assert abs(result - 1.0) < 1e-10

    def test_km_to_m(self) -> None:
        result = convert_via_table(1.0, "km", "m", self._TABLE)
        assert abs(result - 1000.0) < 1e-10

    def test_cm_to_m(self) -> None:
        result = convert_via_table(100.0, "cm", "m", self._TABLE)
        assert abs(result - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# scfm_to_standard_m3_per_hour / inverse
# ---------------------------------------------------------------------------


class TestScfmConversions:
    _STP = StandardCondition.STP
    _SCFM_60F = StandardCondition.SCFM_60F

    def test_scfm_to_m3h_positive(self) -> None:
        result = scfm_to_standard_m3_per_hour(1.0, self._STP, self._SCFM_60F)
        assert result > 0.0

    def test_m3h_to_scfm_positive(self) -> None:
        result = standard_m3_per_hour_to_scfm(1.0, self._SCFM_60F, self._STP)
        assert result > 0.0

    def test_conversion_core_roundtrip(self) -> None:
        original = 100.0
        m3h = scfm_to_standard_m3_per_hour(original, self._STP, self._SCFM_60F)
        back = standard_m3_per_hour_to_scfm(m3h, self._SCFM_60F, self._STP)
        assert abs(back - original) < 0.01
