"""Tests for src.shared.python.upstream_drift_tools.calculators.conversion.service (Issues #1949, #1744)."""

from __future__ import annotations

import math

import pytest
from src.shared.python.upstream_drift_tools.calculators.conversion.service import (
    ConversionResult,
    IncompatibleUnitsError,
    InvalidValueError,
    UnitConversionService,
    UnknownUnitError,
)

# ---------------------------------------------------------------------------
# ConversionResult dataclass
# ---------------------------------------------------------------------------


class TestConversionResult:
    def test_conversion_service_construct(self) -> None:
        r = ConversionResult(value=1.0, from_unit="m", to_unit="km")
        assert r.value == 1.0
        assert r.from_unit == "m"
        assert r.to_unit == "km"

    def test_default_uncertainty(self) -> None:
        r = ConversionResult(value=1.0, from_unit="m", to_unit="km")
        assert r.uncertainty == 0.0

    def test_conversion_service_default_warnings_empty(self) -> None:
        r = ConversionResult(value=1.0, from_unit="m", to_unit="km")
        assert r.warnings == []


# ---------------------------------------------------------------------------
# UnitConversionService.convert
# ---------------------------------------------------------------------------


class TestConvert:
    _SVC = UnitConversionService()

    def test_returns_conversion_result(self) -> None:
        result = self._SVC.convert(1.0, "kg", "kg")
        assert isinstance(result, ConversionResult)

    def test_identity_kg_to_kg(self) -> None:
        result = self._SVC.convert(5.0, "kg", "kg")
        assert abs(result.value - 5.0) < 1e-10

    def test_kg_to_g(self) -> None:
        result = self._SVC.convert(1.0, "kg", "g")
        assert abs(result.value - 1000.0) < 0.01

    def test_g_to_kg_roundtrip(self) -> None:
        grams = self._SVC.convert(1.0, "kg", "g").value
        back = self._SVC.convert(grams, "g", "kg").value
        assert abs(back - 1.0) < 1e-6

    def test_celsius_to_kelvin(self) -> None:
        result = self._SVC.convert(0.0, "C", "K")
        assert abs(result.value - 273.15) < 0.01

    def test_conversion_service_unknown_from_unit_raises(self) -> None:
        with pytest.raises(UnknownUnitError):
            self._SVC.convert(1.0, "INVALID_UNIT_XYZ", "kg")

    def test_conversion_service_unknown_to_unit_raises(self) -> None:
        with pytest.raises(UnknownUnitError):
            self._SVC.convert(1.0, "kg", "INVALID_UNIT_XYZ")

    def test_incompatible_units_raises(self) -> None:
        with pytest.raises(IncompatibleUnitsError):
            self._SVC.convert(1.0, "kg", "K")

    def test_infinite_value_raises(self) -> None:
        with pytest.raises(InvalidValueError):
            self._SVC.convert(math.inf, "kg", "g")

    def test_nan_value_raises(self) -> None:
        with pytest.raises(InvalidValueError):
            self._SVC.convert(math.nan, "kg", "g")
