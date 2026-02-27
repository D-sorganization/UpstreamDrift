"""test_conversion.py module."""

import pytest
from upstream_drift_tools.calculators.conversion.service import (
    UnknownUnitError,
    convert,
    get_service,
)


class TestUnitConversion:
    def test_length_conversion(self):
        # 1 inch = 2.54 cm
        val = convert(1.0, "in", "cm")
        assert abs(val - 2.54) < 1e-6

    def test_temperature_conversion(self):
        # 0C = 32F
        val = convert(0.0, "C", "F")
        assert abs(val - 32.0) < 1e-6

        # 100C = 212F
        val = convert(100.0, "C", "F")
        assert abs(val - 212.0) < 1e-6

    def test_pressure_conversion(self):
        # 1 atm = 101325 Pa
        val = convert(1.0, "atm", "Pa")
        assert abs(val - 101325.0) < 1e-1

    def test_invalid_unit(self):
        with pytest.raises(UnknownUnitError):
            convert(1.0, "invalid_unit_xyz", "m")

    def test_service_singleton(self):
        s1 = get_service()
        s2 = get_service()
        assert s1 is s2
