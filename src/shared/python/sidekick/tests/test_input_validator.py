"""Tests for upstream_drift_tools.protocols.InputValidator.

Covers require_positive, require_in_range, require_keys,
validate_temperature, validate_pressure, validate_composition.
"""

from __future__ import annotations

import pytest
from upstream_drift_tools.protocols import InputValidator


class TestRequirePositive:
    def test_positive_value_passes(self):
        InputValidator.require_positive("flow", 1.0)  # no exception

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="flow"):
            InputValidator.require_positive("flow", 0.0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="pressure"):
            InputValidator.require_positive("pressure", -5.0)


class TestRequireInRange:
    def test_within_range_passes(self):
        InputValidator.require_in_range("x", 5.0, 0.0, 10.0)

    def test_at_lower_bound_passes(self):
        InputValidator.require_in_range("x", 0.0, 0.0, 10.0)

    def test_at_upper_bound_passes(self):
        InputValidator.require_in_range("x", 10.0, 0.0, 10.0)

    def test_below_range_raises(self):
        with pytest.raises(ValueError, match="range"):
            InputValidator.require_in_range("temp", -1.0, 0.0, 100.0)

    def test_above_range_raises(self):
        with pytest.raises(ValueError, match="range"):
            InputValidator.require_in_range("temp", 101.0, 0.0, 100.0)


class TestRequireKeys:
    def test_all_keys_present_passes(self):
        InputValidator.require_keys({"a": 1, "b": 2}, {"a", "b"})

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="Missing"):
            InputValidator.require_keys({"a": 1}, {"a", "b"})

    def test_extra_keys_allowed(self):
        InputValidator.require_keys({"a": 1, "b": 2, "c": 3}, {"a"})


class TestValidateTemperature:
    def test_positive_temperature_passes(self):
        InputValidator.validate_temperature(300.0)

    def test_zero_temp_raises(self):
        with pytest.raises(ValueError, match="Temperature"):
            InputValidator.validate_temperature(0.0)

    def test_negative_temp_raises(self):
        with pytest.raises(ValueError, match="Temperature"):
            InputValidator.validate_temperature(-10.0)


class TestValidatePressure:
    def test_positive_pressure_passes(self):
        InputValidator.validate_pressure(101325.0)

    def test_zero_pressure_raises(self):
        with pytest.raises(ValueError, match="Pressure"):
            InputValidator.validate_pressure(0.0)

    def test_negative_pressure_raises(self):
        with pytest.raises(ValueError, match="Pressure"):
            InputValidator.validate_pressure(-100.0)


class TestValidateComposition:
    def test_valid_composition_passes(self):
        InputValidator.validate_composition({"CO2": 0.5, "N2": 0.5})

    def test_negative_fraction_raises(self):
        with pytest.raises(ValueError, match="negative"):
            InputValidator.validate_composition({"CO2": -0.1, "N2": 1.1})

    def test_fractions_not_summing_to_one_raises(self):
        with pytest.raises(ValueError, match="sum"):
            InputValidator.validate_composition({"CO2": 0.3, "N2": 0.3})

    def test_custom_tolerance_tight(self):
        """Very tight tolerance should fail fractions that sum to 0.9999."""
        with pytest.raises(ValueError, match="sum"):
            InputValidator.validate_composition(
                {"CO2": 0.4999, "N2": 0.4999}, tolerance=1e-6
            )

    def test_custom_tolerance_loose(self):
        """Loose tolerance should accept fractions that sum to ~0.99."""
        InputValidator.validate_composition({"CO2": 0.5, "N2": 0.49}, tolerance=0.02)
