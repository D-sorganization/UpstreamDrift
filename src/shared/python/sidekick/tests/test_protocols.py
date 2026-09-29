from typing import Any

import pytest
from sidekick.protocols import (
    CalculationResult,
    InputValidator,
    ValidationResult,
)


def test_calculation_result_defaults() -> Any:
    res = CalculationResult()
    assert res.values == {}
    assert res.units == {}
    assert res.warnings == []
    assert res.metadata == {}


def test_validation_result_defaults() -> Any:
    vr = ValidationResult()
    assert vr.valid is True
    assert vr.errors == []
    assert vr.warnings == []


def test_input_validator_require_positive() -> Any:
    validator = InputValidator()
    validator.require_positive("flow", 10.0)
    with pytest.raises(ValueError, match="flow must be positive, got 0.0"):
        validator.require_positive("flow", 0.0)
    with pytest.raises(ValueError, match="flow must be positive, got -5.0"):
        validator.require_positive("flow", -5.0)


def test_input_validator_require_in_range() -> Any:
    validator = InputValidator()
    validator.require_in_range("temp", 50.0, 0.0, 100.0)
    with pytest.raises(
        ValueError, match=r"temp must be in range \[0.0, 100.0\], got -10.0"
    ):
        validator.require_in_range("temp", -10.0, 0.0, 100.0)
    with pytest.raises(
        ValueError, match=r"temp must be in range \[0.0, 100.0\], got 110.0"
    ):
        validator.require_in_range("temp", 110.0, 0.0, 100.0)


def test_input_validator_require_keys() -> Any:
    validator = InputValidator()
    inputs = {"a": 1, "b": 2}
    validator.require_keys(inputs, {"a", "b"})
    validator.require_keys(inputs, {"a"})
    with pytest.raises(ValueError, match="Missing required keys: \\['c'\\]"):
        validator.require_keys(inputs, {"a", "b", "c"})


def test_input_validator_validate_temperature() -> Any:
    validator = InputValidator()
    validator.validate_temperature(300.15)
    with pytest.raises(ValueError, match="Temperature must be > 0 K, got 0.0"):
        validator.validate_temperature(0.0)
    with pytest.raises(ValueError, match="Temperature must be > 0 K, got -10.0"):
        validator.validate_temperature(-10.0)


def test_input_validator_validate_pressure() -> Any:
    validator = InputValidator()
    validator.validate_pressure(101325.0)
    with pytest.raises(ValueError, match="Pressure must be > 0 Pa, got 0.0"):
        validator.validate_pressure(0.0)


def test_input_validator_validate_composition() -> Any:
    validator = InputValidator()

    # Valid composition
    validator.validate_composition({"H2": 0.5, "CO": 0.5})

    # Negative fraction
    with pytest.raises(
        ValueError, match="Composition fraction for 'H2' is negative: -0.1"
    ):
        validator.validate_composition({"H2": -0.1, "CO": 1.1})

    # Sum != 1.0
    with pytest.raises(ValueError, match=r"Composition fractions must sum to 1.0"):
        validator.validate_composition({"H2": 0.5, "CO": 0.6})
