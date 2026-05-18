"""Tests for sidekick.protocols (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from sidekick.protocols import (
    CalculationResult,
    Calculator,
    DataTransformer,
    InputValidator,
    ProcessCalculator,
    StateSerializable,
    UnitConverter,
    ValidationResult,
)

# ---------------------------------------------------------------------------
# CalculationResult dataclass
# ---------------------------------------------------------------------------


class TestCalculationResult:
    def test_protocols_defaults(self) -> None:
        result = CalculationResult()
        assert result.values == {}
        assert result.units == {}
        assert result.warnings == []
        assert result.metadata == {}

    def test_construct_with_values(self) -> None:
        result = CalculationResult(values={"x": 1.0}, units={"x": "m"})
        assert result.values["x"] == 1.0
        assert result.units["x"] == "m"


# ---------------------------------------------------------------------------
# ValidationResult dataclass
# ---------------------------------------------------------------------------


class TestValidationResult:
    def test_default_valid_true(self) -> None:
        vr = ValidationResult()
        assert vr.valid is True
        assert vr.errors == []
        assert vr.warnings == []

    def test_invalid(self) -> None:
        vr = ValidationResult(valid=False, errors=["bad input"])
        assert vr.valid is False
        assert len(vr.errors) == 1


# ---------------------------------------------------------------------------
# Protocol runtime-checkable: plain object is not
# ---------------------------------------------------------------------------


class TestProtocolsRuntimeCheckable:
    def test_calculator_plain_object_false(self) -> None:
        assert not isinstance(object(), Calculator)

    def test_process_calculator_plain_object_false(self) -> None:
        assert not isinstance(object(), ProcessCalculator)

    def test_data_transformer_plain_object_false(self) -> None:
        assert not isinstance(object(), DataTransformer)

    def test_state_serializable_plain_object_false(self) -> None:
        assert not isinstance(object(), StateSerializable)

    def test_unit_converter_plain_object_false(self) -> None:
        assert not isinstance(object(), UnitConverter)


# ---------------------------------------------------------------------------
# InputValidator
# ---------------------------------------------------------------------------


class TestInputValidatorRequirePositive:
    def test_positive_passes(self) -> None:
        InputValidator.require_positive("x", 1.0)  # should not raise

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            InputValidator.require_positive("x", 0.0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            InputValidator.require_positive("x", -1.0)


class TestInputValidatorRequireInRange:
    def test_in_range_passes(self) -> None:
        InputValidator.require_in_range("t", 50.0, 0.0, 100.0)

    def test_at_lower_bound_passes(self) -> None:
        InputValidator.require_in_range("t", 0.0, 0.0, 100.0)

    def test_at_upper_bound_passes(self) -> None:
        InputValidator.require_in_range("t", 100.0, 0.0, 100.0)

    def test_below_range_raises(self) -> None:
        with pytest.raises(ValueError):
            InputValidator.require_in_range("t", -1.0, 0.0, 100.0)

    def test_above_range_raises(self) -> None:
        with pytest.raises(ValueError):
            InputValidator.require_in_range("t", 101.0, 0.0, 100.0)


class TestInputValidatorRequireKeys:
    def test_all_keys_present_passes(self) -> None:
        InputValidator.require_keys({"a": 1, "b": 2}, {"a", "b"})

    def test_missing_key_raises(self) -> None:
        with pytest.raises(ValueError, match="Missing required keys"):
            InputValidator.require_keys({"a": 1}, {"a", "b"})
