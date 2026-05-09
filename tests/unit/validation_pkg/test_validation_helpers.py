"""Tests for src.shared.python.validation_pkg.validation_helpers (Issues #1949, #1744)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from src.shared.python.validation_pkg.validation_helpers import (
    PhysicsValidationError,
    ValidationLevel,
    validate_finite,
    validate_magnitude,
)


class TestValidateFinite:
    def test_finite_array_passes_standard(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        # Should not raise
        validate_finite(arr, "test", ValidationLevel.STANDARD)

    def test_nan_raises_in_standard(self) -> None:
        arr = np.array([1.0, np.nan, 3.0])
        with pytest.raises(PhysicsValidationError):
            validate_finite(arr, "test", ValidationLevel.STANDARD)

    def test_inf_raises_in_standard(self) -> None:
        arr = np.array([1.0, np.inf, 3.0])
        with pytest.raises(PhysicsValidationError):
            validate_finite(arr, "test", ValidationLevel.STANDARD)

    def test_nan_raises_in_strict(self) -> None:
        arr = np.array([np.nan])
        with pytest.raises(PhysicsValidationError):
            validate_finite(arr, "test", ValidationLevel.STRICT)

    def test_nan_warns_in_permissive(self) -> None:
        arr = np.array([np.nan, 1.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_finite(arr, "test", ValidationLevel.PERMISSIVE)
        assert len(w) == 1

    def test_zeros_are_finite(self) -> None:
        arr = np.zeros(10)
        validate_finite(arr, "zeros", ValidationLevel.STRICT)

    def test_empty_array_passes(self) -> None:
        arr = np.array([])
        validate_finite(arr, "empty", ValidationLevel.STANDARD)


class TestValidateMagnitude:
    def test_small_values_pass(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        # Should not raise or warn
        validate_magnitude(arr, "test", max_value=100.0, units="m/s")

    def test_exceeds_max_raises_in_strict(self) -> None:
        arr = np.array([1000.0])
        with pytest.raises(PhysicsValidationError):
            validate_magnitude(
                arr, "test", max_value=100.0, units="m/s", level=ValidationLevel.STRICT
            )

    def test_exceeds_max_warns_in_standard(self) -> None:
        arr = np.array([1000.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_magnitude(
                arr,
                "test",
                max_value=100.0,
                units="m/s",
                level=ValidationLevel.STANDARD,
            )
        assert len(w) == 1

    def test_exceeds_max_warns_in_permissive(self) -> None:
        arr = np.array([1000.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_magnitude(
                arr,
                "test",
                max_value=100.0,
                units="m/s",
                level=ValidationLevel.PERMISSIVE,
            )
        assert len(w) == 1

    def test_negative_values_checked_by_abs(self) -> None:
        arr = np.array([-200.0])
        with pytest.raises(PhysicsValidationError):
            validate_magnitude(
                arr, "test", max_value=100.0, units="m/s", level=ValidationLevel.STRICT
            )

    def test_zero_passes_any_threshold(self) -> None:
        arr = np.zeros(10)
        validate_magnitude(
            arr, "zeros", max_value=1.0, units="m/s", level=ValidationLevel.STRICT
        )


class TestPhysicsValidationError:
    def test_is_value_error(self) -> None:
        err = PhysicsValidationError("invalid input")
        assert isinstance(err, ValueError)

    def test_validation_helpers_message(self) -> None:
        err = PhysicsValidationError("test message")
        assert "test message" in str(err)

    def test_can_be_raised(self) -> None:
        with pytest.raises(PhysicsValidationError):
            raise PhysicsValidationError("test")


class TestValidationLevel:
    def test_permissive_exists(self) -> None:
        assert ValidationLevel.PERMISSIVE is not None

    def test_standard_exists(self) -> None:
        assert ValidationLevel.STANDARD is not None

    def test_strict_exists(self) -> None:
        assert ValidationLevel.STRICT is not None

    def test_all_levels_different(self) -> None:
        assert ValidationLevel.PERMISSIVE != ValidationLevel.STANDARD
        assert ValidationLevel.STANDARD != ValidationLevel.STRICT
        assert ValidationLevel.PERMISSIVE != ValidationLevel.STRICT
