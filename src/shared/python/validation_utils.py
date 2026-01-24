"""Validation utilities for eliminating validation code duplication.

This module provides reusable validation patterns.

Usage:
    from src.shared.python.validation_utils import (
        validate_array_shape,
        validate_positive,
        validate_range,
        validate_file_exists,
    )

    validate_array_shape(data, (3, 3), "rotation matrix")
    validate_positive(mass, "mass")
    validate_range(angle, -np.pi, np.pi, "angle")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)


def validate_array_shape(
    array: np.ndarray,
    expected_shape: tuple[int, ...],
    name: str = "array",
) -> None:
    """Validate array has expected shape.

    Args:
        array: Array to validate
        expected_shape: Expected shape tuple
        name: Name for error messages

    Raises:
        ValueError: If shape doesn't match

    Example:
        validate_array_shape(rotation, (3, 3), "rotation matrix")
    """
    if array.shape != expected_shape:
        raise ValueError(
            f"{name} shape mismatch: expected {expected_shape}, got {array.shape}"
        )


def validate_array_dimensions(
    array: np.ndarray,
    expected_ndim: int,
    name: str = "array",
) -> None:
    """Validate array has expected number of dimensions.

    Args:
        array: Array to validate
        expected_ndim: Expected number of dimensions
        name: Name for error messages

    Raises:
        ValueError: If dimensions don't match

    Example:
        validate_array_dimensions(vector, 1, "velocity vector")
    """
    if array.ndim != expected_ndim:
        raise ValueError(
            f"{name} dimension mismatch: expected {expected_ndim}D, got {array.ndim}D"
        )


def validate_array_length(
    array: np.ndarray,
    expected_length: int,
    name: str = "array",
) -> None:
    """Validate array has expected length.

    Args:
        array: Array to validate
        expected_length: Expected length
        name: Name for error messages

    Raises:
        ValueError: If length doesn't match

    Example:
        validate_array_length(positions, model.nq, "position vector")
    """
    if len(array) != expected_length:
        raise ValueError(
            f"{name} length mismatch: expected {expected_length}, got {len(array)}"
        )


def validate_positive(
    value: float | int,
    name: str = "value",
    strict: bool = True,
) -> None:
    """Validate value is positive.

    Args:
        value: Value to validate
        name: Name for error messages
        strict: If True, value must be > 0; if False, value must be >= 0

    Raises:
        ValueError: If value is not positive

    Example:
        validate_positive(mass, "mass")
        validate_positive(distance, "distance", strict=False)  # Allow zero
    """
    if strict and value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    elif not strict and value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def validate_range(
    value: float | int,
    min_value: float | int,
    max_value: float | int,
    name: str = "value",
    inclusive: bool = True,
) -> None:
    """Validate value is within range.

    Args:
        value: Value to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        name: Name for error messages
        inclusive: If True, endpoints are included

    Raises:
        ValueError: If value is out of range

    Example:
        validate_range(angle, -np.pi, np.pi, "angle")
        validate_range(probability, 0, 1, "probability", inclusive=True)
    """
    if inclusive:
        if not (min_value <= value <= max_value):
            raise ValueError(
                f"{name} must be in [{min_value}, {max_value}], got {value}"
            )
    else:
        if not (min_value < value < max_value):
            raise ValueError(
                f"{name} must be in ({min_value}, {max_value}), got {value}"
            )


def validate_file_exists(
    path: str | Path,
    name: str = "file",
) -> Path:
    """Validate file exists and return Path object.

    Args:
        path: Path to file
        name: Name for error messages

    Returns:
        Path object

    Raises:
        FileNotFoundError: If file doesn't exist

    Example:
        model_path = validate_file_exists("model.xml", "model file")
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    if not path_obj.is_file():
        raise ValueError(f"{name} is not a file: {path}")
    return path_obj


def validate_directory_exists(
    path: str | Path,
    name: str = "directory",
) -> Path:
    """Validate directory exists and return Path object.

    Args:
        path: Path to directory
        name: Name for error messages

    Returns:
        Path object

    Raises:
        FileNotFoundError: If directory doesn't exist

    Example:
        output_dir = validate_directory_exists("output", "output directory")
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    if not path_obj.is_dir():
        raise ValueError(f"{name} is not a directory: {path}")
    return path_obj


def validate_extension(
    path: str | Path,
    allowed_extensions: list[str],
    name: str = "file",
) -> None:
    """Validate file has allowed extension.

    Args:
        path: Path to file
        allowed_extensions: List of allowed extensions (e.g., [".xml", ".urdf"])
        name: Name for error messages

    Raises:
        ValueError: If extension is not allowed

    Example:
        validate_extension("model.xml", [".xml", ".urdf"], "model file")
    """
    path_obj = Path(path)
    if path_obj.suffix.lower() not in allowed_extensions:
        raise ValueError(
            f"{name} has invalid extension: {path_obj.suffix}. "
            f"Allowed: {', '.join(allowed_extensions)}"
        )


def validate_not_none(
    value: Any,
    name: str = "value",
) -> None:
    """Validate value is not None.

    Args:
        value: Value to validate
        name: Name for error messages

    Raises:
        ValueError: If value is None

    Example:
        validate_not_none(model, "model")
    """
    if value is None:
        raise ValueError(f"{name} cannot be None")


def validate_type(
    value: Any,
    expected_type: type | tuple[type, ...],
    name: str = "value",
) -> None:
    """Validate value has expected type.

    Args:
        value: Value to validate
        expected_type: Expected type or tuple of types
        name: Name for error messages

    Raises:
        TypeError: If type doesn't match

    Example:
        validate_type(config, dict, "configuration")
        validate_type(value, (int, float), "numeric value")
    """
    if not isinstance(value, expected_type):
        if isinstance(expected_type, tuple):
            type_names = " or ".join(t.__name__ for t in expected_type)
        else:
            type_names = expected_type.__name__
        raise TypeError(f"{name} must be {type_names}, got {type(value).__name__}")


def validate_dict_keys(
    data: dict,
    required_keys: list[str],
    optional_keys: list[str] | None = None,
    name: str = "dictionary",
) -> None:
    """Validate dictionary has required keys.

    Args:
        data: Dictionary to validate
        required_keys: List of required keys
        optional_keys: List of optional keys (for validation)
        name: Name for error messages

    Raises:
        ValueError: If required keys are missing or unknown keys present

    Example:
        validate_dict_keys(
            config,
            required_keys=["engine", "model"],
            optional_keys=["timestep", "gravity"],
            name="configuration"
        )
    """
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise ValueError(f"{name} missing required keys: {', '.join(missing_keys)}")

    if optional_keys is not None:
        all_allowed = set(required_keys) | set(optional_keys)
        unknown_keys = [key for key in data if key not in all_allowed]
        if unknown_keys:
            logger.warning(f"{name} has unknown keys: {', '.join(unknown_keys)}")


def validate_numeric(
    value: Any,
    name: str = "value",
    allow_nan: bool = False,
    allow_inf: bool = False,
) -> None:
    """Validate value is numeric and finite.

    Args:
        value: Value to validate
        name: Name for error messages
        allow_nan: Whether to allow NaN values
        allow_inf: Whether to allow infinite values

    Raises:
        TypeError: If value is not numeric
        ValueError: If value is NaN or infinite (when not allowed)

    Example:
        validate_numeric(mass, "mass")
        validate_numeric(result, "result", allow_nan=True)
    """
    if not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")

    if not allow_nan and np.isnan(value):
        raise ValueError(f"{name} cannot be NaN")

    if not allow_inf and np.isinf(value):
        raise ValueError(f"{name} cannot be infinite")


class ValidationError(Exception):
    """Custom exception for validation errors."""

    pass


def validate_all(*validations: tuple[callable, tuple, dict]) -> None:
    """Run multiple validations and collect all errors.

    Args:
        *validations: Tuples of (validation_func, args, kwargs)

    Raises:
        ValidationError: If any validation fails, with all error messages

    Example:
        validate_all(
            (validate_positive, (mass, "mass"), {}),
            (validate_range, (angle, -np.pi, np.pi, "angle"), {}),
            (validate_array_shape, (rotation, (3, 3), "rotation"), {}),
        )
    """
    errors = []

    for validation_func, args, kwargs in validations:
        try:
            validation_func(*args, **kwargs)
        except (ValueError, TypeError, FileNotFoundError) as e:
            errors.append(str(e))

    if errors:
        raise ValidationError(
            f"Validation failed with {len(errors)} error(s):\n"
            + "\n".join(f"  - {error}" for error in errors)
        )
