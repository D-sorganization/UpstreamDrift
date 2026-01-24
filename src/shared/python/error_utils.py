"""Centralized error handling utilities for the Golf Modeling Suite.

This module consolidates common error handling patterns and provides
standardized exception classes and error message factories.

Usage:
    from src.shared.python.error_utils import (
        GolfSuiteError,
        EngineNotAvailableError,
        ConfigurationError,
        ValidationError,
        format_import_error,
        format_file_error,
    )

    # Raise standardized exceptions
    raise EngineNotAvailableError("mujoco")

    # Use error message factories
    raise ImportError(format_import_error("pinocchio", "forward kinematics"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class GolfSuiteError(Exception):
    """Base exception for Golf Modeling Suite errors."""

    pass


class EngineNotAvailableError(GolfSuiteError):
    """Raised when a required physics engine is not available."""

    def __init__(
        self,
        engine_name: str,
        operation: str | None = None,
        install_hint: str | None = None,
    ):
        self.engine_name = engine_name
        self.operation = operation

        message = f"Physics engine '{engine_name}' is not available"
        if operation:
            message += f" for {operation}"
        message += "."

        if install_hint:
            message += f" Install with: {install_hint}"
        else:
            # Provide common install hints
            install_hints = {
                "mujoco": "pip install mujoco",
                "pinocchio": "conda install -c conda-forge pinocchio",
                "drake": "pip install drake",
                "opensim": "conda install -c opensim-org opensim",
                "dm_control": "pip install dm-control",
            }
            if engine_name.lower() in install_hints:
                message += f" Install with: {install_hints[engine_name.lower()]}"

        super().__init__(message)


class ConfigurationError(GolfSuiteError):
    """Raised when configuration is invalid or missing."""

    def __init__(
        self,
        config_key: str,
        reason: str | None = None,
        expected: Any = None,
        actual: Any = None,
    ):
        self.config_key = config_key
        self.reason = reason
        self.expected = expected
        self.actual = actual

        message = f"Configuration error for '{config_key}'"
        if reason:
            message += f": {reason}"
        if expected is not None and actual is not None:
            message += f". Expected: {expected}, got: {actual}"
        elif expected is not None:
            message += f". Expected: {expected}"

        super().__init__(message)


class ValidationError(GolfSuiteError):
    """Raised when input validation fails."""

    def __init__(
        self,
        field: str,
        value: Any = None,
        reason: str | None = None,
        valid_values: list[Any] | None = None,
    ):
        self.field = field
        self.value = value
        self.reason = reason
        self.valid_values = valid_values

        message = f"Validation failed for '{field}'"
        if value is not None:
            message += f" (value: {value!r})"
        if reason:
            message += f": {reason}"
        if valid_values:
            message += f". Valid values: {valid_values}"

        super().__init__(message)


class ModelError(GolfSuiteError):
    """Raised when a model operation fails."""

    def __init__(
        self,
        model_name: str,
        operation: str,
        details: str | None = None,
    ):
        self.model_name = model_name
        self.operation = operation
        self.details = details

        message = f"Model '{model_name}' failed during {operation}"
        if details:
            message += f": {details}"

        super().__init__(message)


class SimulationError(GolfSuiteError):
    """Raised when a simulation fails."""

    def __init__(
        self,
        message: str,
        time_step: float | None = None,
        state: dict[str, Any] | None = None,
    ):
        self.time_step = time_step
        self.state = state

        error_msg = f"Simulation error: {message}"
        if time_step is not None:
            error_msg += f" at t={time_step:.4f}s"

        super().__init__(error_msg)


class FileOperationError(GolfSuiteError):
    """Raised when a file operation fails."""

    def __init__(
        self,
        path: Path | str,
        operation: str,
        reason: str | None = None,
    ):
        self.path = Path(path)
        self.operation = operation
        self.reason = reason

        message = f"Failed to {operation} file: {self.path}"
        if reason:
            message += f". Reason: {reason}"

        super().__init__(message)


# Error message factory functions


def format_import_error(
    module_name: str,
    feature: str | None = None,
    install_hint: str | None = None,
) -> str:
    """Format a standardized import error message.

    Args:
        module_name: Name of the module that failed to import.
        feature: Feature that requires this module.
        install_hint: Installation instructions.

    Returns:
        Formatted error message.

    Example:
        raise ImportError(format_import_error("mujoco", "physics simulation"))
    """
    message = f"Module '{module_name}' is not installed"
    if feature:
        message += f" but is required for {feature}"
    message += "."

    if install_hint:
        message += f" Install with: {install_hint}"

    return message


def format_file_error(
    path: Path | str,
    operation: str,
    reason: str | None = None,
) -> str:
    """Format a standardized file error message.

    Args:
        path: Path to the file.
        operation: Operation that failed (read, write, delete, etc.).
        reason: Additional details about the failure.

    Returns:
        Formatted error message.

    Example:
        raise FileNotFoundError(format_file_error("config.json", "read"))
    """
    message = f"Cannot {operation} file: {path}"
    if reason:
        message += f". {reason}"
    return message


def format_validation_error(
    field: str,
    value: Any,
    constraint: str,
) -> str:
    """Format a standardized validation error message.

    Args:
        field: Name of the field that failed validation.
        value: The invalid value.
        constraint: Description of the constraint that was violated.

    Returns:
        Formatted error message.

    Example:
        raise ValueError(format_validation_error("time_step", -0.1, "must be positive"))
    """
    return f"Invalid value for '{field}': {value!r}. {constraint}"


def format_type_error(
    field: str,
    expected_type: type | str,
    actual_type: type | str,
) -> str:
    """Format a standardized type error message.

    Args:
        field: Name of the field with wrong type.
        expected_type: Expected type.
        actual_type: Actual type received.

    Returns:
        Formatted error message.

    Example:
        raise TypeError(format_type_error("position", "ndarray", type(value)))
    """
    expected = expected_type if isinstance(expected_type, str) else expected_type.__name__
    actual = actual_type if isinstance(actual_type, str) else actual_type.__name__
    return f"Invalid type for '{field}': expected {expected}, got {actual}"


def format_range_error(
    field: str,
    value: float | int,
    min_value: float | int | None = None,
    max_value: float | int | None = None,
) -> str:
    """Format a standardized range error message.

    Args:
        field: Name of the field with out-of-range value.
        value: The out-of-range value.
        min_value: Minimum allowed value.
        max_value: Maximum allowed value.

    Returns:
        Formatted error message.

    Example:
        raise ValueError(format_range_error("angle", 400, 0, 360))
    """
    message = f"Value for '{field}' ({value}) is out of range"
    if min_value is not None and max_value is not None:
        message += f". Must be between {min_value} and {max_value}"
    elif min_value is not None:
        message += f". Must be >= {min_value}"
    elif max_value is not None:
        message += f". Must be <= {max_value}"
    return message


def handle_import_error(
    module_name: str,
    feature: str | None = None,
    *,
    raise_error: bool = True,
    log_warning: bool = False,
) -> bool:
    """Handle an ImportError with standardized behavior.

    This is a utility for the common pattern of trying to import an optional
    module and handling the failure gracefully.

    Args:
        module_name: Name of the module that failed to import.
        feature: Feature that requires this module.
        raise_error: If True, raise ImportError. If False, return False.
        log_warning: If True, log a warning instead of/before raising.

    Returns:
        False if raise_error is False, otherwise raises ImportError.

    Example:
        try:
            import mujoco
        except ImportError:
            if not handle_import_error("mujoco", "physics", raise_error=False):
                return None
    """
    message = format_import_error(module_name, feature)

    if log_warning:
        import logging

        logging.warning(message)

    if raise_error:
        raise ImportError(message)

    return False
