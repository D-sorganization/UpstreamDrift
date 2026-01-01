"""Unified Exception Hierarchy for Golf Modeling Suite.

This module defines the standard exception classes used across the suite.
All custom exceptions should inherit from GolfModelingError.
"""


class GolfModelingError(Exception):
    """Base exception for all golf modeling suite errors."""

    pass


class EngineNotFoundError(GolfModelingError):
    """Raised when a physics engine is not found or not properly installed."""

    pass


class DataFormatError(GolfModelingError):
    """Raised when data format is invalid or unsupported."""

    pass


class ValidationConstraintError(GolfModelingError):
    """Raised when a value violates a physical or logical constraint."""

    pass


class ArrayDimensionError(GolfModelingError):
    """Raised when an array has incorrect dimensions."""

    pass
