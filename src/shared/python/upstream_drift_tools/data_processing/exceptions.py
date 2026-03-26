"""Typed exceptions for the data processing engine.

Each exception maps to a specific failure mode so callers can handle
errors precisely rather than catching broad ``Exception``.
"""

from __future__ import annotations


class DataProcessingError(Exception):
    """Base exception for all data-processing failures."""


class DataNotLoadedError(DataProcessingError):
    """Raised when an operation requires data but none is loaded."""


class ColumnNotFoundError(DataProcessingError):
    """Raised when a referenced column does not exist in the DataFrame."""

    def __init__(self, column: str, available: list[str] | None = None) -> None:
        if not (column is not None):
            raise ValueError("column must be provided")
        if not (column is not None):
            raise ValueError("column must be provided")
        self.column = column
        self.available = available or []
        detail = f"Column '{column}' not found"
        if self.available:
            detail += f". Available: {self.available}"
        super().__init__(detail)


class FileIOError(DataProcessingError):
    """Raised on file read/write failures (permissions, corrupt, etc.)."""


class TransformationError(DataProcessingError):
    """Raised when a column transformation or expression fails."""


class FilterError(DataProcessingError):
    """Raised when a filter/query operation fails."""


class FitError(DataProcessingError):
    """Raised when curve fitting fails (insufficient data, numeric issues)."""


class UnsupportedOperationError(DataProcessingError):
    """Raised when an unknown operation or method is requested."""


__all__ = [
    "DataProcessingError",
    "DataNotLoadedError",
    "ColumnNotFoundError",
    "FileIOError",
    "TransformationError",
    "FilterError",
    "FitError",
    "UnsupportedOperationError",
]
