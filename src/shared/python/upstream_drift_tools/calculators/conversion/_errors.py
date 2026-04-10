from __future__ import annotations


class UnitConversionError(Exception):
    """Base class for conversion errors."""


class UnknownUnitError(UnitConversionError):
    """Raised when a unit is not recognized."""


class IncompatibleUnitsError(UnitConversionError):
    """Raised when attempting to convert between incompatible categories."""


class InvalidValueError(UnitConversionError):
    """Raised when an input value fails validation."""
