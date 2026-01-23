"""Unit tests for shared exceptions."""

from src.shared.python.exceptions import (
    ArrayDimensionError,
    DataFormatError,
    EngineNotFoundError,
    GolfModelingError,
    ValidationConstraintError,
)


def test_exception_inheritance() -> None:
    """Verify that all custom exceptions inherit from GolfModelingError."""
    assert issubclass(EngineNotFoundError, GolfModelingError)
    assert issubclass(DataFormatError, GolfModelingError)
    assert issubclass(ValidationConstraintError, GolfModelingError)
    assert issubclass(ArrayDimensionError, GolfModelingError)


def test_exception_messages() -> None:
    """Verify that exceptions can be raised with messages."""
    try:
        raise EngineNotFoundError("Engine not found")
    except EngineNotFoundError as e:
        assert str(e) == "Engine not found"
        assert isinstance(e, GolfModelingError)
