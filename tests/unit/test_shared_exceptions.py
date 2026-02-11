"""Unit tests for shared exceptions."""

from src.shared.python.core.exceptions import (
    ArrayDimensionError,
)
from src.shared.python.error_utils import (
    DataFormatError,
    EngineNotAvailableError,
    GolfSuiteError,
    ValidationError,
)


def test_exception_inheritance() -> None:
    """Verify that all custom exceptions inherit from GolfSuiteError."""
    assert issubclass(EngineNotAvailableError, GolfSuiteError)
    assert issubclass(DataFormatError, GolfSuiteError)
    assert issubclass(ValidationError, GolfSuiteError)
    assert issubclass(ArrayDimensionError, GolfSuiteError)


def test_exception_messages() -> None:
    """Verify that exceptions can be raised with messages."""
    try:
        raise EngineNotAvailableError("mujoco")
    except EngineNotAvailableError as e:
        assert "mujoco" in str(e)
        assert "not available" in str(e)
        assert isinstance(e, GolfSuiteError)
