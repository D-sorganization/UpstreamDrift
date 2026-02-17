"""Extended tests for src.shared.python.core.error_utils module.

Covers all error classes, format factories, and helper functions not covered
by the existing test_error_utils.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.shared.python.core.error_utils import (
    DataFormatError,
    EnvironmentError,
    EnvironmentValidationError,
    FileNotFoundError_,
    FileNotFoundIOError,
    FileParseError,
    IOError,
    IOUtilsError,
    PhysicalValidationError,
    ResourceError,
    TimeoutError,
    ValidationError,
    format_range_error,
    format_type_error,
    format_validation_error,
    handle_import_error,
)

# ============================================================================
# Tests for format_validation_error factory
# ============================================================================


class TestFormatValidationError:
    """Tests for format_validation_error factory."""

    def test_basic(self) -> None:
        msg = format_validation_error("time_step", -0.1, "must be positive")
        assert "time_step" in msg
        assert "-0.1" in msg
        assert "must be positive" in msg

    def test_string_value(self) -> None:
        msg = format_validation_error("mode", "invalid", "must be 'debug' or 'release'")
        assert "mode" in msg
        assert "invalid" in msg

    def test_none_value(self) -> None:
        msg = format_validation_error("data", None, "cannot be None")
        assert "None" in msg


# ============================================================================
# Tests for format_type_error factory
# ============================================================================


class TestFormatTypeError:
    """Tests for format_type_error factory."""

    def test_with_type_objects(self) -> None:
        msg = format_type_error("position", float, int)
        assert "position" in msg
        assert "float" in msg
        assert "int" in msg

    def test_with_string_types(self) -> None:
        msg = format_type_error("data", "ndarray", "list")
        assert "ndarray" in msg
        assert "list" in msg

    def test_mixed_type_and_string(self) -> None:
        msg = format_type_error("angle", "float", int)
        assert "float" in msg
        assert "int" in msg


# ============================================================================
# Tests for format_range_error factory
# ============================================================================


class TestFormatRangeError:
    """Tests for format_range_error factory."""

    def test_with_both_bounds(self) -> None:
        msg = format_range_error("angle", 400, 0, 360)
        assert "angle" in msg
        assert "400" in msg
        assert "0" in msg
        assert "360" in msg

    def test_with_min_only(self) -> None:
        msg = format_range_error("mass", -1.0, min_value=0)
        assert "mass" in msg
        assert ">=" in msg

    def test_with_max_only(self) -> None:
        msg = format_range_error("ratio", 1.5, max_value=1.0)
        assert "ratio" in msg
        assert "<=" in msg

    def test_no_bounds(self) -> None:
        msg = format_range_error("value", 999)
        assert "out of range" in msg


# ============================================================================
# Tests for handle_import_error function
# ============================================================================


class TestHandleImportErrorFunction:
    """Tests for handle_import_error utility function."""

    def test_raise_error_true(self) -> None:
        with pytest.raises(ImportError, match="mujoco"):
            handle_import_error("mujoco")

    def test_raise_error_false_returns_false(self) -> None:
        result = handle_import_error("nonexistent", raise_error=False)
        assert result is False

    def test_with_feature(self) -> None:
        with pytest.raises(ImportError, match="physics simulation"):
            handle_import_error("mujoco", feature="physics simulation")

    def test_log_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            handle_import_error("missing_module", raise_error=False, log_warning=True)
        assert any("missing_module" in r.message for r in caplog.records)


# ============================================================================
# Tests for EnvironmentError
# ============================================================================


class TestEnvironmentError:
    """Tests for EnvironmentError exception."""

    def test_basic(self) -> None:
        err = EnvironmentError("API_KEY")
        assert "API_KEY" in str(err)
        assert err.var_name == "API_KEY"

    def test_with_reason(self) -> None:
        err = EnvironmentError("DB_HOST", reason="invalid hostname")
        assert "invalid hostname" in str(err)

    def test_with_expected_and_actual(self) -> None:
        err = EnvironmentError("LOG_LEVEL", expected="DEBUG", actual="UNKNOWN")
        assert "DEBUG" in str(err)
        assert "UNKNOWN" in str(err)

    def test_default_reason(self) -> None:
        err = EnvironmentError("MY_VAR")
        assert "not set or invalid" in str(err).lower() or "MY_VAR" in str(err)

    def test_is_configuration_error(self) -> None:
        from src.shared.python.core.error_utils import ConfigurationError

        err = EnvironmentError("X")
        assert isinstance(err, ConfigurationError)

    def test_alias_matches(self) -> None:
        assert EnvironmentValidationError is EnvironmentError


# ============================================================================
# Tests for IOError and subclasses
# ============================================================================


class TestIOError:
    """Tests for IOError base exception."""

    def test_basic(self) -> None:
        err = IOError("Read failed")
        assert "Read failed" in str(err)

    def test_with_path(self) -> None:
        err = IOError("Cannot read", path="/tmp/test.csv")
        assert str(Path("/tmp/test.csv")) in str(err)
        assert isinstance(err.path, Path)

    def test_without_path(self) -> None:
        err = IOError("No path")
        assert err.path is None

    def test_alias_matches(self) -> None:
        assert IOUtilsError is IOError


class TestFileNotFoundIOError:
    """Tests for FileNotFoundIOError."""

    def test_basic(self) -> None:
        err = FileNotFoundIOError("/data/model.xml")
        assert "not found" in str(err).lower()
        assert str(Path("/data/model.xml")) in str(err)

    def test_with_context(self) -> None:
        err = FileNotFoundIOError("/data/model.xml", context="Loading URDF")
        assert "Loading URDF" in str(err)
        assert err.context == "Loading URDF"

    def test_is_io_error(self) -> None:
        err = FileNotFoundIOError("/missing.txt")
        assert isinstance(err, IOError)

    def test_alias_matches(self) -> None:
        assert FileNotFoundError_ is FileNotFoundIOError


class TestFileParseError:
    """Tests for FileParseError."""

    def test_basic(self) -> None:
        err = FileParseError("/data/config.json", "JSON")
        assert "JSON" in str(err)
        assert str(Path("/data/config.json")) in str(err)
        assert err.format_type == "JSON"

    def test_with_details(self) -> None:
        err = FileParseError("/data/model.xml", "XML", details="unexpected EOF")
        assert "unexpected EOF" in str(err)
        assert err.details == "unexpected EOF"

    def test_is_io_error(self) -> None:
        err = FileParseError("/f.csv", "CSV")
        assert isinstance(err, IOError)


# ============================================================================
# Tests for PhysicalValidationError
# ============================================================================


class TestPhysicalValidationError:
    """Tests for PhysicalValidationError."""

    def test_basic(self) -> None:
        err = PhysicalValidationError("mass", value=-5.0)
        assert "mass" in str(err)

    def test_with_constraint(self) -> None:
        err = PhysicalValidationError(
            "velocity",
            value=5e8,
            physical_constraint="exceeds speed of light",
        )
        assert "exceeds speed of light" in str(err)
        assert err.physical_constraint == "exceeds speed of light"

    def test_default_constraint(self) -> None:
        err = PhysicalValidationError("energy")
        assert err.physical_constraint is None

    def test_is_validation_error(self) -> None:
        err = PhysicalValidationError("torque")
        assert isinstance(err, ValidationError)


# ============================================================================
# Tests for DataFormatError
# ============================================================================


class TestDataFormatError:
    """Tests for DataFormatError."""

    def test_basic(self) -> None:
        err = DataFormatError("Invalid data structure")
        assert "Invalid data structure" in str(err)

    def test_with_expected_and_actual(self) -> None:
        err = DataFormatError(
            "Incompatible format",
            expected_format="CSV",
            actual_format="JSON",
        )
        assert "CSV" in str(err)
        assert "JSON" in str(err)
        assert err.expected_format == "CSV"
        assert err.actual_format == "JSON"

    def test_with_expected_only(self) -> None:
        err = DataFormatError("Bad format", expected_format="HDF5")
        assert "HDF5" in str(err)


# ============================================================================
# Tests for TimeoutError
# ============================================================================


class TestTimeoutError:
    """Tests for TimeoutError."""

    def test_basic(self) -> None:
        err = TimeoutError("simulation", 30.0)
        assert "simulation" in str(err)
        assert "30" in str(err)
        assert err.operation == "simulation"
        assert err.timeout_seconds == 30.0

    def test_with_details(self) -> None:
        err = TimeoutError("rendering", 10.0, details="GPU busy")
        assert "GPU busy" in str(err)
        assert err.details == "GPU busy"


# ============================================================================
# Tests for ResourceError
# ============================================================================


class TestResourceError:
    """Tests for ResourceError."""

    def test_basic(self) -> None:
        err = ResourceError("GPU")
        assert "GPU" in str(err)
        assert err.resource_type == "GPU"

    def test_with_reason(self) -> None:
        err = ResourceError("memory", reason="out of memory")
        assert "out of memory" in str(err)
        assert err.reason == "out of memory"


# ============================================================================
# Tests for ValidationError (custom message path)
# ============================================================================


class TestValidationErrorCustomMessage:
    """Tests for ValidationError with custom message parameter."""

    def test_custom_message(self) -> None:
        err = ValidationError("field", message="Custom error message")
        assert str(err) == "Custom error message"

    def test_custom_message_overrides_formatting(self) -> None:
        err = ValidationError(
            "field",
            value=42,
            reason="bad",
            valid_values=[1, 2],
            message="Override",
        )
        assert str(err) == "Override"

    def test_with_expected_only_in_config_error(self) -> None:
        from src.shared.python.core.error_utils import ConfigurationError

        err = ConfigurationError("port", expected=8080)
        assert "8080" in str(err)
