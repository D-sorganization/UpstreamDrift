"""Tests for src.shared.python.core.error_utils module."""

from __future__ import annotations

from pathlib import Path

from src.shared.python.core.error_utils import (
    ConfigurationError,
    EngineNotAvailableError,
    FileOperationError,
    GolfSuiteError,
    ModelError,
    SimulationError,
    ValidationError,
    format_file_error,
    format_import_error,
)


class TestGolfSuiteError:
    """Tests for GolfSuiteError base exception."""

    def test_is_exception(self) -> None:
        err = GolfSuiteError("test")
        assert isinstance(err, Exception)

    def test_message(self) -> None:
        err = GolfSuiteError("something went wrong")
        assert "something went wrong" in str(err)


class TestEngineNotAvailableError:
    """Tests for EngineNotAvailableError."""

    def test_basic(self) -> None:
        err = EngineNotAvailableError("mujoco")
        assert "mujoco" in str(err).lower()

    def test_with_operation(self) -> None:
        err = EngineNotAvailableError("drake", operation="simulate")
        assert "simulate" in str(err)

    def test_is_golf_suite_error(self) -> None:
        err = EngineNotAvailableError("pinocchio")
        assert isinstance(err, GolfSuiteError)

    def test_known_engine_has_install_hint(self) -> None:
        err = EngineNotAvailableError("mujoco")
        msg = str(err)
        assert "pip install" in msg or "mujoco" in msg.lower()


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_basic(self) -> None:
        err = ConfigurationError("api_key")
        assert "api_key" in str(err)

    def test_with_reason(self) -> None:
        err = ConfigurationError("port", reason="must be integer")
        assert "must be integer" in str(err)

    def test_with_expected_and_actual(self) -> None:
        err = ConfigurationError("mode", expected="debug", actual="release")
        assert "debug" in str(err)
        assert "release" in str(err)


class TestValidationError:
    """Tests for ValidationError."""

    def test_basic(self) -> None:
        err = ValidationError("age")
        assert "age" in str(err)

    def test_with_value(self) -> None:
        err = ValidationError("age", value=-5)
        assert "-5" in str(err)

    def test_with_reason(self) -> None:
        err = ValidationError("age", reason="must be positive")
        assert "must be positive" in str(err)

    def test_with_valid_values(self) -> None:
        err = ValidationError("color", valid_values=["red", "green", "blue"])
        assert "red" in str(err)


class TestModelError:
    """Tests for ModelError."""

    def test_basic(self) -> None:
        err = ModelError("golfer", "load")
        assert "golfer" in str(err)
        assert "load" in str(err)

    def test_with_details(self) -> None:
        err = ModelError("golfer", "simulate", details="NaN detected")
        assert "NaN detected" in str(err)

    def test_attributes(self) -> None:
        err = ModelError("golfer", "step", details="diverged")
        assert err.model_name == "golfer"
        assert err.operation == "step"
        assert err.details == "diverged"


class TestSimulationError:
    """Tests for SimulationError."""

    def test_basic(self) -> None:
        err = SimulationError("diverged")
        assert "diverged" in str(err)

    def test_with_time_step(self) -> None:
        err = SimulationError("exploded", time_step=0.0152)
        assert "0.0152" in str(err)

    def test_attributes(self) -> None:
        state = {"q": [0, 0, 0]}
        err = SimulationError("error", time_step=1.0, state=state)
        assert err.time_step == 1.0
        assert err.state == state


class TestFileOperationError:
    """Tests for FileOperationError."""

    def test_basic(self) -> None:
        err = FileOperationError("/tmp/test.csv", "read")
        assert "read" in str(err)

    def test_with_reason(self) -> None:
        err = FileOperationError("config.json", "write", reason="permission denied")
        assert "permission denied" in str(err)

    def test_path_attribute(self) -> None:
        err = FileOperationError("/foo/bar.txt", "read")
        assert isinstance(err.path, Path)


class TestFormatImportError:
    """Tests for format_import_error factory."""

    def test_basic(self) -> None:
        msg = format_import_error("mujoco")
        assert "mujoco" in msg

    def test_with_feature(self) -> None:
        msg = format_import_error("numpy", feature="matrix operations")
        assert "matrix operations" in msg

    def test_with_install_hint(self) -> None:
        msg = format_import_error("scipy", install_hint="pip install scipy")
        assert "pip install scipy" in msg


class TestFormatFileError:
    """Tests for format_file_error factory."""

    def test_basic(self) -> None:
        msg = format_file_error("test.csv", "read")
        assert "test.csv" in msg
        assert "read" in msg

    def test_with_reason(self) -> None:
        msg = format_file_error("data.json", "write", reason="disk full")
        assert "disk full" in msg
