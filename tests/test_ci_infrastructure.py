"""Tests for CI infrastructure and dependency management.

These tests verify that:
1. All required dependencies are properly declared in pyproject.toml
2. Core modules can be imported without errors
3. Optional dependency handling works correctly
4. CI-critical paths are functional

This file addresses infrastructure issues identified in CI pipeline failures.
"""

import sys

import pytest


class TestCoreDependencies:
    """Test that core dependencies are installed and importable."""

    def test_numpy_available(self) -> None:
        """Test that numpy is available (always required)."""
        import numpy as np

        assert np.__version__ is not None

    def test_scipy_available(self) -> None:
        """Test that scipy is available (always required)."""
        import scipy

        assert scipy.__version__ is not None

    def test_structlog_available(self) -> None:
        """Test that structlog is available (OBS-001 requirement)."""
        import structlog

        assert structlog.__version__ is not None

    def test_fastapi_available(self) -> None:
        """Test that fastapi is available."""
        import fastapi

        assert fastapi.__version__ is not None

    def test_pydantic_available(self) -> None:
        """Test that pydantic is available."""
        import pydantic

        assert pydantic.__version__ is not None


class TestCoreModuleImports:
    """Test that core modules can be imported without errors."""

    def test_import_core(self) -> None:
        """Test that core module imports successfully."""
        from src.shared.python import core

        assert hasattr(core, "setup_logging")
        assert hasattr(core, "setup_structured_logging")
        assert hasattr(core, "get_logger")

    def test_import_engine_availability(self) -> None:
        """Test that engine_availability module imports successfully."""
        from src.shared.python import engine_availability

        assert hasattr(engine_availability, "MUJOCO_AVAILABLE")
        assert hasattr(engine_availability, "STRUCTLOG_AVAILABLE")
        assert hasattr(engine_availability, "is_engine_available")

    def test_import_exceptions(self) -> None:
        """Test that exceptions module imports successfully."""
        from src.shared.python import exceptions

        assert hasattr(exceptions, "GolfModelingError")
        assert hasattr(exceptions, "EngineNotFoundError")

    def test_import_logging_config(self) -> None:
        """Test that logging_config module imports successfully."""
        from src.shared.python import logging_config

        assert hasattr(logging_config, "get_logger")


class TestStructuredLogging:
    """Test structured logging functionality (OBS-001)."""

    def test_get_logger_returns_bound_logger(self) -> None:
        """Test that get_logger returns a bound logger."""
        from src.shared.python.core import get_logger

        logger = get_logger(__name__)
        assert logger is not None
        # Should have info, warning, error methods
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")
        assert hasattr(logger, "debug")

    def test_setup_structured_logging_idempotent(self) -> None:
        """Test that setup_structured_logging can be called multiple times."""
        from src.shared.python.core import setup_structured_logging

        # Should not raise on repeated calls
        setup_structured_logging()
        setup_structured_logging()
        setup_structured_logging()

    def test_logger_accepts_structured_data(self) -> None:
        """Test that logger accepts keyword arguments for structured data."""
        from src.shared.python.core import get_logger

        logger = get_logger(__name__)
        # Should not raise exceptions
        logger.info("test_event", key1="value1", key2=123)


class TestEngineAvailabilityFlags:
    """Test engine availability detection."""

    def test_structlog_available_flag(self) -> None:
        """Test that structlog availability is properly detected."""
        from src.shared.python.engine_availability import STRUCTLOG_AVAILABLE

        # Since we added structlog as a dependency, it should be True
        assert STRUCTLOG_AVAILABLE is True

    def test_numpy_available_flag(self) -> None:
        """Test that numpy availability is properly detected."""
        from src.shared.python.engine_availability import NUMPY_AVAILABLE

        assert NUMPY_AVAILABLE is True

    def test_scipy_available_flag(self) -> None:
        """Test that scipy availability is properly detected."""
        from src.shared.python.engine_availability import SCIPY_AVAILABLE

        assert SCIPY_AVAILABLE is True

    def test_is_engine_available_function(self) -> None:
        """Test is_engine_available function."""
        from src.shared.python.engine_availability import is_engine_available

        # These should always be true since they're core deps
        assert is_engine_available("numpy") is True
        assert is_engine_available("scipy") is True
        assert is_engine_available("structlog") is True

    def test_get_available_engines_returns_list(self) -> None:
        """Test that get_available_engines returns a list."""
        from src.shared.python.engine_availability import get_available_engines

        available = get_available_engines()
        assert isinstance(available, list)
        assert len(available) > 0
        # Core dependencies should be in the list
        assert "numpy" in available
        assert "scipy" in available


class TestOptionalDependencyHandling:
    """Test graceful handling of optional dependencies."""

    def test_pyqt6_availability_flag_exists(self) -> None:
        """Test that PyQt6 availability flag exists."""
        from src.shared.python.engine_availability import PYQT6_AVAILABLE

        # Flag should exist (value depends on environment)
        assert isinstance(PYQT6_AVAILABLE, bool)

    def test_mujoco_availability_flag_exists(self) -> None:
        """Test that MuJoCo availability flag exists."""
        from src.shared.python.engine_availability import MUJOCO_AVAILABLE

        # Flag should exist (value depends on environment)
        assert isinstance(MUJOCO_AVAILABLE, bool)

    def test_skip_if_unavailable_decorator(self) -> None:
        """Test that skip_if_unavailable creates valid pytest marker."""
        from src.shared.python.engine_availability import skip_if_unavailable

        # Should return a pytest marker, not raise
        marker = skip_if_unavailable("nonexistent_engine_xyz")
        assert marker is not None


class TestCIEnvironmentCompatibility:
    """Tests specific to CI environment compatibility."""

    def test_pytest_importable(self) -> None:
        """Test that pytest is importable (test runner itself)."""
        assert pytest is not None
        assert pytest.__version__ is not None

    @pytest.mark.skipif(
        sys.platform != "linux",
        reason="CI runs on Linux",
    )
    def test_xvfb_compatible_qt_platform(self) -> None:
        """Test that QT_QPA_PLATFORM can be set to offscreen."""
        import os

        # This should not raise in CI with xvfb
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class TestPyprojectTomlConsistency:
    """Test that pyproject.toml is properly configured."""

    def test_pyproject_exists(self) -> None:
        """Test that pyproject.toml exists at repo root."""
        from pathlib import Path

        # Navigate from test file to repo root
        repo_root = Path(__file__).parent.parent
        pyproject = repo_root / "pyproject.toml"
        assert pyproject.exists(), f"pyproject.toml not found at {pyproject}"

    def test_pyproject_has_required_sections(self) -> None:
        """Test that pyproject.toml has required sections."""
        from pathlib import Path

        try:
            import tomllib  # Python 3.11+
        except ImportError:
            import tomli as tomllib  # type: ignore[import-not-found]

        repo_root = Path(__file__).parent.parent
        pyproject = repo_root / "pyproject.toml"

        with open(pyproject, "rb") as f:
            data = tomllib.load(f)

        assert "project" in data
        assert "dependencies" in data["project"]
        assert "optional-dependencies" in data["project"]

    def test_structlog_in_dependencies(self) -> None:
        """Test that structlog is declared in dependencies."""
        from pathlib import Path

        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[import-not-found]

        repo_root = Path(__file__).parent.parent
        pyproject = repo_root / "pyproject.toml"

        with open(pyproject, "rb") as f:
            data = tomllib.load(f)

        deps = data["project"]["dependencies"]
        # Check that structlog is in the dependencies
        assert any(
            "structlog" in dep for dep in deps
        ), "structlog must be in core dependencies"
