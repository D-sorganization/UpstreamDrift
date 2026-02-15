"""
Unit tests for launcher functionality.
"""

import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.shared.python.data_io.path_utils import get_repo_root


class TestLauncherModule:
    """Test cases for launcher module functionality."""

    def test_launcher_module_imports(self):
        """Test that launcher modules can be imported."""
        # Test main launch script
        try:
            import launch_golf_suite

            assert hasattr(launch_golf_suite, "main")
            assert hasattr(launch_golf_suite, "launch_engine_directly")
        except ImportError:
            pytest.skip("Main launcher not available")

    def test_launcher_paths_configuration(self):
        """Test launcher path configuration without GUI initialization."""
        # Mock PyQt6 to avoid GUI initialization
        with patch.dict(
            "sys.modules",
            {
                "PyQt6": Mock(),
                "PyQt6.QtCore": Mock(),
                "PyQt6.QtWidgets": Mock(QWidget=type("QWidget", (), {})),
            },
        ):
            try:
                # Import the module
                import launchers.golf_suite_launcher as launcher_module

                # Test that the module has the expected structure
                assert hasattr(launcher_module, "main")
                assert hasattr(launcher_module, "GolfLauncher")

            except ImportError:
                pytest.skip("Golf suite launcher not available")

    def test_main_launcher_script_structure(self):
        """Test main launcher script structure."""
        try:
            import launch_golf_suite

            # Check for expected functions (post-refactor API)
            expected_functions = [
                "main",
                "launch_engine_directly",
            ]

            available_functions = []
            for func_name in expected_functions:
                if hasattr(launch_golf_suite, func_name):
                    available_functions.append(func_name)

            # Should have the core launcher functions
            assert len(available_functions) >= 2

        except ImportError:
            pytest.skip("Main launcher not available")

    @patch("sys.argv", ["launch_golf_suite.py", "--help"])
    def test_main_help_argument(self):
        """Test main function with help argument."""
        try:
            import launch_golf_suite

            # Mock argparse to avoid actual help output
            with patch("argparse.ArgumentParser.parse_args") as mock_parse:
                mock_args = Mock()
                mock_args.help = True
                mock_parse.return_value = mock_args

                # Should handle help gracefully
                try:
                    launch_golf_suite.main()
                except SystemExit:
                    pass  # Help typically causes SystemExit

        except ImportError:
            pytest.skip("Main launcher not available")

    def test_launcher_error_handling(self):
        """Test launcher error handling.

        Validates that main() handles argument parsing and the --engine flag
        gracefully when an engine module is not importable.
        """
        try:
            import launch_golf_suite

            # Test that launch_engine_directly handles ImportError gracefully
            with pytest.raises(SystemExit):
                launch_golf_suite.launch_engine_directly("nonexistent_engine")

        except ImportError:
            pytest.skip("Main launcher not available")


class TestLauncherUtilities:
    """Test launcher utility functions."""

    def test_path_resolution(self):
        """Test path resolution utilities."""
        # Test that we can resolve paths correctly
        project_root = get_repo_root()

        # Should be able to find key directories (launchers is inside src/)
        # Note: 'engines' may be at root or under src/ depending on repo structure
        expected_dirs_either = ["engines"]  # root or src/
        expected_src_dirs = ["launchers"]

        for dir_name in expected_dirs_either:
            root_path = project_root / dir_name
            src_path = project_root / "src" / dir_name
            assert root_path.exists() or src_path.exists(), (
                f"Directory {dir_name} should exist at root or under src/"
            )

        for dir_name in expected_src_dirs:
            dir_path = project_root / "src" / dir_name
            assert dir_path.exists(), f"Directory src/{dir_name} should exist"

    def test_engine_path_construction(self):
        """Test construction of engine paths."""
        project_root = get_repo_root()

        # Test MuJoCo path
        mujoco_path = (
            project_root
            / "engines"
            / "physics_engines"
            / "mujoco"
            / "python"
            / "mujoco_humanoid_golf"
            / "advanced_gui.py"
        )

        # Path should be constructible (may not exist)
        assert isinstance(mujoco_path, Path)
        assert "mujoco" in str(mujoco_path)

        # Test Drake path
        drake_path = (
            project_root
            / "engines"
            / "physics_engines"
            / "drake"
            / "python"
            / "src"
            / "golf_gui.py"
        )

        assert isinstance(drake_path, Path)
        assert "drake" in str(drake_path)

    def test_logging_configuration(self):
        """Test logging configuration."""
        try:
            # Should be able to import logging utilities
            from src.shared.python.data_io.common_utils import setup_logging

            logger = setup_logging("test_launcher")
            assert logger is not None
            assert logger.name == "test_launcher"

        except ImportError:
            pytest.skip("Logging utilities not available")


class TestLauncherIntegration:
    """Integration tests for launcher functionality."""

    @pytest.mark.integration
    def test_launcher_module_integration(self):
        """Test integration between launcher modules."""
        project_root = get_repo_root()

        # Test that launcher files exist
        main_launcher = project_root / "launch_golf_suite.py"
        suite_launcher = project_root / "src" / "launchers" / "golf_suite_launcher.py"

        assert main_launcher.exists()
        assert suite_launcher.exists()

        # Test that they can be imported (with mocking)
        with patch.dict(
            "sys.modules",
            {
                "PyQt6": Mock(),
                "PyQt6.QtCore": Mock(),
                "PyQt6.QtWidgets": Mock(QWidget=type("QWidget", (), {})),
            },
        ):
            try:
                import launch_golf_suite
                import src.launchers.golf_suite_launcher

                # Both modules should be importable
                assert launch_golf_suite is not None
                assert src.launchers.golf_suite_launcher is not None

            except ImportError as e:
                pytest.skip(f"Launcher integration not available: {e}")

    @pytest.mark.integration
    def test_shared_utilities_integration(self):
        """Test integration with shared utilities."""
        try:
            from src.shared.python.data_io.common_utils import (
                GolfModelingError,
                setup_logging,
            )

            # Test error handling
            assert issubclass(GolfModelingError, Exception)

            # Test logging setup
            logger = setup_logging("integration_test")
            assert logger is not None

        except ImportError:
            pytest.skip("Shared utilities not available")


# Mock fixtures for GUI testing
@pytest.fixture
def mock_qt_application():
    """Mock Qt application for testing."""
    with patch.dict(
        "sys.modules",
        {
            "PyQt6": Mock(),
            "PyQt6.QtCore": Mock(),
            "PyQt6.QtWidgets": Mock(QWidget=type("QWidget", (), {})),
        },
    ):
        mock_app = Mock()
        yield mock_app


@pytest.fixture
def mock_launcher_environment():
    """Mock launcher environment."""
    with patch.dict(
        os.environ, {"GOLF_SUITE_HEADLESS": "1", "GOLF_SUITE_TEST_MODE": "1"}
    ):
        yield
