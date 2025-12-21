"""
Unit tests for EngineManager functionality.
"""

# Add the project root to the path for imports
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from shared.python.engine_manager import (  # noqa: E402
    EngineManager,
    EngineStatus,
    EngineType,
)


class TestEngineManager:
    """Test cases for EngineManager functionality."""

    def test_engine_manager_initialization(self):
        """Test that EngineManager initializes correctly."""
        manager = EngineManager()

        # Should have initialized engine status
        assert isinstance(manager.engine_status, dict)
        assert len(manager.engine_status) > 0

        # Should have no current engine initially
        assert manager.current_engine is None

        # Should have engine paths defined
        assert isinstance(manager.engine_paths, dict)
        assert len(manager.engine_paths) == len(EngineType)

    def test_engine_manager_with_custom_root(self):
        """Test EngineManager with custom suite root."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            engines_dir = temp_path / "engines"
            engines_dir.mkdir()

            manager = EngineManager(suite_root=temp_path)

            assert manager.suite_root == temp_path
            assert manager.engines_root == engines_dir

    def test_engine_discovery_with_existing_engines(self):
        """Test engine discovery when engines exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            engines_dir = temp_path / "engines"

            # Create some engine directories
            mujoco_dir = engines_dir / "physics_engines" / "mujoco"
            mujoco_dir.mkdir(parents=True)

            drake_dir = engines_dir / "physics_engines" / "drake"
            drake_dir.mkdir(parents=True)

            manager = EngineManager(suite_root=temp_path)
            available_engines = manager.get_available_engines()

            # Should detect the created engines
            assert EngineType.MUJOCO in available_engines
            assert EngineType.DRAKE in available_engines

            # Should not detect non-existent engines
            assert EngineType.PINOCCHIO not in available_engines

    def test_engine_discovery_with_no_engines(self):
        """Test engine discovery when no engines exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            engines_dir = temp_path / "engines"
            engines_dir.mkdir()

            manager = EngineManager(suite_root=temp_path)
            available_engines = manager.get_available_engines()

            # Should have no available engines
            assert len(available_engines) == 0

            # All engines should be unavailable
            for engine_type in EngineType:
                assert (
                    manager.get_engine_status(engine_type) == EngineStatus.UNAVAILABLE
                )

    def test_switch_engine_unavailable(self):
        """Test switching to unavailable engine."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            engines_dir = temp_path / "engines"
            engines_dir.mkdir()

            manager = EngineManager(suite_root=temp_path)

            # Should fail to switch to unavailable engine
            result = manager.switch_engine(EngineType.MUJOCO)
            assert result is False
            assert manager.current_engine is None

    def test_switch_engine_unknown_type(self):
        """Test switching to unknown engine type."""
        manager = EngineManager()

        # Mock an unknown engine type by removing it from status
        manager.engine_status.pop(EngineType.MUJOCO, None)

        result = manager.switch_engine(EngineType.MUJOCO)
        assert result is False

    def test_get_current_engine(self):
        """Test getting current engine."""
        manager = EngineManager()

        # Initially no current engine
        assert manager.get_current_engine() is None

        # Set current engine
        manager.current_engine = EngineType.MUJOCO
        assert manager.get_current_engine() == EngineType.MUJOCO

    def test_get_engine_status(self):
        """Test getting engine status."""
        manager = EngineManager()

        # Should return status for known engines
        for engine_type in EngineType:
            status = manager.get_engine_status(engine_type)
            assert isinstance(status, EngineStatus)

        # Should return UNAVAILABLE for unknown engines
        manager.engine_status.clear()
        status = manager.get_engine_status(EngineType.MUJOCO)
        assert status == EngineStatus.UNAVAILABLE

    def test_get_engine_info(self):
        """Test getting engine information."""
        manager = EngineManager()

        info = manager.get_engine_info()

        # Should have required keys
        assert "current_engine" in info
        assert "available_engines" in info
        assert "engine_status" in info

        # Should have correct types
        assert isinstance(info["available_engines"], list)
        assert isinstance(info["engine_status"], dict)

        # Current engine should be None initially
        assert info["current_engine"] is None

    def test_validate_engine_configuration_existing(self):
        """Test engine configuration validation for existing engines."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            engines_dir = temp_path / "engines"

            # Create MuJoCo engine with python subdirectory
            mujoco_dir = engines_dir / "physics_engines" / "mujoco"
            mujoco_python_dir = mujoco_dir / "python"
            mujoco_python_dir.mkdir(parents=True)

            manager = EngineManager(suite_root=temp_path)

            # Should validate successfully
            result = manager.validate_engine_configuration(EngineType.MUJOCO)
            assert result is True

    def test_validate_engine_configuration_missing(self):
        """Test engine configuration validation for missing engines."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            engines_dir = temp_path / "engines"
            engines_dir.mkdir()

            manager = EngineManager(suite_root=temp_path)

            # Should fail validation for missing engine
            result = manager.validate_engine_configuration(EngineType.MUJOCO)
            assert result is False

    def test_validate_engine_configuration_unknown_engine(self):
        """Test engine configuration validation for unknown engine."""
        manager = EngineManager()

        # Remove engine from status
        manager.engine_status.pop(EngineType.MUJOCO, None)

        result = manager.validate_engine_configuration(EngineType.MUJOCO)
        assert result is False

    @patch("shared.python.engine_manager.logger")
    def test_engine_loading_error_handling(self, mock_logger):
        """Test error handling during engine loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            engines_dir = temp_path / "engines"

            # Create MuJoCo engine directory
            mujoco_dir = engines_dir / "physics_engines" / "mujoco"
            mujoco_dir.mkdir(parents=True)

            manager = EngineManager(suite_root=temp_path)

            # Mock the loading method to raise an exception
            with patch.object(
                manager, "_load_mujoco_engine", side_effect=Exception("Test error")
            ):
                result = manager.switch_engine(EngineType.MUJOCO)

                assert result is False
                assert (
                    manager.get_engine_status(EngineType.MUJOCO) == EngineStatus.ERROR
                )
                mock_logger.error.assert_called()


class TestEngineTypes:
    """Test cases for EngineType enum."""

    def test_engine_type_values(self):
        """Test that all engine types have correct values."""
        expected_values = {
            EngineType.MUJOCO: "mujoco",
            EngineType.DRAKE: "drake",
            EngineType.PINOCCHIO: "pinocchio",
            EngineType.MATLAB_2D: "matlab_2d",
            EngineType.MATLAB_3D: "matlab_3d",
            EngineType.PENDULUM: "pendulum",
        }

        for engine_type, expected_value in expected_values.items():
            assert engine_type.value == expected_value

    def test_engine_type_completeness(self):
        """Test that we have all expected engine types."""
        engine_values = {e.value for e in EngineType}
        expected_values = {
            "mujoco",
            "drake",
            "pinocchio",
            "matlab_2d",
            "matlab_3d",
            "pendulum",
        }

        assert engine_values == expected_values


class TestEngineStatus:
    """Test cases for EngineStatus enum."""

    def test_engine_status_values(self):
        """Test that all engine statuses have correct values."""
        expected_values = {
            EngineStatus.AVAILABLE: "available",
            EngineStatus.UNAVAILABLE: "unavailable",
            EngineStatus.LOADING: "loading",
            EngineStatus.LOADED: "loaded",
            EngineStatus.ERROR: "error",
        }

        for status, expected_value in expected_values.items():
            assert status.value == expected_value
