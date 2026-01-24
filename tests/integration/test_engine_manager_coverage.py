"""
Coverage tests for EngineManager using mocks.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.engine_manager import (
    EngineManager,
    EngineStatus,
    EngineType,
    GolfModelingError,
)


class TestEngineManagerCoverage:
    """Tests to improve coverage of EngineManager."""

    @pytest.fixture
    def mock_manager(self):
        """Create an EngineManager with mocked probes and paths."""
        with (
            patch("src.shared.python.engine_probes.MuJoCoProbe"),
            patch("src.shared.python.engine_probes.DrakeProbe"),
            patch("src.shared.python.engine_probes.PinocchioProbe"),
            patch("src.shared.python.engine_probes.OpenSimProbe"),
            patch("src.shared.python.engine_probes.PendulumProbe"),
            patch("src.shared.python.engine_probes.MatlabProbe"),
        ):
            manager = EngineManager(suite_root=Path("/tmp/fake_root"))
            # Mock engine paths to exist
            manager.engine_paths = {
                etype: Path(f"/tmp/fake_root/engines/{etype.value}")
                for etype in EngineType
            }
            return manager

    def test_switch_engine_success(self, mock_manager):
        """Test successful engine switching."""
        # Mock status as AVAILABLE
        mock_manager.engine_status[EngineType.MUJOCO] = EngineStatus.AVAILABLE

        # Mock the registry to return a working factory
        from src.shared.python.engine_registry import get_registry

        registry = get_registry()
        mock_factory = MagicMock()
        mock_factory.return_value = MagicMock()  # Mock physics engine

        with patch.object(registry, "get") as mock_get:
            mock_registration = MagicMock()
            mock_registration.factory = mock_factory
            mock_get.return_value = mock_registration

            result = mock_manager.switch_engine(EngineType.MUJOCO)

            assert result is True
            assert mock_manager.current_engine == EngineType.MUJOCO
            assert mock_manager.engine_status[EngineType.MUJOCO] == EngineStatus.LOADED

    def test_switch_engine_unavailable(self, mock_manager):
        """Test switching to an unavailable engine."""
        mock_manager.engine_status[EngineType.MUJOCO] = EngineStatus.UNAVAILABLE

        result = mock_manager.switch_engine(EngineType.MUJOCO)

        assert result is False
        assert mock_manager.current_engine is None

    def test_switch_engine_unknown(self, mock_manager):
        """Test switching to an unknown engine type."""
        result = mock_manager.switch_engine("INVALID_ENGINE")
        assert result is False

    def test_switch_engine_failure(self, mock_manager):
        """Test failure during engine loading."""
        mock_manager.engine_status[EngineType.MUJOCO] = EngineStatus.AVAILABLE

        # Mock the registry to return a factory that raises an error
        from src.shared.python.engine_registry import get_registry

        registry = get_registry()

        def raise_error():
            raise GolfModelingError("Loading failed")

        with patch.object(registry, "get") as mock_get:
            mock_registration = MagicMock()
            mock_registration.factory = raise_error
            mock_get.return_value = mock_registration

            result = mock_manager.switch_engine(EngineType.MUJOCO)

            assert result is False
            assert mock_manager.engine_status[EngineType.MUJOCO] == EngineStatus.ERROR

    def test_validate_engine_configuration(self, mock_manager):
        """Test engine configuration validation."""
        # Mock path existence
        with patch.object(Path, "exists", return_value=True):
            assert mock_manager.validate_engine_configuration(EngineType.MUJOCO) is True

        with patch.object(Path, "exists", return_value=False):
            assert (
                mock_manager.validate_engine_configuration(EngineType.MUJOCO) is False
            )

    def test_validate_engine_configuration_invalid_type(self, mock_manager):
        """Test validation with invalid engine type."""
        assert mock_manager.validate_engine_configuration("INVALID") is False

    def test_get_diagnostic_report(self, mock_manager):
        """Test generating diagnostic report."""
        # Mock probe results
        mock_result = MagicMock()
        mock_result.is_available.return_value = True
        mock_result.engine_name = "MuJoCo"
        mock_result.status = EngineStatus.AVAILABLE
        mock_result.version = "3.2.3"
        mock_result.missing_dependencies = []
        mock_result.diagnostic_message = "Ready"

        mock_manager.probe_results = {EngineType.MUJOCO: mock_result}

        report = mock_manager.get_diagnostic_report()
        assert "Golf Modeling Suite - Engine Readiness Report" in report
        assert "MUJOCO" in report
        assert "✅" in report

    def test_get_engine_info(self, mock_manager):
        """Test getting engine info."""
        mock_manager.current_engine = EngineType.MUJOCO
        mock_manager.engine_status = {EngineType.MUJOCO: EngineStatus.LOADED}

        info = mock_manager.get_engine_info()
        assert info["current_engine"] == "mujoco"
        assert "mujoco" in info["engine_status"]
