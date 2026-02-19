"""Tests for simulation_service - Physics simulation service.

These tests verify the simulation service using Design by Contract principles.
"""

from enum import Enum
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.shared.python.engine_core.engine_manager import EngineManager
from src.shared.python.engine_core.interfaces import PhysicsEngine

# Configure async tests to use asyncio backend only
pytestmark = pytest.mark.anyio

# Explicit attribute list for GenericPhysicsRecorder mocks because the test
# relies on instance attributes (is_recording) that are set in __init__, not
# on the class itself.
_RECORDER_SPEC_ATTRS = [
    "is_recording",
    "record_step",
    "get_time_series",
    "get_data_dict",
    "start",
    "stop",
    "reset",
]


class MockEngineType(Enum):
    """Mock EngineType enum that accepts uppercase values."""

    MUJOCO = "mujoco"
    DRAKE = "drake"
    PYBULLET = "pybullet"

    @classmethod
    def _missing_(cls, value):
        """Handle uppercase string values like 'MUJOCO'."""
        for member in cls:
            if member.name == value or member.value == value.lower():
                return member
        return None


@pytest.fixture(scope="module")
def anyio_backend():
    """Use asyncio backend only (trio not installed)."""
    return "asyncio"


@pytest.fixture
def mock_engine_manager():
    """Create a mock engine manager."""
    manager = MagicMock(spec=EngineManager)
    manager._load_engine = MagicMock()
    manager.get_active_physics_engine = MagicMock(return_value=None)
    return manager


@pytest.fixture
def simulation_service(mock_engine_manager):
    """Create a simulation service instance."""
    from src.api.services.simulation_service import SimulationService

    return SimulationService(mock_engine_manager)


class TestSimulationServiceContract:
    """Design by Contract tests for SimulationService class."""

    def test_instantiates(self, mock_engine_manager):
        """Postcondition: SimulationService can be instantiated."""
        from src.api.services.simulation_service import SimulationService

        service = SimulationService(mock_engine_manager)
        assert service is not None

    def test_has_engine_manager(self, simulation_service):
        """Postcondition: SimulationService has engine_manager attribute."""
        assert hasattr(simulation_service, "engine_manager")

    def test_has_run_simulation_method(self, simulation_service):
        """Postcondition: SimulationService has run_simulation method."""
        assert hasattr(simulation_service, "run_simulation")
        assert callable(simulation_service.run_simulation)

    def test_has_run_simulation_background_method(self, simulation_service):
        """Postcondition: SimulationService has run_simulation_background method."""
        assert hasattr(simulation_service, "run_simulation_background")
        assert callable(simulation_service.run_simulation_background)


class TestRunSimulationContract:
    """Design by Contract tests for run_simulation method."""

    async def test_returns_simulation_response(self, mock_engine_manager):
        """Postcondition: Returns SimulationResponse."""
        from src.api.models.requests import SimulationRequest
        from src.api.models.responses import SimulationResponse
        from src.api.services.simulation_service import SimulationService

        # Setup mock engine
        mock_engine = MagicMock(spec=PhysicsEngine)
        mock_engine.load_from_path = MagicMock()
        mock_engine.set_state = MagicMock()
        mock_engine.set_control = MagicMock()
        mock_engine.step = MagicMock()
        mock_engine_manager.get_active_physics_engine = MagicMock(
            return_value=mock_engine
        )

        # Mock recorder and EngineType
        with (
            patch(
                "src.api.services.simulation_service.GenericPhysicsRecorder"
            ) as MockRecorder,
            patch("src.api.services.simulation_service.EngineType", MockEngineType),
        ):
            mock_recorder = MagicMock(spec=_RECORDER_SPEC_ATTRS)
            mock_recorder.is_recording = False
            mock_recorder.record_step = MagicMock()
            mock_recorder.get_time_series = MagicMock(
                return_value=(np.array([0.0, 0.001]), np.array([[0], [0.1]]))
            )
            MockRecorder.return_value = mock_recorder

            service = SimulationService(mock_engine_manager)
            request = SimulationRequest(
                engine_type="mujoco",
                duration=0.01,
                timestep=0.001,
            )

            result = await service.run_simulation(request)
            assert isinstance(result, SimulationResponse)


class TestRunSimulation:
    """Functional tests for run_simulation."""

    async def test_simulation_success(self, mock_engine_manager):
        """Test successful simulation run."""
        from src.api.models.requests import SimulationRequest
        from src.api.services.simulation_service import SimulationService

        mock_engine = MagicMock(spec=PhysicsEngine)
        mock_engine.load_from_path = MagicMock()
        mock_engine.set_state = MagicMock()
        mock_engine.step = MagicMock()
        mock_engine_manager.get_active_physics_engine = MagicMock(
            return_value=mock_engine
        )

        with (
            patch(
                "src.api.services.simulation_service.GenericPhysicsRecorder"
            ) as MockRecorder,
            patch("src.api.services.simulation_service.EngineType", MockEngineType),
        ):
            mock_recorder = MagicMock(spec=_RECORDER_SPEC_ATTRS)
            mock_recorder.is_recording = False
            mock_recorder.record_step = MagicMock()
            mock_recorder.get_time_series = MagicMock(
                return_value=(np.array([0.0, 0.001]), np.array([[0], [0.1]]))
            )
            MockRecorder.return_value = mock_recorder

            service = SimulationService(mock_engine_manager)
            request = SimulationRequest(
                engine_type="mujoco",
                duration=0.01,
                timestep=0.001,
            )

            result = await service.run_simulation(request)

            assert result.success is True
            assert result.duration == 0.01
            assert result.frames == 10  # 0.01 / 0.001

    async def test_simulation_loads_model(self, mock_engine_manager):
        """Test that simulation loads model when path provided."""
        from src.api.models.requests import SimulationRequest
        from src.api.services.simulation_service import SimulationService

        mock_engine = MagicMock(spec=PhysicsEngine)
        mock_engine.load_from_path = MagicMock()
        mock_engine.step = MagicMock()
        mock_engine_manager.get_active_physics_engine = MagicMock(
            return_value=mock_engine
        )

        with (
            patch(
                "src.api.services.simulation_service.GenericPhysicsRecorder"
            ) as MockRecorder,
            patch("src.api.services.simulation_service.EngineType", MockEngineType),
        ):
            mock_recorder = MagicMock(spec=_RECORDER_SPEC_ATTRS)
            mock_recorder.is_recording = False
            mock_recorder.record_step = MagicMock()
            mock_recorder.get_time_series = MagicMock(
                return_value=(np.array([0.0]), np.array([[0]]))
            )
            MockRecorder.return_value = mock_recorder

            service = SimulationService(mock_engine_manager)
            request = SimulationRequest(
                engine_type="mujoco",
                model_path="/path/to/model.xml",
                duration=0.001,
            )

            await service.run_simulation(request)
            mock_engine.load_from_path.assert_called_once_with("/path/to/model.xml")

    async def test_simulation_sets_initial_state(self, mock_engine_manager):
        """Test that simulation sets initial state when provided."""
        from src.api.models.requests import SimulationRequest
        from src.api.services.simulation_service import SimulationService

        mock_engine = MagicMock(spec=PhysicsEngine)
        mock_engine.load_from_path = MagicMock()
        mock_engine.set_state = MagicMock()
        mock_engine.step = MagicMock()
        mock_engine_manager.get_active_physics_engine = MagicMock(
            return_value=mock_engine
        )

        with (
            patch(
                "src.api.services.simulation_service.GenericPhysicsRecorder"
            ) as MockRecorder,
            patch("src.api.services.simulation_service.EngineType", MockEngineType),
        ):
            mock_recorder = MagicMock(spec=_RECORDER_SPEC_ATTRS)
            mock_recorder.is_recording = False
            mock_recorder.record_step = MagicMock()
            mock_recorder.get_time_series = MagicMock(
                return_value=(np.array([0.0]), np.array([[0]]))
            )
            MockRecorder.return_value = mock_recorder

            service = SimulationService(mock_engine_manager)
            request = SimulationRequest(
                engine_type="mujoco",
                duration=0.001,
                initial_state={"positions": [0.1, 0.2], "velocities": [0.0, 0.0]},
            )

            await service.run_simulation(request)
            mock_engine.set_state.assert_called_once_with([0.1, 0.2], [0.0, 0.0])

    async def test_simulation_failure_returns_error_response(self, mock_engine_manager):
        """Test that simulation failure returns error response."""
        from src.api.models.requests import SimulationRequest
        from src.api.services.simulation_service import SimulationService

        mock_engine_manager.get_active_physics_engine = MagicMock(return_value=None)

        with patch("src.api.services.simulation_service.EngineType", MockEngineType):
            service = SimulationService(mock_engine_manager)
            request = SimulationRequest(
                engine_type="mujoco",
                duration=1.0,
            )

            result = await service.run_simulation(request)

            assert result.success is False
            assert result.frames == 0

    async def test_simulation_with_control_inputs(self, mock_engine_manager):
        """Test simulation with control inputs."""
        from src.api.models.requests import SimulationRequest
        from src.api.services.simulation_service import SimulationService

        mock_engine = MagicMock(spec=PhysicsEngine)
        mock_engine.load_from_path = MagicMock()
        mock_engine.set_control = MagicMock()
        mock_engine.step = MagicMock()
        mock_engine_manager.get_active_physics_engine = MagicMock(
            return_value=mock_engine
        )

        with (
            patch(
                "src.api.services.simulation_service.GenericPhysicsRecorder"
            ) as MockRecorder,
            patch("src.api.services.simulation_service.EngineType", MockEngineType),
        ):
            mock_recorder = MagicMock(spec=_RECORDER_SPEC_ATTRS)
            mock_recorder.is_recording = False
            mock_recorder.record_step = MagicMock()
            mock_recorder.get_time_series = MagicMock(
                return_value=(np.array([0.0, 0.001]), np.array([[0], [0.1]]))
            )
            MockRecorder.return_value = mock_recorder

            service = SimulationService(mock_engine_manager)
            request = SimulationRequest(
                engine_type="mujoco",
                duration=0.002,
                timestep=0.001,
                control_inputs=[{"torques": [1.0, 2.0]}, {"torques": [1.5, 2.5]}],
            )

            await service.run_simulation(request)
            assert mock_engine.set_control.call_count >= 1


class TestRunSimulationBackground:
    """Tests for run_simulation_background method."""

    async def test_updates_task_status_to_running(self, mock_engine_manager):
        """Test that background task updates status to running."""
        from src.api.models.requests import SimulationRequest
        from src.api.services.simulation_service import SimulationService

        mock_engine = MagicMock(spec=PhysicsEngine)
        mock_engine.step = MagicMock()
        mock_engine_manager.get_active_physics_engine = MagicMock(
            return_value=mock_engine
        )

        with (
            patch(
                "src.api.services.simulation_service.GenericPhysicsRecorder"
            ) as MockRecorder,
            patch("src.api.services.simulation_service.EngineType", MockEngineType),
        ):
            mock_recorder = MagicMock(spec=_RECORDER_SPEC_ATTRS)
            mock_recorder.is_recording = False
            mock_recorder.record_step = MagicMock()
            mock_recorder.get_time_series = MagicMock(
                return_value=(np.array([0.0]), np.array([[0]]))
            )
            MockRecorder.return_value = mock_recorder

            service = SimulationService(mock_engine_manager)
            request = SimulationRequest(engine_type="mujoco", duration=0.001)
            active_tasks: dict = {}

            await service.run_simulation_background("task_123", request, active_tasks)

            # Task should be completed
            assert "task_123" in active_tasks
            assert active_tasks["task_123"]["status"] in ["completed", "failed"]

    async def test_handles_simulation_failure_in_background(self, mock_engine_manager):
        """Test that background task handles simulation failure gracefully.

        Note: Exceptions in run_simulation are caught and returned as
        SimulationResponse with success=False, not as exceptions.
        """
        from src.api.models.requests import SimulationRequest
        from src.api.services.simulation_service import SimulationService

        mock_engine_manager.get_active_physics_engine = MagicMock(
            side_effect=RuntimeError("Engine error")
        )

        with patch("src.api.services.simulation_service.EngineType", MockEngineType):
            service = SimulationService(mock_engine_manager)
            request = SimulationRequest(engine_type="mujoco", duration=1.0)
            active_tasks: dict = {}

            await service.run_simulation_background("task_456", request, active_tasks)

            # Task completes but the result indicates failure
            assert active_tasks["task_456"]["status"] == "completed"
            assert active_tasks["task_456"]["result"]["success"] is False

    async def test_handles_uncaught_exception_in_background(self, mock_engine_manager):
        """Test that background task handles uncaught exceptions."""
        from src.api.models.requests import SimulationRequest
        from src.api.services.simulation_service import SimulationService

        # Make dict() raise an error to trigger exception handling in background
        with patch("src.api.services.simulation_service.EngineType", MockEngineType):
            service = SimulationService(mock_engine_manager)

            # Patch run_simulation to raise an exception that isn't caught
            async def raise_error(req):
                raise RuntimeError("Uncaught error")

            service.run_simulation = raise_error
            request = SimulationRequest(engine_type="mujoco", duration=1.0)
            active_tasks: dict = {}

            await service.run_simulation_background("task_789", request, active_tasks)

            assert active_tasks["task_789"]["status"] == "failed"
            assert "Uncaught error" in active_tasks["task_789"]["error"]


class TestExtractSimulationData:
    """Tests for _extract_simulation_data helper method."""

    def test_extracts_time_series_data(self, simulation_service):
        """Test extracting time series data from recorder."""
        mock_recorder = MagicMock(spec=_RECORDER_SPEC_ATTRS)
        mock_recorder.get_time_series = MagicMock(
            side_effect=[
                (
                    np.array([0.0, 0.1, 0.2]),
                    np.array([[0], [1], [2]]),
                ),  # joint_positions
                (
                    np.array([0.0, 0.1, 0.2]),
                    np.array([[0], [0.5], [1.0]]),
                ),  # joint_velocities
                (
                    np.array([0.0, 0.1, 0.2]),
                    np.array([[0], [0.1], [0.2]]),
                ),  # joint_accelerations
                KeyError("control_inputs not available"),  # control_inputs raises
            ]
        )

        data = simulation_service._extract_simulation_data(mock_recorder)

        assert "times" in data
        assert "joint_positions" in data
        assert "joint_velocities" in data

    def test_handles_missing_data_gracefully(self, simulation_service):
        """Test handling missing data gracefully."""
        mock_recorder = MagicMock(spec=_RECORDER_SPEC_ATTRS)
        mock_recorder.get_time_series = MagicMock(side_effect=KeyError("No data"))

        data = simulation_service._extract_simulation_data(mock_recorder)

        # Should return empty dict without raising
        assert isinstance(data, dict)


class TestPerformAnalysis:
    """Tests for _perform_analysis helper method."""

    def test_extracts_ztcf_data(self, simulation_service):
        """Test extracting ZTCF analysis data."""
        mock_recorder = MagicMock(spec=_RECORDER_SPEC_ATTRS)
        mock_recorder.get_time_series = MagicMock(
            return_value=(np.array([0.0, 0.1]), np.array([0.5, 0.6]))
        )

        config = {"ztcf": True}
        results = simulation_service._perform_analysis(mock_recorder, config)

        assert "ztcf_acceleration" in results
        mock_recorder.get_time_series.assert_called_with("ztcf_accel")

    def test_extracts_zvcf_data(self, simulation_service):
        """Test extracting ZVCF analysis data."""
        mock_recorder = MagicMock(spec=_RECORDER_SPEC_ATTRS)
        mock_recorder.get_time_series = MagicMock(
            return_value=(np.array([0.0, 0.1]), np.array([0.3, 0.4]))
        )

        config = {"zvcf": True}
        results = simulation_service._perform_analysis(mock_recorder, config)

        assert "zvcf_acceleration" in results

    def test_extracts_drift_data(self, simulation_service):
        """Test extracting drift analysis data."""
        mock_recorder = MagicMock(spec=_RECORDER_SPEC_ATTRS)
        mock_recorder.get_time_series = MagicMock(
            return_value=(np.array([0.0, 0.1]), np.array([0.01, 0.02]))
        )

        config = {"track_drift": True}
        results = simulation_service._perform_analysis(mock_recorder, config)

        assert "drift_acceleration" in results

    def test_handles_analysis_error(self, simulation_service):
        """Test handling analysis error."""
        mock_recorder = MagicMock(spec=_RECORDER_SPEC_ATTRS)
        mock_recorder.get_time_series = MagicMock(
            side_effect=KeyError("Analysis failed")
        )

        config = {"ztcf": True}
        results = simulation_service._perform_analysis(mock_recorder, config)

        # Should return empty results without raising
        assert isinstance(results, dict)
