"""Tests for simulation_service - Physics simulation service.

These tests verify the simulation service using Design by Contract principles.
"""

from enum import Enum
from pathlib import Path
from typing import Any, NoReturn
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from src.shared.python.engine_core.engine_manager import EngineManager
from src.shared.python.engine_core.interfaces import PhysicsEngine
import contextlib

# Configure async tests to use asyncio backend only and unit suite marker
pytestmark = [pytest.mark.anyio, pytest.mark.unit]

try:
    import mujoco  # noqa: F401

    _MUJOCO_AVAILABLE = True
except ImportError:
    _MUJOCO_AVAILABLE = False

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
def anyio_backend() -> str:
    """Use asyncio backend only (trio not installed)."""
    return "asyncio"


@pytest.fixture
def mock_engine_manager() -> MagicMock:
    """Create a mock engine manager."""
    manager = MagicMock(spec=EngineManager)
    manager._load_engine = MagicMock()
    manager.get_active_physics_engine = MagicMock(return_value=None)
    return manager


@pytest.fixture(autouse=True)
def _isolate_output_manager(tmp_path, monkeypatch) -> None:
    """Root the service's default OutputManager under tmp_path (issue #8871).

    ``SimulationService`` now persists completed runs via ``OutputManager``
    (see ``_persist_simulation_results``). Without this, every test that
    exercises a successful ``run_simulation`` would write real files into
    the repo's tracked ``output/`` tree. Tests that want to assert on the
    written files should construct their own ``OutputManager(base_path=...)``
    and pass it explicitly instead of relying on this default.
    """
    from src.shared.python.data_io.output_manager import (
        OutputManager as RealOutputManager,
    )

    monkeypatch.setattr(
        "src.api.services.simulation_service.OutputManager",
        lambda *_a, **_k: RealOutputManager(base_path=tmp_path),
    )


@pytest.fixture
def simulation_service(mock_engine_manager: MagicMock) -> Any:
    """Create a simulation service instance."""
    from src.api.services.simulation_service import SimulationService

    return SimulationService(mock_engine_manager)


class TestSimulationServiceContract:
    """Design by Contract tests for SimulationService class."""

    def test_simulation_service_instantiates(self, mock_engine_manager) -> None:
        """Postcondition: SimulationService can be instantiated."""
        from src.api.services.simulation_service import SimulationService

        service = SimulationService(mock_engine_manager)
        assert service is not None

    def test_simulation_service_has_engine_manager(self, simulation_service) -> None:
        """Postcondition: SimulationService has engine_manager attribute."""
        assert hasattr(simulation_service, "engine_manager")

    def test_has_run_simulation_method(self, simulation_service) -> None:
        """Postcondition: SimulationService has run_simulation method."""
        assert hasattr(simulation_service, "run_simulation")
        assert callable(simulation_service.run_simulation)

    def test_has_run_simulation_background_method(self, simulation_service) -> None:
        """Postcondition: SimulationService has run_simulation_background method."""
        assert hasattr(simulation_service, "run_simulation_background")
        assert callable(simulation_service.run_simulation_background)


class TestRunSimulationContract:
    """Design by Contract tests for run_simulation method."""

    async def test_returns_simulation_response(self, mock_engine_manager) -> None:
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

    async def test_simulation_success(self, mock_engine_manager) -> None:
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
            assert result.frames == 11  # 0.01 / 0.001 steps + 1 initial frame at t=0

    async def test_simulation_loads_model(self, mock_engine_manager) -> None:
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

    async def test_simulation_sets_initial_state(self, mock_engine_manager) -> None:
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

    async def test_simulation_failure_returns_error_response(
        self, mock_engine_manager
    ) -> None:
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

    async def test_simulation_with_control_inputs(self, mock_engine_manager) -> None:
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


@pytest.mark.unit
class TestExportPathsPersistence:
    """export_paths must reflect what OutputManager actually wrote (issue #8871)."""

    async def test_success_populates_export_paths_with_existing_files(
        self, mock_engine_manager, tmp_path
    ) -> None:
        """On success, export_paths is non-empty and every path exists on disk."""
        from src.api.models.requests import SimulationRequest
        from src.api.services.simulation_service import SimulationService
        from src.shared.python.data_io.output_manager import OutputManager

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
                return_value=(np.array([0.0, 0.001]), np.array([[0], [0.1]]))
            )
            MockRecorder.return_value = mock_recorder

            output_manager = OutputManager(base_path=tmp_path)
            service = SimulationService(mock_engine_manager, output_manager)
            request = SimulationRequest(
                engine_type="mujoco",
                duration=0.01,
                timestep=0.001,
            )

            result = await service.run_simulation(request)

            assert result.success is True
            assert result.export_paths
            for path_str in result.export_paths:
                path = Path(path_str)
                assert path.is_file(), f"export path does not exist: {path}"
                assert tmp_path in path.parents

    async def test_failure_leaves_export_paths_empty(
        self, mock_engine_manager, tmp_path
    ) -> None:
        """On failure, nothing was persisted, so export_paths stays empty."""
        from src.api.models.requests import SimulationRequest
        from src.api.services.simulation_service import SimulationService
        from src.shared.python.data_io.output_manager import OutputManager

        mock_engine_manager.get_active_physics_engine = MagicMock(return_value=None)

        with patch("src.api.services.simulation_service.EngineType", MockEngineType):
            output_manager = OutputManager(base_path=tmp_path)
            service = SimulationService(mock_engine_manager, output_manager)
            request = SimulationRequest(engine_type="mujoco", duration=1.0)

            result = await service.run_simulation(request)

            assert result.success is False
            assert not result.export_paths
            # Nothing should have been written under the isolated output tree.
            assert list(tmp_path.rglob("*.json")) == []


class MockTaskManager:
    # ``TaskManager.set`` is synchronous (#4843 compatibility contract);
    # ``run_simulation_background`` calls it without awaiting, so the mock
    # mirrors that synchronous contract.
    def __init__(self):
        self.tasks = {}

    def exists(self, task_id: str) -> bool:
        return task_id in self.tasks

    def get(self, task_id: str) -> dict | None:
        return self.tasks.get(task_id)

    def set(self, task_id: str, data: dict):
        self.tasks[task_id] = data


class TestRunSimulationBackground:
    """Tests for run_simulation_background method."""

    async def test_updates_task_status_to_running(self, mock_engine_manager) -> None:
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
            active_tasks = MockTaskManager()

            await service.run_simulation_background("task_123", request, active_tasks)

            # Task should be completed
            assert "task_123" in active_tasks.tasks
            assert active_tasks.tasks["task_123"]["status"] in ["completed", "failed"]

    async def test_handles_simulation_failure_in_background(
        self, mock_engine_manager
    ) -> None:
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
            active_tasks = MockTaskManager()

            with contextlib.suppress(Exception):
                await service.run_simulation_background(
                    "task_456", request, active_tasks
                )

            # Task completes but the result indicates failure
            assert active_tasks.tasks["task_456"]["status"] == "failed"
            assert active_tasks.tasks["task_456"]["result"]["success"] is False

    async def test_handles_uncaught_exception_in_background(
        self, mock_engine_manager
    ) -> None:
        """Test that background task handles uncaught exceptions."""
        from src.api.models.requests import SimulationRequest
        from src.api.services.simulation_service import SimulationService

        # Make dict() raise an error to trigger exception handling in background
        with patch("src.api.services.simulation_service.EngineType", MockEngineType):
            service = SimulationService(mock_engine_manager)

            # Patch run_simulation to raise an exception that isn't caught
            async def raise_error(req) -> NoReturn:
                raise RuntimeError("Uncaught error")

            service.run_simulation = raise_error
            request = SimulationRequest(engine_type="mujoco", duration=1.0)
            active_tasks = MockTaskManager()

            with contextlib.suppress(Exception):
                await service.run_simulation_background(
                    "task_789", request, active_tasks
                )

            assert active_tasks.tasks["task_789"]["status"] == "failed"
            assert "Uncaught error" in active_tasks.tasks["task_789"]["error"]


class TestExtractSimulationData:
    """Tests for _extract_simulation_data helper method."""

    def test_extracts_time_series_data(self, simulation_service) -> None:
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

    def test_handles_missing_data_gracefully(self, simulation_service) -> None:
        """Test handling missing data gracefully."""
        mock_recorder = MagicMock(spec=_RECORDER_SPEC_ATTRS)
        mock_recorder.get_time_series = MagicMock(side_effect=KeyError("No data"))

        data = simulation_service._extract_simulation_data(mock_recorder)

        # Should return empty dict without raising
        assert isinstance(data, dict)


class TestPerformAnalysis:
    """Tests for _perform_analysis helper method."""

    def test_extracts_ztcf_data(self, simulation_service) -> None:
        """Test extracting ZTCF analysis data."""
        mock_recorder = MagicMock(spec=_RECORDER_SPEC_ATTRS)
        mock_recorder.get_time_series = MagicMock(
            return_value=(np.array([0.0, 0.1]), np.array([0.5, 0.6]))
        )

        config = {"ztcf": True}
        results = simulation_service._perform_analysis(mock_recorder, config)

        assert "ztcf_acceleration" in results
        mock_recorder.get_time_series.assert_called_with("ztcf_accel")

    def test_extracts_zvcf_data(self, simulation_service) -> None:
        """Test extracting ZVCF analysis data."""
        mock_recorder = MagicMock(spec=_RECORDER_SPEC_ATTRS)
        mock_recorder.get_time_series = MagicMock(
            return_value=(np.array([0.0, 0.1]), np.array([0.3, 0.4]))
        )

        config = {"zvcf": True}
        results = simulation_service._perform_analysis(mock_recorder, config)

        assert "zvcf_acceleration" in results

    def test_extracts_drift_data(self, simulation_service) -> None:
        """Test extracting drift analysis data."""
        mock_recorder = MagicMock(spec=_RECORDER_SPEC_ATTRS)
        mock_recorder.get_time_series = MagicMock(
            return_value=(np.array([0.0, 0.1]), np.array([0.01, 0.02]))
        )

        config = {"track_drift": True}
        results = simulation_service._perform_analysis(mock_recorder, config)

        assert "drift_acceleration" in results

    def test_handles_analysis_error(self, simulation_service) -> None:
        """Test handling analysis error."""
        mock_recorder = MagicMock(spec=_RECORDER_SPEC_ATTRS)
        mock_recorder.get_time_series = MagicMock(
            side_effect=KeyError("Analysis failed")
        )

        config = {"ztcf": True}
        results = simulation_service._perform_analysis(mock_recorder, config)

        # Should return empty results without raising
        assert isinstance(results, dict)


# ──────────────────────────────────────────────────────────────
#  SimulationStats — authoritative runtime state (issue #2469)
# ──────────────────────────────────────────────────────────────


class TestSimulationStats:
    """SimulationStats dataclass is the single source of truth for runtime state."""

    def test_can_import(self) -> None:
        from src.api.services.simulation_service import SimulationStats  # noqa: F401

    def test_default_frame_count_is_zero(self) -> None:
        from src.api.services.simulation_service import SimulationStats

        s = SimulationStats()
        assert s.frame_count == 0

    def test_default_is_not_recording(self) -> None:
        from src.api.services.simulation_service import SimulationStats

        s = SimulationStats()
        assert s.is_recording is False

    def test_default_recorded_frames_is_empty(self) -> None:
        from src.api.services.simulation_service import SimulationStats

        s = SimulationStats()
        assert s.recorded_frames == []

    def test_default_speed_factor_is_one(self) -> None:
        from src.api.services.simulation_service import SimulationStats

        s = SimulationStats()
        assert s.speed_factor == 1.0

    def test_start_time_is_float(self) -> None:
        from src.api.services.simulation_service import SimulationStats

        s = SimulationStats()
        assert isinstance(s.start_time, float)


class TestSimulationServiceStatsTracking:
    """SimulationService owns stats and wires them through the sim loop."""

    def test_service_exposes_stats_property(self, mock_engine_manager) -> None:
        from src.api.services.simulation_service import (
            SimulationService,
            SimulationStats,
        )

        service = SimulationService(mock_engine_manager)
        assert isinstance(service.stats, SimulationStats)

    def test_start_recording_sets_flag(self, mock_engine_manager) -> None:
        from src.api.services.simulation_service import SimulationService

        service = SimulationService(mock_engine_manager)
        service.start_recording()
        assert service.stats.is_recording is True

    def test_start_recording_clears_frames(self, mock_engine_manager) -> None:
        from src.api.services.simulation_service import SimulationService

        service = SimulationService(mock_engine_manager)
        # Manually place a frame to verify it gets cleared
        service.stats.recorded_frames.append({"t": 0.0})
        service.start_recording()
        assert service.stats.recorded_frames == []

    def test_stop_recording_clears_flag(self, mock_engine_manager) -> None:
        from src.api.services.simulation_service import SimulationService

        service = SimulationService(mock_engine_manager)
        service.start_recording()
        service.stop_recording()
        assert service.stats.is_recording is False

    def test_set_speed_factor_updates_stats(self, mock_engine_manager) -> None:
        from src.api.services.simulation_service import SimulationService

        service = SimulationService(mock_engine_manager)
        service.set_speed_factor(2.5)
        assert service.stats.speed_factor == 2.5

    async def test_frame_count_reflects_simulation_steps(
        self, mock_engine_manager
    ) -> None:
        """After run_simulation, stats.frame_count equals the steps executed."""
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
                return_value=(np.array([0.0, 0.001]), np.array([[0], [0.1]]))
            )
            MockRecorder.return_value = mock_recorder

            service = SimulationService(mock_engine_manager)
            request = SimulationRequest(
                engine_type="mujoco",
                duration=0.01,
                timestep=0.001,
            )

            await service.run_simulation(request)
            # 0.01 / 0.001 = 10 steps
            assert service.stats.frame_count == 10

    async def test_start_time_reset_on_run_simulation(
        self, mock_engine_manager
    ) -> None:
        """run_simulation resets start_time so wall_time is accurate."""
        import time

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
            before = time.time()
            request = SimulationRequest(engine_type="mujoco", duration=0.001)
            await service.run_simulation(request)
            after = time.time()

            assert before <= service.stats.start_time <= after


class TestSimulationRecorderLifecycleAndValidation:
    """TDD tests for issue #11142: Real recorder lifecycle, control preservation,

    data validation, and buffer exhaustion visibility.
    """

    async def test_real_recorder_lifecycle_with_fake_engine(self) -> None:
        """Deterministic fake engine + real recorder produces initial & stepped samples,

        nonzero commanded controls, aligned lengths, and stopped lifecycle.
        """
        from src.shared.python.engine_core.mock_engine import MockPhysicsEngine
        from src.api.models.requests import SimulationRequest
        from src.api.services.simulation_service import SimulationService

        fake_engine = MockPhysicsEngine(num_joints=2)
        fake_engine.load_model("test_model")
        mock_engine_manager = MagicMock(spec=EngineManager)
        mock_engine_manager._load_engine = MagicMock()
        mock_engine_manager.get_active_physics_engine = MagicMock(
            return_value=fake_engine
        )

        with patch("src.api.services.simulation_service.EngineType", MockEngineType):
            service = SimulationService(mock_engine_manager)
            request = SimulationRequest(
                engine_type="mujoco",
                duration=0.003,
                timestep=0.001,
                control_inputs=[
                    {"torques": [1.0, 2.0]},
                    {"torques": [3.0, 4.0]},
                    {"torques": [5.0, 6.0]},
                ],
            )

            result = await service.run_simulation(request)

            assert result.success is True
            # 1 initial sample (t=0) + 3 steps = 4 frames
            assert result.frames == 4
            assert "times" in result.data
            assert len(result.data["times"]) == 4
            np.testing.assert_allclose(result.data["times"], [0.0, 0.001, 0.002, 0.003])

            # Aligned state channels
            assert len(result.data["joint_positions"]) == 4
            assert len(result.data["joint_velocities"]) == 4
            assert len(result.data["joint_accelerations"]) == 4

            # Commanded controls preserved
            assert "control_inputs" in result.data
            assert len(result.data["control_inputs"]) == 4
            np.testing.assert_allclose(result.data["control_inputs"][1], [1.0, 2.0])
            np.testing.assert_allclose(result.data["control_inputs"][2], [3.0, 4.0])
            np.testing.assert_allclose(result.data["control_inputs"][3], [5.0, 6.0])

            # Recorder lifecycle
            recorder = service.active_recorder
            assert recorder is not None
            assert recorder.current_idx == 4
            assert recorder.is_recording is False

    async def test_real_recorder_both_rest_paths(self, tmp_path) -> None:
        """Both sync (/simulate) and async (/simulate/async) REST paths return aligned results."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from src.api.routes.simulation import router as simulation_router
        from src.api.dependencies import get_simulation_service, get_task_manager
        from src.shared.python.engine_core.mock_engine import MockPhysicsEngine
        from src.api.services.simulation_service import SimulationService
        from src.shared.python.data_io.output_manager import OutputManager

        fake_engine = MockPhysicsEngine(num_joints=2)
        fake_engine.load_model("test_model")
        mock_engine_manager = MagicMock(spec=EngineManager)
        mock_engine_manager._load_engine = MagicMock()
        mock_engine_manager.get_active_physics_engine = MagicMock(
            return_value=fake_engine
        )

        with patch("src.api.services.simulation_service.EngineType", MockEngineType):
            out_mgr = OutputManager(base_path=tmp_path)
            service = SimulationService(mock_engine_manager, output_manager=out_mgr)
            task_mgr = MockTaskManager()

            app = FastAPI()
            app.include_router(simulation_router)
            app.dependency_overrides[get_simulation_service] = lambda: service
            app.dependency_overrides[get_task_manager] = lambda: task_mgr

            client = TestClient(app)
            payload = {
                "engine_type": "mujoco",
                "duration": 0.002,
                "timestep": 0.001,
                "control_inputs": [{"torques": [1.0, 2.0]}, {"torques": [3.0, 4.0]}],
            }

            # 1. Sync path: POST /simulate
            res_sync = client.post("/simulate", json=payload)
            assert res_sync.status_code == 200
            data_sync = res_sync.json()
            assert data_sync["success"] is True
            assert data_sync["frames"] == 3
            assert len(data_sync["data"]["times"]) == 3
            assert len(data_sync["data"]["control_inputs"]) == 3
            assert data_sync["data"]["control_inputs"][1] == [1.0, 2.0]

            # 2. Async path: POST /simulate/async -> GET /simulate/status/{id}
            fake_engine.reset()
            res_async = client.post("/simulate/async", json=payload)
            assert res_async.status_code == 200
            task_id = res_async.json()["task_id"]

            res_status = client.get(f"/simulate/status/{task_id}")
            assert res_status.status_code == 200
            status_data = res_status.json()
            assert status_data["status"] == "completed"
            async_result = status_data["result"]
            assert async_result["success"] is True
            assert async_result["frames"] == 3
            assert len(async_result["data"]["times"]) == 3
            assert len(async_result["data"]["control_inputs"]) == 3

    async def test_reject_zero_recorded_samples(self) -> None:
        """A simulation producing zero recorded samples cannot return successful status."""
        from src.api.models.requests import SimulationRequest
        from src.api.services.simulation_service import SimulationService

        mock_engine = MagicMock(spec=PhysicsEngine)
        mock_engine_manager = MagicMock(spec=EngineManager)
        mock_engine_manager.get_active_physics_engine = MagicMock(
            return_value=mock_engine
        )

        with (
            patch(
                "src.api.services.simulation_service.GenericPhysicsRecorder"
            ) as MockRecorder,
            patch("src.api.services.simulation_service.EngineType", MockEngineType),
        ):
            mock_rec = MagicMock(spec=_RECORDER_SPEC_ATTRS)
            mock_rec.is_recording = False
            mock_rec.current_idx = 0
            # get_time_series returns empty arrays
            mock_rec.get_time_series.return_value = (np.array([]), np.array([]))
            MockRecorder.return_value = mock_rec

            service = SimulationService(mock_engine_manager)
            request = SimulationRequest(
                engine_type="mujoco",
                duration=0.002,
                timestep=0.001,
            )

            result = await service.run_simulation(request)
            assert result.success is False
            assert result.frames == 0
            assert service.stats.last_run is not None
            assert service.stats.last_run["status"] == "failed"

    async def test_buffer_exhaustion_is_explicit(self) -> None:
        """Buffer exhaustion surfaces explicitly and counts cannot silently disagree."""
        from src.shared.python.engine_core.mock_engine import MockPhysicsEngine
        from src.shared.python.dashboard.recorder import GenericPhysicsRecorder
        from src.api.models.requests import SimulationRequest
        from src.api.services.simulation_service import SimulationService

        fake_engine = MockPhysicsEngine(num_joints=2)
        fake_engine.load_model("test_model")
        mock_engine_manager = MagicMock(spec=EngineManager)
        mock_engine_manager._load_engine = MagicMock()
        mock_engine_manager.get_active_physics_engine = MagicMock(
            return_value=fake_engine
        )

        with (
            patch("src.api.services.simulation_service.EngineType", MockEngineType),
            patch(
                "src.api.services.simulation_service.GenericPhysicsRecorder",
                lambda eng, **kwargs: GenericPhysicsRecorder(
                    eng, max_samples=2, initial_capacity=2
                ),
            ),
        ):
            service = SimulationService(mock_engine_manager)
            request = SimulationRequest(
                engine_type="mujoco",
                duration=0.005,  # 5 steps -> 6 requested samples, exceeds max_samples=2
                timestep=0.001,
            )

            result = await service.run_simulation(request)
            assert result.success is False
            assert service.stats.last_run is not None
            assert service.stats.last_run["status"] == "failed"
            error_msg = service.stats.last_run["error"]
            assert error_msg is not None
            assert "capacity exhausted" in error_msg.lower()

    @pytest.mark.smoke
    @pytest.mark.requires_mujoco
    @pytest.mark.skipif(not _MUJOCO_AVAILABLE, reason="MuJoCo runtime is not installed")
    async def test_mujoco_smoke_journey_run_record_analyze_export_reload(
        self, tmp_path: Path
    ) -> None:
        """One installed MuJoCo smoke journey runs -> records -> analyzes -> exports -> reloads."""
        import json
        from src.api.models.requests import SimulationRequest
        from src.api.services.simulation_service import SimulationService
        from src.shared.python.data_io.output_manager import OutputManager

        engine_manager = EngineManager()
        output_manager = OutputManager(base_path=tmp_path / "output")
        service = SimulationService(engine_manager, output_manager=output_manager)

        request = SimulationRequest(
            engine_type="mujoco",
            duration=0.01,
            timestep=0.001,
            analysis_config={"zvcf": True, "ztcf": True},
        )

        # 1. Run & Record
        response = await service.run_simulation(request)
        assert response.success is True
        assert response.duration == 0.01
        # 10 steps + 1 initial sample at t=0 = 11 frames
        assert response.frames == 11
        assert response.data is not None

        times = response.data["times"]
        assert len(times) == 11
        assert np.isclose(times[0], 0.0)
        assert np.isclose(times[-1], 0.01)

        for channel in ("joint_positions", "joint_velocities", "joint_accelerations"):
            assert channel in response.data
            assert len(response.data[channel]) == 11

        # 2. Analyze
        assert response.analysis_results is not None
        assert "zvcf_acceleration" in response.analysis_results
        assert "ztcf_acceleration" in response.analysis_results

        # 3. Export
        assert response.export_paths is not None
        assert len(response.export_paths) == 1
        export_file = Path(response.export_paths[0])
        assert export_file.is_file()

        # 4. Reload and verify matching run identity and values
        reloaded = json.loads(export_file.read_text(encoding="utf-8"))
        assert reloaded.get("engine") == "mujoco"
        assert reloaded["metadata"].get("duration") == response.duration
        assert reloaded["metadata"].get("frames") == response.frames

        # Verify values match between original response and reloaded export
        assert reloaded["results"]["times"] == response.data["times"]
        assert (
            reloaded["results"]["joint_positions"] == response.data["joint_positions"]
        )
        assert (
            reloaded["results"]["joint_velocities"] == response.data["joint_velocities"]
        )
        assert (
            reloaded["results"]["joint_accelerations"]
            == response.data["joint_accelerations"]
        )
        assert (
            reloaded["metadata"]["analysis_results"]["zvcf_acceleration"]
            == response.analysis_results["zvcf_acceleration"]
        )
        assert (
            reloaded["metadata"]["analysis_results"]["ztcf_acceleration"]
            == response.analysis_results["ztcf_acceleration"]
        )
