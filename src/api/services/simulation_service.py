"""Simulation service for Golf Modeling Suite API."""

import time
from dataclasses import dataclass, field
from typing import Any

import anyio.to_thread

from src.shared.python.core.contracts import precondition
from src.shared.python.core.error_utils import (
    EngineLaunchError,
    GolfSuiteError,
    ModelLoadError,
)
from src.shared.python.dashboard.recorder import GenericPhysicsRecorder
from src.shared.python.engine_core.engine_manager import EngineManager
from src.shared.python.engine_core.engine_registry import EngineType
from src.shared.python.logging_pkg.logging_config import get_logger

from ..models.requests import SimulationRequest
from ..models.responses import SimulationResponse

logger = get_logger(__name__)

_DEFAULT_SPEED_FACTOR = 1.0


@dataclass
class SimulationStats:
    """Authoritative runtime state for an active simulation session.

    Owned by SimulationService and updated by the real simulation loop.
    Routes read from this instead of engine_manager private fields.
    """

    start_time: float = field(default_factory=time.time)
    frame_count: int = 0
    speed_factor: float = _DEFAULT_SPEED_FACTOR
    is_recording: bool = False
    recorded_frames: list[Any] = field(default_factory=list)


class SimulationService:
    """Service for managing physics simulations."""

    def __init__(self, engine_manager: EngineManager) -> None:
        """Initialize simulation service.

        Args:
            engine_manager: Engine manager instance
        """
        self.engine_manager = engine_manager
        self._stats = SimulationStats()

    @property
    def stats(self) -> SimulationStats:
        """Return the authoritative runtime stats for this session."""
        return self._stats

    def start_recording(self) -> None:
        """Begin recording trajectory frames. Clears any previously recorded data."""
        self._stats.is_recording = True
        self._stats.recorded_frames = []

    def stop_recording(self) -> None:
        """Stop recording trajectory frames."""
        self._stats.is_recording = False

    def set_speed_factor(self, value: float) -> None:
        """Set simulation speed multiplier.

        Args:
            value: Speed multiplier (>0).
        """
        self._stats.speed_factor = value

    @precondition(
        lambda self, request: request is not None,
        "Simulation request must not be None",
    )
    @precondition(
        lambda self, request: request.duration > 0,
        "Simulation duration must be positive",
    )
    @precondition(
        lambda self, request: (
            request.engine_type is not None and len(request.engine_type) > 0
        ),
        "Engine type must be specified",
    )
    def _prepare_engine(self, request: SimulationRequest) -> Any:
        """Load and configure the physics engine for simulation.

        Args:
            request: Simulation request with engine type and model path.

        Returns:
            Configured engine instance.

        Raises:
            EngineLaunchError: If engine fails to load.
            ModelLoadError: If model file fails to load.
        """
        engine_type = EngineType(request.engine_type.lower())
        self.engine_manager._load_engine(engine_type)

        engine = self.engine_manager.get_active_physics_engine()
        if not engine:
            raise EngineLaunchError(
                request.engine_type,
                reason="engine loaded but no active engine returned",
            )

        if request.model_path:
            try:
                engine.load_from_path(request.model_path)
            except (FileNotFoundError, OSError, ValueError) as e:
                raise ModelLoadError(str(request.model_path), reason=str(e)) from e

        if request.initial_state:
            positions = request.initial_state.get("positions", [])
            velocities = request.initial_state.get("velocities", [])
            if positions and velocities:
                engine.set_state(positions, velocities)

        return engine

    def _execute_simulation_loop(
        self,
        engine: Any,
        recorder: GenericPhysicsRecorder,
        request: SimulationRequest,
        timestep: float,
        steps: int,
    ) -> None:
        """Execute the main simulation stepping loop.

        Args:
            engine: Physics engine instance.
            recorder: Recording object for simulation data.
            request: Simulation request with control inputs.
            timestep: Time step per simulation step.
            steps: Total number of steps to execute.
        """
        if not (recorder is not None):
            raise ValueError("recorder must be provided")
        if not (engine is not None):
            raise ValueError("engine must be provided")
        if not recorder.is_recording:
            recorder.record_step()

        for step in range(steps):
            if request.control_inputs and step < len(request.control_inputs):
                control = request.control_inputs[step]
                if "torques" in control:
                    engine.set_control(control["torques"])
            engine.step(timestep)
            recorder.record_step()
            self._stats.frame_count += 1

    def _run_simulation_sync(self, request: SimulationRequest) -> SimulationResponse:
        """Run the full CPU-bound simulation pipeline synchronously.

        This performs engine preparation, the stepping loop, data extraction,
        and analysis. It is intentionally blocking and must be invoked off the
        event loop (see :meth:`run_simulation`) so the FastAPI worker is not
        frozen for the duration of the simulation (issue #6988).
        """
        self._stats.start_time = time.time()
        self._stats.frame_count = 0
        engine = self._prepare_engine(request)
        recorder = GenericPhysicsRecorder(engine)

        if request.analysis_config:
            recorder.set_analysis_config(request.analysis_config)

        timestep = request.timestep or 0.001
        if timestep <= 0:
            raise ValueError(f"Timestep must be positive, got {timestep}")
        if timestep > request.duration:
            raise ValueError(
                f"Timestep ({timestep}) must not exceed duration ({request.duration})"
            )
        steps = int(request.duration / timestep)

        self._execute_simulation_loop(engine, recorder, request, timestep, steps)

        simulation_data = self._extract_simulation_data(recorder)
        analysis_results = None
        if request.analysis_config:
            analysis_results = self._perform_analysis(recorder, request.analysis_config)

        return SimulationResponse(
            success=True,
            duration=request.duration,
            frames=steps,
            data=simulation_data,
            analysis_results=analysis_results,
            export_paths=[],
        )

    async def run_simulation(self, request: SimulationRequest) -> SimulationResponse:
        """Run a physics simulation based on request parameters.

        The CPU-bound stepping loop is offloaded to a worker thread via
        ``anyio.to_thread.run_sync`` so the event loop stays responsive and
        concurrent requests are not starved (issue #6988).

        Args:
            request: Simulation request parameters

        Returns:
            Simulation results and data
        """
        try:
            # run_sync is typed to return Any; bind to the declared type so
            # mypy-strict's no-any-return is satisfied.
            response: SimulationResponse = await anyio.to_thread.run_sync(
                self._run_simulation_sync, request
            )
            return response
        except (GolfSuiteError, ValueError, RuntimeError) as e:
            logger.error("Simulation failed: %s", e, exc_info=True)
            return SimulationResponse(
                success=False,
                duration=0.0,
                frames=0,
                data={},
                analysis_results=None,
                export_paths=[],
            )

    @precondition(
        lambda self, task_id, request, active_tasks: (
            task_id is not None and len(task_id) > 0
        ),
        "Task ID must be a non-empty string",
    )
    @precondition(
        lambda self, task_id, request, active_tasks: active_tasks is not None,
        "Active tasks dictionary must not be None",
    )
    async def run_simulation_background(
        self, task_id: str, request: SimulationRequest, active_tasks: Any
    ) -> None:
        """Run simulation as background task.

        Args:
            task_id: Unique task identifier
            request: Simulation request
            active_tasks: Dictionary to store task status
        """
        try:
            active_tasks.set(task_id, {"status": "running", "progress": 0})

            result = await self.run_simulation(request)

            if result.success:
                active_tasks.set(
                    task_id,
                    {
                        "status": "completed",
                        "result": result.model_dump(),
                    },
                )
            else:
                active_tasks.set(
                    task_id,
                    {
                        "status": "failed",
                        "result": result.model_dump(),
                    },
                )

        except (GolfSuiteError, ValueError, RuntimeError, OSError) as e:
            active_tasks.set(task_id, {"status": "failed", "error": str(e)})

    def _extract_simulation_data(
        self, recorder: GenericPhysicsRecorder
    ) -> dict[str, Any]:
        """Extract simulation data from recorder.

        Args:
            recorder: Physics recorder with simulation data

        Returns:
            Dictionary containing simulation data
        """
        if not (recorder is not None):
            raise ValueError("recorder must be provided")
        data = {}

        try:
            # Extract time series data
            times, positions = recorder.get_time_series("joint_positions")
            data["times"] = times.tolist() if hasattr(times, "tolist") else times
            data["joint_positions"] = (
                positions.tolist() if hasattr(positions, "tolist") else positions
            )

            times, velocities = recorder.get_time_series("joint_velocities")
            data["joint_velocities"] = (
                velocities.tolist() if hasattr(velocities, "tolist") else velocities
            )

            times, accelerations = recorder.get_time_series("joint_accelerations")
            data["joint_accelerations"] = (
                accelerations.tolist()
                if hasattr(accelerations, "tolist")
                else accelerations
            )

            # Extract control data if available
            try:
                times, controls = recorder.get_time_series("control_inputs")
                data["control_inputs"] = (
                    controls.tolist() if hasattr(controls, "tolist") else controls
                )
            except (KeyError, ValueError, AttributeError) as e:
                logger.debug("Control inputs not available: %s", e)

        except (KeyError, ValueError, AttributeError, TypeError) as e:
            logger.warning("Error extracting simulation data: %s", e)

        return data

    def _perform_analysis(
        self, recorder: GenericPhysicsRecorder, config: dict[str, Any]
    ) -> dict[str, Any]:
        """Perform analysis on simulation data.

        Args:
            recorder: Physics recorder with simulation data
            config: Analysis configuration

        Returns:
            Analysis results
        """
        if not (recorder is not None):
            raise ValueError("recorder must be provided")
        results = {}

        try:
            # Extract ZTCF data if enabled
            if config.get("ztcf", False):
                times, ztcf = recorder.get_time_series("ztcf_accel")
                results["ztcf_acceleration"] = (
                    ztcf.tolist() if hasattr(ztcf, "tolist") else ztcf
                )

            # Extract ZVCF data if enabled
            if config.get("zvcf", False):
                times, zvcf = recorder.get_time_series("zvcf_accel")
                results["zvcf_acceleration"] = (
                    zvcf.tolist() if hasattr(zvcf, "tolist") else zvcf
                )

            # Extract drift analysis if enabled
            if config.get("track_drift", False):
                times, drift = recorder.get_time_series("drift_accel")
                results["drift_acceleration"] = (
                    drift.tolist() if hasattr(drift, "tolist") else drift
                )

        except (KeyError, ValueError, AttributeError, TypeError) as e:
            logger.warning("Error performing analysis: %s", e)

        return results
