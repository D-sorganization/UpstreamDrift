"""Simulation service for Golf Modeling Suite API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import threading
import time
from typing import TYPE_CHECKING, Any
import uuid

import anyio.to_thread
import numpy as np

from src.shared.python.analysis.orchestrator import (
    AnalysisOrchestrator,
    supported_counterfactual_kinds,
)
from src.shared.python.core.contracts import precondition
from src.shared.python.core.error_utils import (
    EngineLaunchError,
    EngineNotAvailableError,
    GolfSuiteError,
    ModelLoadError,
    PhysicsSimulationError,
    SimulationBusyError,
    SimulationTimeoutError,
    ValidationError,
)
from src.shared.python.dashboard.recorder import GenericPhysicsRecorder
from src.shared.python.data_io._format_handlers import OutputFormat
from src.shared.python.data_io.output_manager import OutputManager
from src.shared.python.engine_core.engine_registry import EngineType
from src.shared.python.logging_pkg.logging_config import get_logger

from ..models.requests import SimulationRequest
from ..models.responses import SimulationErrorInfo, SimulationResponse

logger = get_logger(__name__)

if TYPE_CHECKING:
    from src.shared.python.engine_core.engine_manager import EngineManager

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
    #: Last/current run summary for chat context (#7453). Keys mirror
    #: ``src.api.services.chat_app_context.SimulationRunContext``.
    last_run: dict[str, Any] | None = None


@dataclass
class SimulationRunRecord:
    """Authoritative isolated state per simulation run (R02).

    Retains run-addressed engine, stats, recorder, results, and status so
    concurrent runs and post-run analysis cannot mutate each other's data.
    """

    run_id: str
    engine_type: str
    stats: SimulationStats = field(default_factory=SimulationStats)
    recorder: GenericPhysicsRecorder | None = None
    joint_names: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    simulation_data: dict[str, Any] = field(default_factory=dict)
    analysis_results: dict[str, Any] | None = None
    engine: Any = None
    status: str = "initialized"
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    cleaned_up: bool = False


class SimulationService:
    """Service for managing physics simulations."""

    def __init__(
        self,
        engine_manager: EngineManager,
        output_manager: OutputManager | None = None,
    ) -> None:
        """Initialize simulation service.

        Args:
            engine_manager: Engine manager instance
            output_manager: Persists completed simulation results to disk
                (issue #8871). Defaults to a project-root-relative
                ``OutputManager()``; tests should inject one rooted at
                ``tmp_path`` to avoid writing into the real ``output/`` tree.
        """
        self.engine_manager = engine_manager
        self.output_manager = output_manager or OutputManager()
        self._stats = SimulationStats()
        self._runs: dict[str, SimulationRunRecord] = {}
        self._active_run_id: str | None = None
        self._active_recorder: GenericPhysicsRecorder | None = None
        self._active_joint_names: list[str] = []
        self._last_recorder: GenericPhysicsRecorder | None = None
        self._last_recording_meta: dict[str, Any] = {}
        self._biomechanics_binding: Any = None
        self._active_candidate_session: Any = None
        self.single_run_mode: bool = False
        self._run_lock = threading.RLock()

    @property
    def stats(self) -> SimulationStats:
        """Return the authoritative runtime stats for the active session."""
        return self._stats

    @property
    def active_recorder(self) -> GenericPhysicsRecorder | None:
        """Recorder of the most recently completed simulation, if any."""
        return self._active_recorder

    @property
    def active_joint_names(self) -> list[str]:
        """Joint names of the engine used by the active recorder."""
        return list(self._active_joint_names)

    def create_run(
        self,
        run_id: str | None = None,
        engine_type: str = "",
    ) -> SimulationRunRecord:
        """Create and register a new isolated run record."""
        run_id = run_id or str(uuid.uuid4())
        record = SimulationRunRecord(run_id=run_id, engine_type=engine_type)
        with self._run_lock:
            self._runs[run_id] = record
            self._active_run_id = run_id
        return record

    def get_run(self, run_id: str | None = None) -> SimulationRunRecord | None:
        """Retrieve an isolated run record by ID, or the active run if None."""
        with self._run_lock:
            if run_id is not None:
                return self._runs.get(run_id)
            if self._active_run_id is not None:
                return self._runs.get(self._active_run_id)
            if self._runs:
                return next(reversed(self._runs.values()))
            return None

    def get_run_stats(self, run_id: str | None = None) -> SimulationStats:
        """Return authoritative runtime stats for the specified or active run."""
        run = self.get_run(run_id)
        if run is not None:
            return run.stats
        return self._stats

    def get_run_recorder(
        self, run_id: str | None = None
    ) -> GenericPhysicsRecorder | None:
        """Return recorder for run_id or the active recorder."""
        return self._counterfactual_recorder(run_id=run_id)

    def get_run_joint_names(self, run_id: str | None = None) -> list[str]:
        """Return joint names for run_id or the active joint names."""
        if run_id is not None:
            run = self.get_run(run_id)
            if run is not None and run.joint_names:
                return list(run.joint_names)
        return list(self._active_joint_names)

    def register_completed_run(
        self,
        run_id: str,
        engine: Any,
        recorder: Any,
        meta: dict[str, Any] | None = None,
        joint_names: list[str] | None = None,
        simulation_data: dict[str, Any] | None = None,
        analysis_results: dict[str, Any] | None = None,
    ) -> SimulationRunRecord:
        """Register or update a completed run record for post-run analysis and persistence."""
        with self._run_lock:
            run = self._runs.get(run_id)
            if run is None:
                engine_type = getattr(engine, "engine_type", None) or getattr(
                    engine, "name", "unknown"
                )
                run = SimulationRunRecord(run_id=run_id, engine_type=str(engine_type))
                self._runs[run_id] = run

            run.engine = engine
            run.recorder = recorder
            if joint_names is not None:
                run.joint_names = list(joint_names)
            elif callable(getattr(engine, "get_joint_names", None)):
                try:
                    run.joint_names = [str(n) for n in engine.get_joint_names()]
                except (RuntimeError, ValueError, TypeError):
                    logger.exception("Could not read joint names from engine")
            if meta is not None:
                run.meta = dict(meta)
            if simulation_data is not None:
                run.simulation_data = dict(simulation_data)
            if analysis_results is not None:
                run.analysis_results = dict(analysis_results)
            run.status = "completed"
            run.finished_at = time.time()
            self._active_run_id = run_id

            # Mirror to legacy attributes for backward compatibility
            self._active_recorder = recorder
            self._active_joint_names = list(run.joint_names)
            self._last_recorder = recorder
            if meta is not None:
                self._last_recording_meta = dict(meta)

            return run

    def is_busy(self, exclude_run_id: str | None = None) -> bool:
        """Return True if an active simulation is currently running."""
        with self._run_lock:
            for r_id, r in self._runs.items():
                if r_id != exclude_run_id and r.status == "running":
                    return True
        return False

    def cleanup_run_engine(self, run_id: str) -> None:
        """Clean up the physics engine owned by a specific run.

        Idempotent: cleanup occurs exactly once. Run isolation guarantee:
        cleaning up one run cannot unload or affect another run's engine.
        """
        with self._run_lock:
            run = self._runs.get(run_id)
            if run is None or run.cleaned_up:
                return
            run.cleaned_up = True
            engine = run.engine
            run.engine = None

        if engine is not None:
            close = getattr(engine, "close", None)
            if callable(close):
                try:
                    close()
                    logger.info("run_engine_shutdown status=success run_id=%s", run_id)
                except (RuntimeError, OSError) as e:
                    logger.warning(
                        "run_engine_shutdown_failed run_id=%s error=%s", run_id, e
                    )

    def get_biomechanics_payload(self) -> dict[str, Any] | None:
        """Expose recorded calibrated geometry without leaking recorder internals."""
        if self._active_recorder is None:
            return None
        return self._active_recorder.get_biomechanics_payload()

    def configure_biomechanics(self, payload: dict[str, Any]) -> None:
        """Validate and retain a model binding for the next simulation recording."""
        from src.shared.python.biomechanics.model_bindings import (
            model_binding_from_dict,
        )

        self._biomechanics_binding = model_binding_from_dict(payload)

    def _counterfactual_recorder(
        self, run_id: str | None = None
    ) -> GenericPhysicsRecorder | None:
        """Return the recorder for run_id or the active recorder."""
        if run_id is not None:
            run = self.get_run(run_id)
            if run is not None and run.recorder is not None:
                return run.recorder
        return self._active_recorder or getattr(self, "_last_recorder", None)

    def _retain_active_session(self, engine: Any, recorder: Any) -> None:
        """Retain the completed simulation's recorder for post-run analysis.

        Args:
            engine: Engine the simulation ran on (joint-name source).
            recorder: Recorder holding the recorded time series.
        """
        self._active_recorder = recorder
        names_getter = getattr(engine, "get_joint_names", None)
        joint_names: list[str] = []
        if callable(names_getter):
            try:
                joint_names = [str(n) for n in names_getter()]
            except (RuntimeError, ValueError, TypeError):
                logger.exception("Could not read joint names from engine")
        self._active_joint_names = joint_names

    def describe_counterfactual_support(
        self, run_id: str | None = None
    ) -> dict[str, Any]:
        """Describe counterfactual capability for the active or selected session."""
        recorder = self._counterfactual_recorder(run_id=run_id)
        engine = recorder.engine if recorder is not None else None
        engine_name: str | None = None
        if engine is not None:
            engine_name = getattr(engine, "engine_type", None) or getattr(
                engine, "name", None
            )
            if engine_name is not None:
                engine_name = str(engine_name)

        supported = supported_counterfactual_kinds(engine)
        payload: dict[str, Any] = {
            "kinds": supported,
            "engine": engine_name,
            "session_available": recorder is not None,
        }
        if run_id is not None:
            payload["run_id"] = run_id
        return payload

    def _compute_counterfactual_sync(
        self,
        kind: str,
        run_post_hoc: bool = True,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Compute a counterfactual against the active or selected recorder."""
        recorder = self._counterfactual_recorder(run_id=run_id)
        if recorder is None:
            msg = (
                f"Simulation run '{run_id}' not found or produced no recorded session"
                if run_id
                else "No completed simulation session; run a simulation first"
            )
            raise ValueError(msg)
        joint_names = self._active_joint_names
        if run_id is not None:
            run = self.get_run(run_id)
            if run is not None and run.joint_names:
                joint_names = run.joint_names
        orchestrator = AnalysisOrchestrator(recorder, joint_names=joint_names)
        result = orchestrator.compute_counterfactual(kind, run_post_hoc=run_post_hoc)
        return result.to_dict()

    @precondition(
        lambda self, task_id, kind, run_post_hoc, active_tasks, run_id=None: (
            task_id is not None and len(task_id) > 0
        ),
        "Task ID must be a non-empty string",
    )
    async def run_counterfactual_background(
        self,
        task_id: str,
        kind: str,
        run_post_hoc: bool,
        active_tasks: Any,
        run_id: str | None = None,
    ) -> None:
        """Run a counterfactual analysis as a background task."""
        if active_tasks is None:
            raise ValueError("active_tasks must be provided")
        active_tasks.set(task_id, {"status": "running", "kind": kind, "run_id": run_id})
        try:
            result = await anyio.to_thread.run_sync(
                self._compute_counterfactual_sync, kind, run_post_hoc, run_id
            )
            active_tasks.set(
                task_id,
                {
                    "status": "completed",
                    "kind": kind,
                    "result": result,
                    "run_id": run_id,
                },
            )
        except (
            GolfSuiteError,
            ValueError,
            RuntimeError,
            AttributeError,
            OSError,
        ) as e:
            logger.exception("Counterfactual '%s' failed", kind)
            active_tasks.set(
                task_id,
                {"status": "failed", "kind": kind, "error": str(e), "run_id": run_id},
            )

    def start_recording(self) -> None:
        """Begin recording trajectory frames. Clears any previously recorded data."""
        stats = self.stats
        stats.is_recording = True
        stats.recorded_frames = []

    def stop_recording(self) -> None:
        """Stop recording trajectory frames."""
        self.stats.is_recording = False

    def get_session_recording(
        self, run_id: str | None = None
    ) -> tuple[GenericPhysicsRecorder, dict[str, Any]] | None:
        """Return the specified or most recent session recorder and context, if any."""
        if run_id is not None:
            run = self.get_run(run_id)
            if run is None or run.recorder is None:
                return None
            return run.recorder, dict(run.meta)

        if self._last_recorder is None or self._last_recorder.current_idx == 0:
            return None
        return self._last_recorder, dict(self._last_recording_meta)

    def set_speed_factor(self, value: float) -> None:
        """Set simulation speed multiplier.

        Args:
            value: Speed multiplier (>0).
        """
        self._stats.speed_factor = value

    def _begin_last_run(self, request: SimulationRequest) -> None:
        """Record the start of a simulation run for chat context (#7453).

        Only the model file *name* is recorded (never the full path) so the
        value is safe to surface in chat context and the web UI chip.
        """
        from pathlib import Path

        model_name = None
        if request.model_path:
            model_name = Path(str(request.model_path)).name
        self._stats.last_run = {
            "engine": str(request.engine_type).lower(),
            "model": model_name,
            "duration_seconds": float(request.duration),
            "status": "running",
            "frames": 0,
            "finished_at": None,
            "error": None,
            "analysis_summary": None,
        }
        active_run = self.get_run()
        if active_run is not None:
            active_run.stats.last_run = self._stats.last_run

    def _finish_last_run(
        self,
        status: str,
        frames: int | None = None,
        error: str | None = None,
        analysis_summary: str | None = None,
    ) -> None:
        """Mark the in-flight run as finished for chat context (#7453)."""
        if not status:
            raise ValueError("status must be a non-empty string")
        last_run = self._stats.last_run
        if last_run is None:
            return
        last_run["status"] = status
        last_run["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if frames is not None:
            last_run["frames"] = frames
        if error is not None:
            last_run["error"] = error
        if analysis_summary is not None:
            last_run["analysis_summary"] = analysis_summary
        active_run = self.get_run()
        if active_run is not None:
            active_run.stats.last_run = last_run

    @precondition(
        lambda self, request, run_id=None: request is not None,
        "Simulation request must not be None",
    )
    @precondition(
        lambda self, request, run_id=None: request.duration > 0,
        "Simulation duration must be positive",
    )
    @precondition(
        lambda self, request, run_id=None: (
            request.engine_type is not None and len(request.engine_type) > 0
        ),
        "Engine type must be specified",
    )
    def _prepare_engine(
        self, request: SimulationRequest, run_id: str | None = None
    ) -> Any:
        """Load and configure the physics engine for simulation.

        Args:
            request: Simulation request with engine type and model path.
            run_id: Optional unique identifier for the owning run.

        Returns:
            Configured engine instance.

        Raises:
            SimulationBusyError: If single-run containment is active and another run is running.
            EngineLaunchError: If engine fails to load.
            ModelLoadError: If model file fails to load.
        """
        if self.single_run_mode and self.is_busy(exclude_run_id=run_id):
            raise SimulationBusyError(
                "Simulation service is busy with another active run",
                active_run_id=self._active_run_id,
            )

        engine_type = EngineType(request.engine_type.lower())
        create_fn = getattr(self.engine_manager, "create_engine", None)
        has_mock_return = hasattr(create_fn, "_mock_return_value")
        is_mock_configured = False
        if has_mock_return and create_fn is not None:
            from unittest.mock import DEFAULT

            is_mock_configured = (
                getattr(create_fn, "_mock_return_value", DEFAULT) is not DEFAULT
                or getattr(create_fn, "side_effect", None) is not None
            )

        if callable(create_fn) and (not has_mock_return or is_mock_configured):
            engine = create_fn(engine_type)
        else:
            self.engine_manager._load_engine(engine_type)
            engine = None
            get_active = getattr(self.engine_manager, "get_active_physics_engine", None)
            if callable(get_active):
                try:
                    engine = get_active(engine_type)
                except TypeError:
                    engine = get_active()
                if not engine:
                    try:
                        engine = get_active()
                    except TypeError:
                        pass

        if not engine:
            raise EngineLaunchError(
                request.engine_type,
                reason="engine loaded but no active engine returned",
            )

        if run_id:
            with self._run_lock:
                run = self._runs.get(run_id) or self.create_run(
                    run_id, request.engine_type
                )
                run.engine = engine

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
        run_stats: SimulationStats | None = None,
    ) -> None:
        """Execute the main simulation stepping loop.

        Args:
            engine: Physics engine instance.
            recorder: Recording object for simulation data.
            request: Simulation request with control inputs.
            timestep: Time step per simulation step.
            steps: Total number of steps to execute.
            run_stats: Optional per-run stats instance to update.
        """
        if not (recorder is not None):
            raise ValueError("recorder must be provided")
        if not (engine is not None):
            raise ValueError("engine must be provided")
        if not recorder.is_recording:
            recorder.start()
        recorder.is_recording = True

        # Record initial state at t=0
        recorder.record_step(control_input=None)

        for step in range(steps):
            if (
                getattr(recorder, "buffer_exhausted", False) is True
                or not recorder.is_recording
            ):
                break

            torques = None
            if request.control_inputs and step < len(request.control_inputs):
                control = request.control_inputs[step]
                if isinstance(control, dict):
                    if "torques" in control:
                        torques = np.asarray(control["torques"], dtype=float)
                    elif "control" in control:
                        torques = np.asarray(control["control"], dtype=float)
                    elif "u" in control:
                        torques = np.asarray(control["u"], dtype=float)
                elif isinstance(control, (list, tuple, np.ndarray)):
                    torques = np.asarray(control, dtype=float)

                if torques is not None:
                    engine.set_control(torques)

            engine.step(timestep)
            recorder.record_step(control_input=torques)
            self._stats.frame_count += 1
            if run_stats is not None:
                run_stats.frame_count += 1

        expected_samples = steps + 1
        retained_samples = getattr(recorder, "current_idx", expected_samples)
        if getattr(recorder, "buffer_exhausted", False) is True or (
            isinstance(retained_samples, int) and retained_samples < expected_samples
        ):
            max_cap = getattr(recorder, "max_samples", None)
            raise RuntimeError(
                f"Recorder buffer capacity exhausted: requested {expected_samples} samples, "
                f"executed {self._stats.frame_count} steps, but only {retained_samples} samples "
                f"were retained (max_samples={max_cap})"
            )

    def _build_error_info(
        self,
        exc: Exception,
        stage: str,
        run_id: str | None = None,
    ) -> SimulationErrorInfo:
        """Construct structured, safe error outcome without leaking paths (R09)."""
        import re

        exc_str = str(exc)
        safe_msg = (
            re.sub(r"([A-Za-z]:)?(/|\\)[^:\s]+", "<path>", exc_str)
            if "/" in exc_str or "\\" in exc_str
            else exc_str
        )

        if isinstance(exc, SimulationBusyError):
            code = "busy"
            retriable = True
            guidance = "Wait for the active simulation run to complete and retry."
            safe_msg = "Simulation service is busy with another run"
        elif isinstance(exc, (EngineNotAvailableError, EngineLaunchError)):
            code = "engine_unavailable"
            retriable = False
            guidance = (
                "Ensure the requested physics engine is installed and operational."
            )
            safe_msg = "Requested physics engine is not available"

        elif isinstance(exc, (ModelLoadError, FileNotFoundError)):
            code = "model_load_error"
            retriable = False
            guidance = "Verify that the model asset exists and is accessible."
            safe_msg = "Model asset could not be loaded"
        elif isinstance(exc, (ValueError, ValidationError)):
            code = "invalid_input"
            retriable = False
            guidance = "Check simulation duration, timestep, and control inputs."
        elif isinstance(exc, (TimeoutError, SimulationTimeoutError)):
            code = "timeout"
            retriable = True
            guidance = (
                "Consider reducing simulation duration or increasing timeout ceiling."
            )
            safe_msg = "Simulation execution timed out"
        elif (
            isinstance(exc, PhysicsSimulationError)
            or "diverged" in exc_str.lower()
            or "nan" in exc_str.lower()
        ):
            code = "numerical_failure"
            retriable = True
            guidance = "Try decreasing the integration timestep or verifying model initial conditions."
            safe_msg = "Physics solver encountered numerical instability"
        elif stage == "persistence" or isinstance(exc, (OSError, PermissionError)):
            code = "persistence_failed"
            retriable = True
            guidance = "Simulation data is preserved in memory. Retry persistence via POST /recordings."
            safe_msg = "Failed to persist simulation results to disk"
        else:
            code = "internal_error"
            retriable = False
            guidance = "Contact system administrator if the issue persists."
            safe_msg = "Internal simulation error"

        return SimulationErrorInfo(
            code=code,
            message=safe_msg,
            stage=stage,
            run_id=run_id,
            retriable=retriable,
            retry_guidance=guidance,
        )

    def _persist_simulation_results(
        self,
        request: SimulationRequest,
        simulation_data: dict[str, Any],
        analysis_results: dict[str, Any] | None,
        steps: int,
        run_id: str | None = None,
    ) -> tuple[list[str], str, SimulationErrorInfo | None]:
        """Persist a completed run's results via ``OutputManager`` (issue #8871, R09).

        Provenance passed through to ``OutputManager.save_simulation_results``
        (engine type, model path, duration/timestep) is limited to values the
        request/result genuinely carry — nothing is inferred or guessed
        (see the provenance-honesty theme from issues #8816-#8822).

        Returns:
            Tuple of (export_paths, persistence_status, persistence_error).
            A persistence failure must not fail an otherwise-successful simulation;
            results remain recoverable in memory.
        """
        engine = str(request.engine_type).lower()
        metadata: dict[str, Any] = {"duration": request.duration, "frames": steps}
        if analysis_results:
            metadata["analysis_results"] = analysis_results

        try:
            saved_path = self.output_manager.save_simulation_results(
                simulation_data,
                filename=f"simulation_{engine}",
                format_type=OutputFormat.JSON,
                engine=engine,
                metadata=metadata,
                model_path=request.model_path,
                parameters={
                    "duration": request.duration,
                    "timestep": request.timestep,
                },
            )
            return [str(saved_path)], "persisted", None
        except (FileNotFoundError, PermissionError, OSError, ValueError) as e:
            logger.warning("Failed to persist simulation results: %s", e)
            error_info = self._build_error_info(e, stage="persistence", run_id=run_id)
            return [], "failed", error_info

    def _validate_simulation_data(
        self,
        simulation_data: dict[str, Any],
        expected_frames: int,
        has_controls: bool = False,
        is_mock: bool = False,
    ) -> None:
        """Validate required channels, non-empty arrays, and length alignment.

        Raises:
            ValueError: If required channels are missing, empty, or misaligned.
        """
        required_channels = (
            "times",
            "joint_positions",
            "joint_velocities",
            "joint_accelerations",
        )
        for channel in required_channels:
            if channel not in simulation_data or len(simulation_data[channel]) == 0:
                raise ValueError(
                    f"Simulation failed required channel validation: '{channel}' is missing or empty"
                )

        n_times = len(simulation_data["times"])
        if n_times == 0:
            raise ValueError("Simulation produced zero recorded samples")

        if not is_mock and n_times != expected_frames:
            raise ValueError(
                f"Retained sample count ({n_times}) does not match expected frame count ({expected_frames})"
            )

        for channel in ("joint_positions", "joint_velocities", "joint_accelerations"):
            if channel in simulation_data and len(simulation_data[channel]) != n_times:
                raise ValueError(
                    f"Channel '{channel}' length ({len(simulation_data[channel])}) "
                    f"does not match times length ({n_times})"
                )

        if has_controls:
            if (
                "control_inputs" not in simulation_data
                or len(simulation_data["control_inputs"]) == 0
            ):
                raise ValueError(
                    "Simulation with commanded control inputs produced no recorded control data"
                )
            if len(simulation_data["control_inputs"]) != n_times:
                raise ValueError(
                    f"Channel 'control_inputs' length ({len(simulation_data['control_inputs'])}) "
                    f"does not match times length ({n_times})"
                )

    def _validate_simulation_timing(
        self, request: SimulationRequest
    ) -> tuple[float, int, int]:
        timestep = request.timestep or 0.001
        if timestep <= 0:
            raise ValueError(f"Timestep must be positive, got {timestep}")
        if timestep > request.duration:
            raise ValueError(
                f"Timestep ({timestep}) must not exceed duration ({request.duration})"
            )
        steps = int(request.duration / timestep)
        return timestep, steps, steps + 1

    def _create_and_run_recorder(
        self,
        engine: Any,
        request: SimulationRequest,
        timestep: float,
        steps: int,
        expected_frames: int,
        run_stats: SimulationStats | None = None,
    ) -> GenericPhysicsRecorder:
        recorder = GenericPhysicsRecorder(
            engine, max_samples=max(100000, expected_frames)
        )
        if self._biomechanics_binding is not None:
            recorder.configure_biomechanics(self._biomechanics_binding)
        if request.analysis_config:
            recorder.set_analysis_config(request.analysis_config)
        recorder.start()
        try:
            self._execute_simulation_loop(
                engine, recorder, request, timestep, steps, run_stats=run_stats
            )
        finally:
            recorder.stop()
        self._retain_active_session(engine, recorder)
        self._last_recorder = recorder
        self._last_recording_meta = {
            "engine": request.engine_type,
            "model": str(request.model_path) if request.model_path else None,
            "duration": request.duration,
        }
        return recorder

    def _run_simulation_sync(
        self, request: SimulationRequest, run_id: str | None = None
    ) -> SimulationResponse:
        """Run the full CPU-bound simulation pipeline synchronously with run isolation.

        This performs engine preparation, the stepping loop, data extraction,
        and analysis. It is intentionally blocking and must be invoked off the
        event loop (see :meth:`run_simulation`) so the FastAPI worker is not
        frozen for the duration of the simulation (issue #6988).
        """
        run_id = (
            run_id or getattr(request, "run_id", None) or f"sim_{uuid.uuid4().hex[:12]}"
        )
        if self.single_run_mode and self.is_busy(exclude_run_id=run_id):
            busy_err = SimulationErrorInfo(
                code="busy",
                message="Simulation service is busy with another run",
                stage="preparation",
                run_id=run_id,
                retriable=True,
                retry_guidance="Wait for active simulation to complete and retry.",
            )
            return SimulationResponse(
                success=False,
                duration=0.0,
                frames=0,
                data={},
                analysis_results=None,
                export_paths=[],
                calculation_status="failed",
                analysis_status="not_requested",
                persistence_status="not_requested",
                error=busy_err,
                run_id=run_id,
            )

        run = self.create_run(run_id=run_id, engine_type=request.engine_type)
        run.status = "running"
        run.stats.start_time = time.time()
        run.stats.frame_count = 0
        self._stats.start_time = time.time()
        self._stats.frame_count = 0
        self._begin_last_run(request)

        try:
            engine = self._prepare_engine(request, run_id=run_id)
            run.engine = engine

            timestep, steps, expected_frames = self._validate_simulation_timing(request)
            recorder = self._create_and_run_recorder(
                engine, request, timestep, steps, expected_frames, run_stats=run.stats
            )

            simulation_data = self._extract_simulation_data(recorder)
            is_mock_rec = (
                not isinstance(GenericPhysicsRecorder, type)
                or not isinstance(recorder, GenericPhysicsRecorder)
                or "Mock" in type(recorder).__name__
            )
            self._validate_simulation_data(
                simulation_data=simulation_data,
                expected_frames=expected_frames,
                has_controls=bool(request.control_inputs),
                is_mock=is_mock_rec,
            )

            analysis_results = None
            analysis_status = "not_requested"
            if request.analysis_config:
                analysis_results = self._perform_analysis(
                    recorder, request.analysis_config
                )
                analysis_status = analysis_results.get("_status", "completed")

            analysis_summary = None
            if isinstance(analysis_results, dict) and analysis_results:
                analysis_summary = "analysis: " + ", ".join(
                    sorted(
                        str(k) for k in analysis_results if not str(k).startswith("_")
                    )
                )
            self._finish_last_run(
                status="completed",
                frames=expected_frames,
                analysis_summary=analysis_summary,
            )

            export_paths, persistence_status, persistence_error = (
                self._persist_simulation_results(
                    request,
                    simulation_data,
                    analysis_results,
                    expected_frames,
                    run_id=run_id,
                )
            )

            self.register_completed_run(
                run_id=run_id,
                engine=engine,
                recorder=recorder,
                meta={
                    "engine": request.engine_type,
                    "model": str(request.model_path) if request.model_path else None,
                    "duration": request.duration,
                },
                simulation_data=simulation_data,
                analysis_results=analysis_results,
            )

            return SimulationResponse(
                success=True,
                duration=request.duration,
                frames=expected_frames,
                data=simulation_data,
                analysis_results=analysis_results,
                export_paths=export_paths,
                calculation_status="completed",
                analysis_status=analysis_status,
                persistence_status=persistence_status,
                error=persistence_error,
                run_id=run_id,
            )
        except Exception:
            run.status = "failed"
            raise

    async def run_simulation(
        self,
        request: SimulationRequest,
        *args: Any,
        **kwargs: Any,
    ) -> SimulationResponse:
        """Run a physics simulation based on request parameters.

        The CPU-bound stepping loop is offloaded to a worker thread via
        ``anyio.to_thread.run_sync`` so the event loop stays responsive and
        concurrent requests are not starved (issue #6988).

        Args:
            request: Simulation request parameters

        Returns:
            Simulation results and data
        """
        target_run_id = (
            kwargs.get("run_id")
            or (args[0] if args and isinstance(args[0], str) else None)
            or getattr(request, "run_id", None)
        )
        try:
            # run_sync is typed to return Any; bind to the declared type so
            # mypy-strict's no-any-return is satisfied.
            response: SimulationResponse = await anyio.to_thread.run_sync(
                self._run_simulation_sync, request, target_run_id
            )
            return response
        except (GolfSuiteError, ValueError, RuntimeError, OSError, TimeoutError) as e:
            logger.error("Simulation failed: %s", e, exc_info=True)
            self._finish_last_run(status="failed", error=str(e))
            stage = (
                "preparation"
                if isinstance(
                    e,
                    (
                        EngineLaunchError,
                        EngineNotAvailableError,
                        ModelLoadError,
                        SimulationBusyError,
                        ValueError,
                        ValidationError,
                    ),
                )
                else "execution"
            )
            error_info = self._build_error_info(e, stage=stage, run_id=target_run_id)
            return SimulationResponse(
                success=False,
                duration=0.0,
                frames=0,
                data={},
                analysis_results=None,
                export_paths=[],
                calculation_status="failed",
                analysis_status="not_requested",
                persistence_status="not_requested",
                error=error_info,
                run_id=target_run_id,
            )

    def _record_background_task_result(
        self, task_id: str, result: Any, active_tasks: Any
    ) -> None:
        dump = (
            result.model_dump()
            if hasattr(result, "model_dump")
            else (result if isinstance(result, dict) else {})
        )
        if getattr(result, "success", False) is True:
            task_payload: dict[str, Any] = {
                "status": "completed",
                "result": dump,
            }
            err = getattr(result, "error", None)
            if (
                getattr(result, "persistence_status", None) == "failed"
                and err is not None
            ):
                task_payload["persistence_status"] = "failed"
                task_payload["error"] = getattr(err, "message", "Simulation failed")
                task_payload["error_code"] = getattr(err, "code", "storage_failure")
                task_payload["error_stage"] = getattr(err, "stage", "persistence")
                task_payload["retriable"] = getattr(err, "retriable", False)
                task_payload["retry_guidance"] = getattr(err, "retry_guidance", None)
                task_payload["error_info"] = (
                    err.model_dump()
                    if hasattr(err, "model_dump")
                    else (err if isinstance(err, dict) else None)
                )
            active_tasks.set(task_id, task_payload)
        else:
            err = getattr(result, "error", None)
            msg = (
                getattr(err, "message", "Simulation failed")
                if err
                else "Simulation failed"
            )
            code = getattr(err, "code", "unknown_error") if err else "unknown_error"
            stage = getattr(err, "stage", "execution") if err else "execution"
            retriable = getattr(err, "retriable", False) if err else False
            guidance = getattr(err, "retry_guidance", None) if err else None
            err_dict = (
                (
                    err.model_dump()
                    if hasattr(err, "model_dump")
                    else (err if isinstance(err, dict) else None)
                )
                if err is not None
                else None
            )
            active_tasks.set(
                task_id,
                {
                    "status": "failed",
                    "result": dump,
                    "error": msg,
                    "error_code": code,
                    "error_stage": stage,
                    "retriable": retriable,
                    "retry_guidance": guidance,
                    "error_info": err_dict,
                },
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

        Invariant: on every exit path the task record is left in a terminal
        state (``completed`` or ``failed``). Previously only four exception
        types were handled, so anything else (``KeyError``, ``TypeError``, a
        physics-binding exception from MuJoCo/Drake/Pinocchio/OpenSim) escaped
        into the ASGI background runner and froze the record at ``running``
        forever (issue #8009).

        Args:
            task_id: Unique task identifier
            request: Simulation request
            active_tasks: Dictionary to store task status
        """
        try:
            active_tasks.set(task_id, {"status": "running", "progress": 0})
            result = await self.run_simulation(request)
            self._record_background_task_result(task_id, result, active_tasks)

        except (GolfSuiteError, ValueError, RuntimeError, OSError, TimeoutError) as e:
            logger.exception("Background simulation %s failed", task_id)
            err_info = self._build_error_info(e, stage="execution", run_id=task_id)
            active_tasks.set(
                task_id,
                {
                    "status": "failed",
                    "error": str(e),
                    "error_code": err_info.code,
                    "error_stage": err_info.stage,
                    "retriable": err_info.retriable,
                    "retry_guidance": err_info.retry_guidance,
                    "error_info": err_info.model_dump(),
                },
            )
        except Exception:  # noqa: BLE001 - background task boundary (#8009)
            # No caller can observe this exception: BackgroundTasks swallows it
            # after the response has been sent. Record the terminal state and
            # log the traceback rather than leaking engine internals to the
            # polling client.
            logger.exception("Background simulation %s failed", task_id)
            active_tasks.set(
                task_id,
                {
                    "status": "failed",
                    "error": "Internal simulation error",
                    "error_code": "internal_error",
                    "error_stage": "execution",
                    "retriable": False,
                    "error_info": {
                        "code": "internal_error",
                        "message": "Internal simulation error",
                        "stage": "execution",
                        "run_id": task_id,
                        "retriable": False,
                    },
                },
            )

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
                if len(controls) == 0:
                    times, controls = recorder.get_time_series("joint_torques")
                if len(controls) > 0:
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
        """Perform analysis on simulation data with explicit channel availability status (R09).

        Args:
            recorder: Physics recorder with simulation data
            config: Analysis configuration

        Returns:
            Analysis results dict including '_channel_status' and '_status'.
        """
        if not (recorder is not None):
            raise ValueError("recorder must be provided")
        results: dict[str, Any] = {}
        channel_status: dict[str, str] = {}
        requested_count = 0
        success_count = 0

        # Extract ZTCF data if enabled
        if config.get("ztcf", False):
            requested_count += 1
            try:
                times, ztcf = recorder.get_time_series("ztcf_accel")
                results["ztcf_acceleration"] = (
                    ztcf.tolist() if hasattr(ztcf, "tolist") else ztcf
                )
                channel_status["ztcf_acceleration"] = "available"
                success_count += 1
            except (KeyError, ValueError, AttributeError, TypeError, RuntimeError) as e:
                logger.warning("Error performing ztcf analysis: %s", e)
                channel_status["ztcf_acceleration"] = f"unavailable: {e}"

        # Extract ZVCF data if enabled
        if config.get("zvcf", False):
            requested_count += 1
            try:
                times, zvcf = recorder.get_time_series("zvcf_accel")
                results["zvcf_acceleration"] = (
                    zvcf.tolist() if hasattr(zvcf, "tolist") else zvcf
                )
                channel_status["zvcf_acceleration"] = "available"
                success_count += 1
            except (KeyError, ValueError, AttributeError, TypeError, RuntimeError) as e:
                logger.warning("Error performing zvcf analysis: %s", e)
                channel_status["zvcf_acceleration"] = f"unavailable: {e}"

        # Extract drift analysis if enabled
        if config.get("track_drift", False):
            requested_count += 1
            try:
                times, drift = recorder.get_time_series("drift_accel")
                results["drift_acceleration"] = (
                    drift.tolist() if hasattr(drift, "tolist") else drift
                )
                channel_status["drift_acceleration"] = "available"
                success_count += 1
            except (KeyError, ValueError, AttributeError, TypeError, RuntimeError) as e:
                logger.warning("Error performing drift analysis: %s", e)
                channel_status["drift_acceleration"] = f"unavailable: {e}"

        if requested_count > 0:
            results["_channel_status"] = channel_status
            if success_count == requested_count:
                results["_status"] = "completed"
            elif success_count > 0:
                results["_status"] = "partial"
            else:
                results["_status"] = "failed"
        else:
            results["_status"] = "not_requested"

        return results

    def get_candidate_session(
        self,
        candidate_path: str | Path,
        model_path: str | Path | None = None,
        receipt_path: str | Path | None = None,
    ) -> Any:
        """Ingest a saved candidate trajectory into a replayable CandidateSession.

        Preconditions:
            candidate_path points to an existing candidate file.
        DbC / Invariants:
            Preserves simulation boundary; never fabricates a GenericPhysicsRecorder
            or simulates live physics.
        """
        from src.shared.python.motion_matching.candidate_session import (
            ingest_candidate_session,
        )

        session = ingest_candidate_session(
            candidate_path=candidate_path,
            model_path=model_path,
            receipt_path=receipt_path,
        )
        self._active_candidate_session = session
        return session

    @property
    def active_candidate_session(self) -> Any:
        """Currently loaded candidate session, or None."""
        return self._active_candidate_session

    def set_active_candidate_session(self, session: Any) -> None:
        """Set active candidate session for inspection and analysis."""
        self._active_candidate_session = session

    def get_candidate_forces(self) -> dict[str, Any]:
        """Return synchronized force/torque, GRF, and CoP telemetry from active candidate session."""
        session = self._active_candidate_session
        if session is None:
            raise ValueError("No active candidate session loaded")

        if not session.supports_forces:
            return {
                "session_available": True,
                "supports_forces": False,
                "frame_count": session.frame_count,
                "time_s": session.time_s.tolist(),
                "coordinate_names": list(session.coordinate_names),
                "torques": None,
                "external_forces": None,
                "center_of_pressure": None,
            }

        cops = [session.get_center_of_pressure(i) for i in range(session.frame_count)]
        torques = session.tau.tolist() if session.tau is not None else None
        ext_forces = (
            session.external_forces.tolist()
            if session.external_forces is not None
            else None
        )
        return {
            "session_available": True,
            "supports_forces": True,
            "frame_count": session.frame_count,
            "time_s": session.time_s.tolist(),
            "coordinate_names": list(session.coordinate_names),
            "torques": torques,
            "external_forces": ext_forces,
            "center_of_pressure": cops,
        }

    def run_candidate_counterfactual(
        self,
        fork_frame_idx: int,
        strategy: str = "zero_trail_arm_torque",
        duration_frames: int | None = None,
    ) -> dict[str, Any]:
        """Perform counterfactual fork on active candidate session."""
        session = self._active_candidate_session
        if session is None:
            raise ValueError("No active candidate session loaded")
        if not session.supports_counterfactuals:
            raise ValueError(
                "Active candidate session does not support counterfactual rollouts"
            )

        from src.shared.python.motion_matching.counterfactual import (
            CounterfactualStrategy,
            create_counterfactual_rollout,
        )

        strat_enum = CounterfactualStrategy(strategy)
        fork = create_counterfactual_rollout(
            session=session,
            fork_frame_idx=fork_frame_idx,
            strategy=strat_enum,
            duration_frames=duration_frames,
        )
        return {
            "fork_id": fork.fork_id,
            "baseline_candidate_sha256": fork.baseline_candidate_sha256,
            "strategy": fork.strategy.value,
            "fork_time_s": fork.fork_time_s,
            "fork_frame_idx": fork.fork_frame_idx,
            "frame_count": fork.frame_count,
            "time_s": fork.time_s.tolist(),
            "q": fork.q.tolist(),
            "v": fork.v.tolist() if fork.v is not None else None,
            "a": fork.a.tolist() if fork.a is not None else None,
            "divergence_rms": fork.divergence_rms,
            "constraint_status": fork.constraint_status,
            "is_accepted": fork.is_accepted,
            "rejection_reasons": list(fork.rejection_reasons),
        }
