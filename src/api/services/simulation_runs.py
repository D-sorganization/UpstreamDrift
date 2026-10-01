"""Authoritative runtime stats, isolated run records, and run store for simulation service."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import TYPE_CHECKING, Any
import uuid

from src.shared.python.core.error_utils import (
    EngineLaunchError,
    EngineNotAvailableError,
    ModelLoadError,
    PhysicsSimulationError,
    SimulationBusyError,
    SimulationTimeoutError,
    ValidationError,
)
from src.shared.python.logging_pkg.logging_config import get_logger

from ..models.responses import SimulationErrorInfo

if TYPE_CHECKING:
    from src.shared.python.dashboard.recorder import GenericPhysicsRecorder

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
    deadline: float | None = None
    cancellation_requested: bool = False
    cancellation_reason: str | None = None

    def cancel(self, reason: str = "Simulation cancelled by user") -> None:
        """Signal cooperative cancellation for this run (R03)."""
        self.cancellation_requested = True
        self.cancellation_reason = reason
        self.status = "cancelled"

    def is_cancelled(self) -> bool:
        """Return True if cancellation has been requested."""
        return self.cancellation_requested or self.status == "cancelled"

    def is_deadline_exceeded(self) -> bool:
        """Return True if execution deadline has passed."""
        return self.deadline is not None and time.time() > self.deadline


class SimulationRunStore:
    """Thread-safe store for isolated simulation runs."""

    def __init__(self) -> None:
        self._runs: dict[str, SimulationRunRecord] = {}
        self._active_run_id: str | None = None
        self._lock = threading.RLock()

    @property
    def runs(self) -> dict[str, SimulationRunRecord]:
        """Access backing runs dict."""
        return self._runs

    @property
    def active_run_id(self) -> str | None:
        """Access active run id."""
        return self._active_run_id

    @active_run_id.setter
    def active_run_id(self, val: str | None) -> None:
        self._active_run_id = val

    @property
    def lock(self) -> threading.RLock:
        """Access store lock."""
        return self._lock

    def create_run(
        self,
        run_id: str | None = None,
        engine_type: str = "",
    ) -> SimulationRunRecord:
        """Create and register a new isolated run record."""
        run_id = run_id or str(uuid.uuid4())
        record = SimulationRunRecord(run_id=run_id, engine_type=engine_type)
        with self._lock:
            self._runs[run_id] = record
            self._active_run_id = run_id
        return record

    def get_run(self, run_id: str | None = None) -> SimulationRunRecord | None:
        """Retrieve an isolated run record by ID, or the active run if None."""
        with self._lock:
            if run_id is not None:
                return self._runs.get(run_id)
            if self._active_run_id is not None:
                return self._runs.get(self._active_run_id)
            if self._runs:
                return next(reversed(self._runs.values()))
            return None

    def get_run_stats(
        self,
        fallback_stats: SimulationStats,
        run_id: str | None = None,
    ) -> SimulationStats:
        """Return authoritative runtime stats for the specified or active run."""
        run = self.get_run(run_id)
        if run is not None:
            return run.stats
        return fallback_stats

    def get_run_recorder(
        self,
        fallback_recorder: GenericPhysicsRecorder | None = None,
        run_id: str | None = None,
    ) -> GenericPhysicsRecorder | None:
        """Return recorder for run_id or fallback."""
        if run_id is not None:
            run = self.get_run(run_id)
            if run is not None and run.recorder is not None:
                return run.recorder
        return fallback_recorder

    def get_run_joint_names(
        self,
        fallback_joint_names: list[str],
        run_id: str | None = None,
    ) -> list[str]:
        """Return joint names for run_id or fallback."""
        if run_id is not None:
            run = self.get_run(run_id)
            if run is not None and run.joint_names:
                return list(run.joint_names)
        return list(fallback_joint_names)

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
        """Register or update a completed run record for post-run analysis."""
        with self._lock:
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
            return run

    def is_busy(self, exclude_run_id: str | None = None) -> bool:
        """Return True if an active simulation is currently running."""
        with self._lock:
            for r_id, r in self._runs.items():
                if r_id != exclude_run_id and r.status == "running":
                    return True
        return False

    def cleanup_run_engine(self, run_id: str) -> None:
        """Clean up the physics engine owned by a specific run (idempotent)."""
        with self._lock:
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


def build_simulation_error_info(
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
        guidance = "Ensure the requested physics engine is installed and operational."
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


def record_background_task_result(
    task_id: str,
    result: Any,
    active_tasks: Any,
) -> None:
    """Record async background simulation task outcomes into active_tasks."""
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
        if getattr(result, "persistence_status", None) == "failed" and err is not None:
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
            getattr(err, "message", "Simulation failed") if err else "Simulation failed"
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
        calc_status = getattr(result, "calculation_status", None)
        task_status = "cancelled" if calc_status == "cancelled" else "failed"
        active_tasks.set(
            task_id,
            {
                "status": task_status,
                "result": dump,
                "error": msg,
                "error_code": code,
                "error_stage": stage,
                "retriable": retriable,
                "retry_guidance": guidance,
                "error_info": err_dict,
            },
        )
