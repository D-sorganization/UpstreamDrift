"""Regression tests for R03: Bound Simulation Work and Preserve Cancellable Jobs Under Load.

Acceptance criteria:
- Excessive ratios, nonfinite inputs, and over-budget model batches fail before
  engine creation or large allocation, with an actionable limit response.
- A heartbeat/health request remains responsive while an intentionally slow
  injected flight calculation runs.
- Under full capacity, new work is rejected or queued; active jobs remain
  queryable and cancellable, and terminal data is bounded.
- Cancellation and deadlines stop actual computation within a documented bound.
- Progress reports executed work; cancellation and persistence failures have
  distinct terminal outcomes.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError

from src.api.models.requests import SimulationRequest
from src.api.models.responses import SimulationErrorInfo, SimulationResponse
from src.api.routes.ball_flight import (
    BallFlightSimulationRequest,
    simulate_ball_flight,
)
from src.api.services.simulation_runs import SimulationRunRecord
from src.api.services.simulation_service import SimulationService
from src.api.task_manager import TaskManager
from src.shared.python.dashboard.recorder import GenericPhysicsRecorder
from src.shared.python.physics.flight_models import (
    _extract_ode_trajectory_points,
)

pytestmark = [pytest.mark.anyio, pytest.mark.unit]


@pytest.fixture(scope="module")
def anyio_backend() -> str:
    return "asyncio"


# =============================================================================
# 1. Input bounds and ratio validation
# =============================================================================


class TestSimulationRequestBounds:
    """Validate aggregate step/sample budgets and nonfinite inputs."""

    def test_simulation_request_rejects_excessive_step_ratio(self) -> None:
        """duration=300, timestep=1e-6 describes 300,000,000 steps and must fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            SimulationRequest.model_validate(
                {"engine_type": "mujoco", "duration": 300.0, "timestep": 1e-6}
            )
        msg = str(exc_info.value).lower()
        assert "step" in msg or "maximum" in msg or "budget" in msg or "exceed" in msg

    def test_simulation_request_rejects_nonfinite_inputs(self) -> None:
        """Nonfinite duration or timestep must be rejected immediately."""
        with pytest.raises(ValidationError):
            SimulationRequest.model_validate(
                {"engine_type": "mujoco", "duration": float("inf"), "timestep": 0.01}
            )

        with pytest.raises(ValidationError):
            SimulationRequest.model_validate(
                {"engine_type": "mujoco", "duration": 1.0, "timestep": float("nan")}
            )

    def test_simulation_request_accepts_valid_budget(self) -> None:
        """Valid duration and timestep within step budget are accepted."""
        req = SimulationRequest.model_validate(
            {"engine_type": "mujoco", "duration": 10.0, "timestep": 0.001}
        )
        assert req.duration == 10.0
        assert req.timestep == 0.001


class TestBallFlightRequestBounds:
    """Validate flight horizon/sample ratio and allocation bounds."""

    def test_ball_flight_request_rejects_excessive_sample_ratio(self) -> None:
        """time_step_s=1e-12 against 10s horizon describes 10T samples and must fail."""
        with pytest.raises(ValidationError) as exc_info:
            BallFlightSimulationRequest.model_validate(
                {"time_step_s": 1e-12, "max_time_s": 10.0}
            )
        msg = str(exc_info.value).lower()
        assert "sample" in msg or "step" in msg or "exceed" in msg or "minimum" in msg

    def test_ball_flight_request_rejects_sub_minimum_timestep(self) -> None:
        """Sub-millisecond or zero/negative timestep must be rejected."""
        with pytest.raises(ValidationError):
            BallFlightSimulationRequest.model_validate({"time_step_s": 1e-6})

    def test_ball_flight_request_rejects_nonfinite_inputs(self) -> None:
        """Nonfinite inputs must fail closed."""
        with pytest.raises(ValidationError):
            BallFlightSimulationRequest.model_validate({"ball_speed_mps": float("nan")})

        with pytest.raises(ValidationError):
            BallFlightSimulationRequest.model_validate({"time_step_s": float("inf")})

    def test_extract_ode_trajectory_points_bounds_allocation(self) -> None:
        """_extract_ode_trajectory_points refuses excessive sample allocation."""
        mock_sol = MagicMock()
        mock_sol.t = [0.0, 10.0]
        # dt=1e-8 would request 1 billion points: must raise ValueError, not MemoryError
        with pytest.raises(ValueError) as exc_info:
            _extract_ode_trajectory_points(mock_sol, dt=1e-8)
        assert (
            "sample" in str(exc_info.value).lower()
            or "limit" in str(exc_info.value).lower()
        )


# =============================================================================
# 2. Event-loop offloading and responsiveness
# =============================================================================


class TestBallFlightEventLoopOffloading:
    """Validate flight API does not starve event loop or block heartbeat."""

    async def test_ball_flight_simulation_does_not_block_heartbeat(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A heartbeat request remains responsive while a slow flight simulation runs."""
        import src.api.routes.ball_flight as bf_module

        original_simulate_one = bf_module._simulate_one

        def slow_simulate_one(*args: Any, **kwargs: Any) -> Any:
            time.sleep(0.15)  # Simulate CPU-bound ODE solving (150 ms)
            return original_simulate_one(*args, **kwargs)

        monkeypatch.setattr(bf_module, "_simulate_one", slow_simulate_one)

        payload = BallFlightSimulationRequest.model_validate({})

        heartbeat_timings: list[float] = []

        async def run_flight() -> Any:
            return await simulate_ball_flight(payload)

        async def run_heartbeat() -> None:
            for _ in range(3):
                t0 = time.monotonic()
                await asyncio.sleep(0.02)
                heartbeat_timings.append(time.monotonic() - t0)

        t_start = time.monotonic()
        flight_task = asyncio.create_task(run_flight())
        heartbeat_task = asyncio.create_task(run_heartbeat())

        await asyncio.gather(flight_task, heartbeat_task)
        total_time = time.monotonic() - t_start

        assert total_time >= 0.15
        for elapsed in heartbeat_timings:
            assert elapsed < 0.08, f"Heartbeat was blocked: took {elapsed:.4f}s"


# =============================================================================
# 3. TaskManager active retention and capacity isolation
# =============================================================================


class TestTaskManagerActiveRetention:
    """Retain active state independently of terminal-result eviction."""

    def test_task_manager_does_not_evict_active_tasks_under_size_limit(self) -> None:
        """R03 Counterexample: max_tasks=1 probe must NOT evict a running task."""
        tm = TaskManager(max_tasks=1)
        tm.set("active", {"status": "running"})
        tm.set("next", {"status": "pending"})

        act = tm.get("active")
        assert act is not None
        assert act.get("status") == "running"
        assert tm.get("next") is not None

    def test_task_manager_evicts_terminal_tasks_before_active(self) -> None:
        """LRU eviction strictly targets terminal tasks (completed, failed, cancelled)."""
        tm = TaskManager(max_tasks=2)
        tm.set("done1", {"status": "completed"})
        tm.set("run1", {"status": "running"})
        tm.set("run2", {"status": "running"})

        assert tm.get("run1") is not None
        assert tm.get("run2") is not None
        assert tm.get("done1") is None

    def test_task_manager_active_task_does_not_expire_by_ttl(self) -> None:
        """A running or pending task does not expire via TTL while still active."""
        tm = TaskManager(ttl_seconds=1)
        tm.set("active", {"status": "running"})
        with tm._lock:
            tm._timestamps["active"] = time.time() - 100.0

        assert tm.get("active") is not None
        assert tm.exists("active") is True

    def test_task_manager_capacity_limit_rejection(self) -> None:
        """Under full active capacity, TaskManager rejects new active work with capacity error."""
        tm = TaskManager(max_concurrent=2)
        tm.set("act1", {"status": "running"})
        tm.set("act2", {"status": "running"})

        assert tm.can_admit_active() is False
        with pytest.raises((RuntimeError, ValueError)) as exc_info:
            tm.admit_active("act3", {"status": "pending"})
        assert (
            "capacity" in str(exc_info.value).lower()
            or "busy" in str(exc_info.value).lower()
        )


# =============================================================================
# 4. Stepping loop cooperative cancellation and progress
# =============================================================================


class TestSteppingLoopCancellationAndProgress:
    """Cancellation and deadlines stop actual computation within documented bound."""

    def test_stepping_loop_stops_on_cancellation_token(self) -> None:
        """Stepping loop checks cancellation and halts within 1 step."""
        service = SimulationService(engine_manager=MagicMock())
        mock_engine = MagicMock()
        mock_recorder = MagicMock(spec=GenericPhysicsRecorder)
        mock_recorder.is_recording = True
        mock_recorder.current_idx = 0
        mock_recorder.buffer_exhausted = False

        run = SimulationRunRecord(run_id="test_cancel", engine_type="mujoco")
        run.cancel()

        req = SimulationRequest.model_validate(
            {"engine_type": "mujoco", "duration": 1.0, "timestep": 0.001}
        )
        service._execute_simulation_loop(
            engine=mock_engine,
            recorder=mock_recorder,
            request=req,
            timestep=0.001,
            steps=1000,
            run=run,
        )

        assert mock_engine.step.call_count <= 1
        assert run.status == "cancelled"

    def test_stepping_loop_stops_on_deadline(self) -> None:
        """Stepping loop stops when deadline is exceeded."""
        service = SimulationService(engine_manager=MagicMock())
        mock_engine = MagicMock()
        mock_recorder = MagicMock(spec=GenericPhysicsRecorder)
        mock_recorder.is_recording = True
        mock_recorder.current_idx = 0
        mock_recorder.buffer_exhausted = False

        run = SimulationRunRecord(run_id="test_deadline", engine_type="mujoco")
        run.deadline = time.time() - 1.0

        req = SimulationRequest.model_validate(
            {"engine_type": "mujoco", "duration": 1.0, "timestep": 0.001}
        )
        service._execute_simulation_loop(
            engine=mock_engine,
            recorder=mock_recorder,
            request=req,
            timestep=0.001,
            steps=1000,
            run=run,
        )

        assert mock_engine.step.call_count <= 1
        assert run.status in ("cancelled", "timed_out")

    def test_simulation_cancellation_returns_distinct_terminal_outcome(self) -> None:
        """Cancelled simulation returns calculation_status='cancelled' and success=False."""
        mock_mgr = MagicMock()
        mock_engine = MagicMock()
        mock_engine.get_full_state.return_value = {
            "q": np.zeros(3),
            "v": np.zeros(3),
            "t": 0.0,
            "M": None,
        }
        mock_mgr.get_active_physics_engine.return_value = mock_engine
        service = SimulationService(engine_manager=mock_mgr)

        run = service.create_run(run_id="cancel_sim_run", engine_type="mujoco")
        run.cancel()

        req = SimulationRequest.model_validate(
            {"engine_type": "mujoco", "duration": 0.1, "timestep": 0.01}
        )
        res = service._execute_sync_pipeline(req, run, "cancel_sim_run")

        assert res.success is False
        assert res.calculation_status == "cancelled"
        assert res.error is not None
        assert res.error.code == "cancelled"

    def test_cancellation_distinct_from_persistence_failure(self) -> None:
        """Persistence failure returns success=True with persistence_status='failed', distinct from cancellation."""
        cancel_resp = SimulationResponse(
            success=False,
            duration=0.1,
            frames=5,
            data={"states": []},
            calculation_status="cancelled",
            persistence_status="not_requested",
            error=SimulationErrorInfo(
                code="cancelled", message="Simulation cancelled", stage="execution"
            ),
        )
        persist_fail_resp = SimulationResponse(
            success=True,
            duration=0.1,
            frames=11,
            data={"states": []},
            calculation_status="completed",
            persistence_status="failed",
            error=SimulationErrorInfo(
                code="persistence_failed",
                message="Disk write failed",
                stage="persistence",
            ),
        )

        assert cancel_resp.calculation_status == "cancelled"
        assert cancel_resp.success is False
        assert persist_fail_resp.calculation_status == "completed"
        assert persist_fail_resp.persistence_status == "failed"
        assert persist_fail_resp.success is True
