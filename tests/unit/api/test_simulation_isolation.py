"""Tests for R02 simulation state isolation per run.

Acceptance criteria:
- A deterministic barrier test proves two overlapping preparations cannot acquire
  each other's engine/model/configuration.
- Overlapping REST/REST and REST/WebSocket requests either receive an explicit
  busy/queue response or independent run IDs, clocks, controls, results, and recording state.
- Analysis/export references a selected run ID; completion of another run cannot
  change the analyzed dataset.
- Engine cleanup occurs once after its owner finishes or cancels; one run cannot
  unload another's engine.
"""

from __future__ import annotations

import concurrent.futures
from enum import Enum
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.api.models.requests import SimulationRequest
from src.api.services.simulation_service import SimulationService, SimulationStats
from src.shared.python.engine_core.interfaces import PhysicsEngine

pytestmark = [pytest.mark.anyio, pytest.mark.unit]


class MockEngineType(Enum):
    """Mock EngineType enum."""

    MUJOCO = "mujoco"
    DRAKE = "drake"
    PINOCCHIO = "pinocchio"

    @classmethod
    def _missing_(cls, value: object) -> MockEngineType | None:
        if isinstance(value, str):
            for member in cls:
                if member.name == value or member.value == value.lower():
                    return member
        return None


class FakeEngine:
    """Fake physics engine tracking its own identity, loaded model, state, and cleanup."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.engine_type = name
        self.model_path: str | None = None
        self.positions: list[float] = []
        self.velocities: list[float] = []
        self.controls: list[float] = []
        self.close_count: int = 0
        self.time: float = 0.0

    def load_from_path(self, path: str) -> None:
        self.model_path = path

    def set_state(self, positions: list[float], velocities: list[float]) -> None:
        self.positions = list(positions)
        self.velocities = list(velocities)

    def set_control(self, control: Any) -> None:
        self.controls = list(control)

    def step(self, dt: float) -> None:
        self.time += dt

    def get_joint_names(self) -> list[str]:
        return [f"{self.name}_j1", f"{self.name}_j2"]

    def close(self) -> None:
        self.close_count += 1


@pytest.fixture(scope="module")
def anyio_backend() -> str:
    return "asyncio"


class BarrierEngineManager:
    """Injected engine manager with a 2-party barrier inside engine creation/loading.

    This reproduces the controlled interleaving test from the R02 review brief.
    """

    def __init__(self, barrier: threading.Barrier) -> None:
        self.barrier = barrier
        self.active_physics_engine: FakeEngine | None = None
        self.created_engines: dict[str, FakeEngine] = {}

    def create_engine(self, engine_type: Any) -> FakeEngine:
        name = engine_type.value if hasattr(engine_type, "value") else str(engine_type)
        engine = FakeEngine(name)
        self.created_engines[name] = engine
        self.active_physics_engine = engine
        self.barrier.wait(timeout=5.0)
        return engine

    def _load_engine(self, engine_type: Any) -> FakeEngine:
        return self.create_engine(engine_type)

    def get_active_physics_engine(self) -> FakeEngine | None:
        return self.active_physics_engine


class TestR02DeterministicBarrierIsolation:
    """Criterion 1: Deterministic barrier test for overlapping preparations."""

    def test_overlapping_preparations_cannot_acquire_each_others_engine_model_config(
        self,
    ) -> None:
        """Prove that concurrent preparations do not cross-contaminate engine/model/state."""
        barrier = threading.Barrier(2)
        manager = BarrierEngineManager(barrier)
        service = SimulationService(manager)  # type: ignore[arg-type]

        req1 = SimulationRequest.model_validate(
            {
                "engine_type": "mujoco",
                "model_path": "/path/to/mujoco_model.xml",
                "initial_state": {"positions": [1.0, 2.0], "velocities": [0.1, 0.2]},
                "duration": 0.01,
                "timestep": 0.001,
            }
        )
        req2 = SimulationRequest.model_validate(
            {
                "engine_type": "drake",
                "model_path": "/path/to/drake_model.urdf",
                "initial_state": {"positions": [3.0, 4.0], "velocities": [0.3, 0.4]},
                "duration": 0.01,
                "timestep": 0.001,
            }
        )

        with patch("src.api.services.simulation_service.EngineType", MockEngineType):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future1 = executor.submit(service._prepare_engine, req1, "run-1")
                future2 = executor.submit(service._prepare_engine, req2, "run-2")
                engine1 = future1.result(timeout=10.0)
                engine2 = future2.result(timeout=10.0)

        assert engine1 is not engine2
        assert engine1.name == "mujoco"
        assert engine1.model_path == "/path/to/mujoco_model.xml"
        assert engine1.positions == [1.0, 2.0]
        assert engine1.velocities == [0.1, 0.2]

        assert engine2.name == "drake"
        assert engine2.model_path == "/path/to/drake_model.urdf"
        assert engine2.positions == [3.0, 4.0]
        assert engine2.velocities == [0.3, 0.4]


class TestR02IndependentRunStateAndBusyResponse:
    """Criterion 2: Independent run IDs, clocks, controls, results, and recording state or busy."""

    def test_independent_run_sessions_and_stats(self) -> None:
        """Each run must own its independent stats, clocks, speed factor, and recorder."""
        manager = MagicMock()
        service = SimulationService(manager)

        run1 = service.create_run(run_id="run-alpha", engine_type="mujoco")
        run2 = service.create_run(run_id="run-beta", engine_type="drake")

        assert run1.run_id == "run-alpha"
        assert run2.run_id == "run-beta"
        assert run1.stats is not run2.stats

        run1.stats.speed_factor = 2.5
        run1.stats.frame_count = 50
        run2.stats.speed_factor = 0.5
        run2.stats.frame_count = 120

        assert service.get_run_stats("run-alpha").speed_factor == 2.5
        assert service.get_run_stats("run-alpha").frame_count == 50
        assert service.get_run_stats("run-beta").speed_factor == 0.5
        assert service.get_run_stats("run-beta").frame_count == 120

    def test_explicit_busy_response_in_single_run_containment(self) -> None:
        """Under single-run containment mode, an overlapping preparation returns busy."""
        manager = MagicMock()
        service = SimulationService(manager)
        service.single_run_mode = True

        run1 = service.create_run(run_id="active-run", engine_type="mujoco")
        run1.status = "running"

        req = SimulationRequest.model_validate(
            {"engine_type": "drake", "duration": 0.01}
        )
        with patch("src.api.services.simulation_service.EngineType", MockEngineType):
            response = service._run_simulation_sync(req, run_id="contending-run")

        assert response.success is False
        assert response.error is not None
        assert response.error.code == "busy"
        assert response.error.retriable is True
        assert response.error.stage == "preparation"


class TestR02AnalysisExportRunIdImmutability:
    """Criterion 3: Analysis and export reference selected run ID; new run cannot mutate dataset."""

    def test_analysis_and_export_reference_selected_run_id(self) -> None:
        """Completion of run 2 does not change the analyzed dataset or recorder of run 1."""
        manager = MagicMock()
        service = SimulationService(manager)

        rec1 = MagicMock()
        rec1.current_idx = 10
        rec1.engine = FakeEngine("mujoco")
        rec1.get_time_series = MagicMock(
            return_value=(np.array([0.0, 0.01]), np.array([[1.0], [1.1]]))
        )
        rec1.get_data_dict = MagicMock(
            return_value={"times": [0.0, 0.01], "q": [1.0, 1.1]}
        )

        rec2 = MagicMock()
        rec2.current_idx = 20
        rec2.engine = FakeEngine("drake")
        rec2.get_time_series = MagicMock(
            return_value=(np.array([0.0, 0.02]), np.array([[9.0], [9.9]]))
        )
        rec2.get_data_dict = MagicMock(
            return_value={"times": [0.0, 0.02], "q": [9.0, 9.9]}
        )

        service.register_completed_run(
            run_id="run-001",
            engine=rec1.engine,
            recorder=rec1,
            meta={"engine": "mujoco", "model": "m1.xml"},
        )

        service.register_completed_run(
            run_id="run-002",
            engine=rec2.engine,
            recorder=rec2,
            meta={"engine": "drake", "model": "m2.urdf"},
        )

        # Querying run 1 returns run 1's recorder and joint names even after run 2 finished
        session1 = service.get_session_recording(run_id="run-001")
        assert session1 is not None
        rec_retrieved, meta_retrieved = session1
        assert rec_retrieved is rec1
        assert meta_retrieved["engine"] == "mujoco"

        # Support check for run 1 uses run 1's engine
        support1 = service.describe_counterfactual_support(run_id="run-001")
        assert support1["engine"] == "mujoco"

        support2 = service.describe_counterfactual_support(run_id="run-002")
        assert support2["engine"] == "drake"


class TestR02EngineCleanupIsolation:
    """Criterion 4: Cleanup occurs once per owner; one run cannot unload another's engine."""

    def test_cleanup_occurs_once_and_does_not_unload_other_engines(self) -> None:
        """Finishing/cancelling run 1 unloads engine 1 once without unloading engine 2."""
        manager = MagicMock()
        service = SimulationService(manager)

        engine1 = FakeEngine("mujoco")
        engine2 = FakeEngine("drake")

        run1 = service.create_run(run_id="run-1", engine_type="mujoco")
        run1.engine = engine1

        run2 = service.create_run(run_id="run-2", engine_type="drake")
        run2.engine = engine2

        # Clean up run 1
        service.cleanup_run_engine("run-1")
        assert engine1.close_count == 1
        assert engine2.close_count == 0

        # Repeated cleanup on run 1 is idempotent
        service.cleanup_run_engine("run-1")
        assert engine1.close_count == 1
        assert engine2.close_count == 0

        # Clean up run 2
        service.cleanup_run_engine("run-2")
        assert engine1.close_count == 1
        assert engine2.close_count == 1


class TestR02RestWebSocketIsolation:
    """Criterion 2 & 4: REST / WebSocket isolation."""

    def test_websocket_and_rest_isolated_stats_and_speed(self) -> None:
        """WebSocket speed adjustments and stats resets do not mutate REST run stats."""
        import types
        from src.api.routes.simulation_ws import (
            _apply_set_speed,
            _reset_simulation_stats,
            _resolve_sim_stats,
        )

        manager = MagicMock()
        service = SimulationService(manager)

        # REST run
        rest_run = service.create_run(run_id="rest-run-123", engine_type="mujoco")
        rest_run.stats.speed_factor = 1.0
        rest_run.stats.frame_count = 85

        # WebSocket mock
        ws = MagicMock()
        ws.state = types.SimpleNamespace()
        ws.state.sim_stats = SimulationStats(speed_factor=1.0, frame_count=10)
        ws.app = types.SimpleNamespace(
            state=types.SimpleNamespace(simulation_service=service)
        )

        # Resolve stats should resolve WebSocket's own stats
        ws_stats = _resolve_sim_stats(ws)
        assert ws_stats is ws.state.sim_stats
        assert ws_stats is not rest_run.stats

        # Changing speed on WebSocket updates WebSocket stats only
        _apply_set_speed(ws, config={}, msg={"speed_factor": 3.0})
        assert ws.state.sim_stats.speed_factor == 3.0
        assert rest_run.stats.speed_factor == 1.0
        assert service.get_run_stats("rest-run-123").speed_factor == 1.0

        # Resetting WebSocket stats zeroes WebSocket frame count, not REST run
        _reset_simulation_stats(ws, config={})
        assert ws.state.sim_stats.frame_count == 0
        assert rest_run.stats.frame_count == 85
        assert service.get_run_stats("rest-run-123").frame_count == 85
