"""Behavioral unit tests for R11: Truthful Simulation Horizons and Sampling Clocks.

Acceptance criteria (Issue #11151 / R11):
- Non-divisible, divisible, sub-step and floating-point-boundary durations produce truthful clocks
  in REST, WebSocket, recorder and export.
- A deterministic engine recording every dt proves the reported final state time equals the sum of executed steps.
- Cross-engine tests state which backends accept variable final steps; no silent per-backend reinterpretation.
- The state timestamp remains consistent through trajectory interchange and live analysis.
- Requested duration, integrated duration, step count and retained sample count are separated.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.api.models.requests import SimulationRequest
from src.api.models.responses import SimulationResponse
from src.api.services.simulation_service import SimulationService
from src.shared.python.dashboard.recorder import GenericPhysicsRecorder
from src.shared.python.engine_core.mock_engine import MockPhysicsEngine
from src.shared.python.engine_core.simulation_timing import (
    SimulationTimingPlan,
    compute_simulation_timing,
    engine_supports_variable_step,
)

pytestmark = [pytest.mark.unit]


class TestSimulationTimingPlanCalculations:
    """Contract tests for compute_simulation_timing."""

    def test_divisible_boundary_exact_steps(self) -> None:
        """Divisible durations produce exact integer steps with no remainder."""
        plan = compute_simulation_timing(duration=0.05, timestep=0.01)
        assert plan.is_divisible is True
        assert plan.step_count == 5
        assert len(plan.step_sizes) == 5
        assert all(math.isclose(dt, 0.01, abs_tol=1e-12) for dt in plan.step_sizes)
        assert plan.integrated_duration == pytest.approx(0.05)
        assert plan.requested_duration == 0.05
        assert plan.retained_samples == 6
        assert plan.has_remainder_step is False
        assert plan.remainder_dt is None

    def test_floating_point_boundaries_avoid_int_truncation(self) -> None:
        """Floating-point boundaries (e.g. 0.03/0.01=2.99999996) must not truncate steps."""
        test_cases = [
            (0.03, 0.01, 3),
            (0.07, 0.01, 7),
            (0.14, 0.01, 14),
            (0.29, 0.01, 29),
            (0.57, 0.01, 57),
            (0.70, 0.01, 70),
        ]
        for duration, timestep, expected_steps in test_cases:
            plan = compute_simulation_timing(duration=duration, timestep=timestep)
            assert plan.is_divisible is True
            assert plan.step_count == expected_steps, (
                f"Failed for {duration}/{timestep}: expected {expected_steps}, got {plan.step_count}"
            )
            assert plan.integrated_duration == pytest.approx(duration)
            assert plan.has_remainder_step is False

    def test_non_divisible_with_remainder_step(self) -> None:
        """Non-divisible durations take a supported remainder step when permitted."""
        # 0.025s at 0.01s -> 2 full steps of 0.01s + 1 remainder step of 0.005s
        plan = compute_simulation_timing(
            duration=0.025, timestep=0.01, allow_remainder_step=True
        )
        assert plan.is_divisible is False
        assert plan.has_remainder_step is True
        assert plan.step_count == 3
        assert len(plan.step_sizes) == 3
        assert plan.step_sizes[0] == pytest.approx(0.01)
        assert plan.step_sizes[1] == pytest.approx(0.01)
        assert plan.step_sizes[2] == pytest.approx(0.005)
        assert plan.remainder_dt == pytest.approx(0.005)
        assert plan.integrated_duration == pytest.approx(0.025)
        assert plan.requested_duration == 0.025
        assert plan.retained_samples == 4

    def test_non_divisible_fixed_step_reports_truthful_horizon(self) -> None:
        """When variable final steps are disallowed, reports actual integrated horizon."""
        plan = compute_simulation_timing(
            duration=0.025, timestep=0.01, allow_remainder_step=False
        )
        assert plan.is_divisible is False
        assert plan.has_remainder_step is False
        assert plan.step_count == 2
        assert len(plan.step_sizes) == 2
        assert all(dt == 0.01 for dt in plan.step_sizes)
        # Truthful clock: actual integrated horizon is 0.02s, NOT the requested 0.025s!
        assert plan.integrated_duration == pytest.approx(0.02)
        assert plan.requested_duration == 0.025
        assert plan.retained_samples == 3

    def test_sub_step_duration_with_variable_step(self) -> None:
        """Sub-step durations (duration < timestep) execute a single bounded step."""
        plan = compute_simulation_timing(
            duration=0.0005, timestep=0.001, allow_remainder_step=True
        )
        assert plan.is_divisible is False
        assert plan.has_remainder_step is True
        assert plan.step_count == 1
        assert len(plan.step_sizes) == 1
        assert plan.step_sizes[0] == pytest.approx(0.0005)
        assert plan.integrated_duration == pytest.approx(0.0005)
        assert plan.retained_samples == 2

    def test_invalid_timing_inputs_rejected(self) -> None:
        """Non-positive or non-finite timing parameters must fail-closed."""
        with pytest.raises(ValueError, match="Duration must be positive"):
            compute_simulation_timing(duration=-0.1, timestep=0.01)
        with pytest.raises(ValueError, match="Timestep must be positive"):
            compute_simulation_timing(duration=0.1, timestep=-0.01)
        with pytest.raises(ValueError, match="Duration must be finite"):
            compute_simulation_timing(duration=float("nan"), timestep=0.01)
        with pytest.raises(ValueError, match="Timestep must be finite"):
            compute_simulation_timing(duration=0.1, timestep=float("inf"))


class TestDeterministicEngineClockAccumulation:
    """Verifies that a deterministic engine recording every dt proves final time == sum(dt)."""

    def test_engine_recording_proves_final_time_equals_sum_of_steps(self) -> None:
        engine = MockPhysicsEngine()
        recorder = GenericPhysicsRecorder(engine)
        recorder.start()

        # Step with mixed dt: 0.01, 0.01, 0.005
        step_sizes = (0.01, 0.01, 0.005)
        recorder.record_step(control_input=None)  # t=0

        for dt in step_sizes:
            engine.step(dt)
            recorder.record_step(control_input=None)

        recorder.stop()
        data = recorder.get_data_dict()
        times = data["times"]

        assert len(times) == len(step_sizes) + 1
        assert times[0] == pytest.approx(0.0)
        assert times[1] == pytest.approx(0.01)
        assert times[2] == pytest.approx(0.02)
        assert times[3] == pytest.approx(0.025)
        assert times[-1] == pytest.approx(sum(step_sizes))
        assert engine.get_state_dict()["time"] == pytest.approx(sum(step_sizes))


class TestRestSimulationSeparatedDurationsAndTruthfulClocks:
    """Tests REST simulation execution with R11 timing fields."""

    @pytest.fixture
    def service(self) -> SimulationService:
        mock_mgr = MagicMock()
        mock_mgr.create_engine.side_effect = lambda *args, **kwargs: MockPhysicsEngine()
        mock_mgr.get_active_physics_engine.side_effect = lambda *args, **kwargs: (
            MockPhysicsEngine()
        )
        return SimulationService(mock_mgr)

    @pytest.mark.anyio
    async def test_rest_simulation_non_divisible_with_remainder(
        self, service: SimulationService
    ) -> None:
        """REST simulation executes remainder step and reports separated truthful horizons."""
        request = SimulationRequest.model_validate(
            {
                "engine_type": "pendulum",
                "duration": 0.025,
                "timestep": 0.01,
            }
        )

        response = await service.run_simulation(request)
        assert response.success is True
        # Check separated duration and horizon fields
        assert response.requested_duration == 0.025
        assert response.integrated_duration == pytest.approx(0.025)
        assert response.duration == pytest.approx(0.025)
        assert response.step_count == 3
        assert response.retained_samples == 4
        assert response.frames == 4

        # Truthful clock: final state timestamp matches integrated duration
        times = response.data["times"]
        assert len(times) == 4
        assert times[0] == pytest.approx(0.0)
        assert times[-1] == pytest.approx(0.025)
        assert times[-1] == pytest.approx(response.integrated_duration)

    @pytest.mark.anyio
    async def test_rest_simulation_sub_step_duration(
        self, service: SimulationService
    ) -> None:
        """REST simulation supports sub-step duration without raising timestep-exceeds error."""
        request = SimulationRequest.model_validate(
            {
                "engine_type": "pendulum",
                "duration": 0.0005,
                "timestep": 0.001,
            }
        )

        response = await service.run_simulation(request)
        assert response.success is True
        assert response.requested_duration == 0.0005
        assert response.integrated_duration == pytest.approx(0.0005)
        assert response.duration == pytest.approx(0.0005)
        assert response.step_count == 1
        assert response.retained_samples == 2
        assert response.frames == 2
        assert response.data["times"][-1] == pytest.approx(0.0005)

    @pytest.mark.anyio
    async def test_rest_simulation_fixed_step_non_divisible_truthful_clock(
        self, service: SimulationService
    ) -> None:
        """When fixed step mode is requested, reports actual integrated horizon without relabeling."""
        request = SimulationRequest.model_validate(
            {
                "engine_type": "pendulum",
                "duration": 0.025,
                "timestep": 0.01,
                "allow_remainder_step": False,
            }
        )

        response = await service.run_simulation(request)
        assert response.success is True
        assert response.requested_duration == 0.025
        # Truthful clock: executed 2 steps of 0.01 = 0.02s
        assert response.integrated_duration == pytest.approx(0.02)
        assert response.duration == pytest.approx(0.02)
        assert response.step_count == 2
        assert response.retained_samples == 3
        assert response.frames == 3
        assert response.data["times"][-1] == pytest.approx(0.02)
        # Crucial: never relabel state with earlier requested duration
        assert response.data["times"][-1] != 0.025


class TestCrossEngineVariableStepBackendContracts:
    """Verifies that cross-engine backends explicitly declare variable final step support."""

    def test_backend_variable_step_support_declared(self) -> None:
        """Backends state whether they accept variable final steps (no silent reinterpretation)."""
        from src.shared.python.simulation_backends.model_params import GolfModelParams
        from src.shared.python.simulation_backends.ode_backend import ODEBackend

        # ODE backend supports variable dt
        ode = ODEBackend(params=GolfModelParams.default())
        assert engine_supports_variable_step(ode) is True

        # Mock engine supports variable dt
        mock_eng = MockPhysicsEngine()
        assert engine_supports_variable_step(mock_eng) is True

        # MuJoCo backend supports variable dt
        try:
            from src.shared.python.simulation_backends.mujoco_backend import (
                MuJoCoBackend,
            )

            mj = MuJoCoBackend(params=GolfModelParams.default())
            assert engine_supports_variable_step(mj) is True
        except ImportError:
            pass  # Optional dependency on machines without mujoco

        # A dummy fixed-step engine declaring supports_variable_step=False
        class FixedStepEngine:
            timestep: float = 0.01
            supports_variable_step: bool = False

            def step(self, dt: float | None = None) -> None:
                pass

        fixed_eng = FixedStepEngine()
        assert engine_supports_variable_step(fixed_eng) is False


class TestWebSocketTruthfulClocks:
    """Tests WebSocket simulation loop timing and non-clamping contracts."""

    @pytest.mark.anyio
    async def test_ws_simulation_loop_truthful_times_without_clamping(self) -> None:
        """WebSocket loop reports exact integrated horizon without min(duration, frame*timestep) clamping."""
        from src.api.routes import simulation_ws as ws_module

        class FakeWsEngine:
            def __init__(self) -> None:
                self.time = 0.0
                self.steps: list[float] = []

            def step(self, dt: float) -> None:
                self.time += dt
                self.steps.append(dt)

            def get_state(self) -> tuple[np.ndarray, np.ndarray]:
                return np.zeros(2), np.zeros(2)

            def get_time(self) -> float:
                return self.time

        class RecordingWs:
            def __init__(self) -> None:
                self.sent: list[dict[str, Any]] = []

            async def send_json(self, payload: dict[str, Any]) -> None:
                self.sent.append(payload)

            async def receive_json(self) -> dict[str, Any]:
                import asyncio

                await asyncio.sleep(10)
                return {}

        engine = FakeWsEngine()
        ws = RecordingWs()
        config = {
            "duration": 0.025,
            "timestep": 0.01,
            "speed_factor": 1000.0,
        }

        frame, elapsed = await ws_module._run_simulation_loop(ws, engine, config)
        assert frame == 3
        assert elapsed == pytest.approx(0.025)
        assert engine.steps == [0.01, 0.01, 0.005]

        # Verify emitted frame times are monotonically truthful
        frame_messages = [msg for msg in ws.sent if "frame" in msg]
        assert len(frame_messages) > 0
        final_frame_msg = frame_messages[-1]
        assert final_frame_msg["time"] == pytest.approx(0.025)
