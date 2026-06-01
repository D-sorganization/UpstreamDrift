"""Regression test for non-blocking /simulate (issue #6988).

The CPU-bound simulation loop must run in a worker thread so the FastAPI
event loop stays responsive; previously ``run_simulation`` executed the
synchronous stepping loop inline and froze the worker for the whole sim.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.api.services.simulation_service import SimulationService

pytestmark = pytest.mark.anyio


@pytest.fixture(scope="module")
def anyio_backend() -> str:
    return "asyncio"


def _make_request() -> Any:
    req = MagicMock()
    req.duration = 1.0
    req.timestep = 0.5
    req.analysis_config = None
    req.control_inputs = None
    req.engine_type = "mujoco"
    req.model_path = None
    req.initial_state = None
    return req


async def test_run_simulation_offloads_to_thread() -> None:
    """The blocking pipeline runs off the event loop, not on the loop thread."""
    service = SimulationService(engine_manager=MagicMock())

    loop_thread = threading.get_ident()
    worker_thread: dict[str, int] = {}

    def _fake_sync(_request: Any) -> Any:
        worker_thread["id"] = threading.get_ident()
        result = MagicMock()
        result.success = True
        return result

    service._run_simulation_sync = _fake_sync  # type: ignore[assignment]

    await service.run_simulation(_make_request())

    assert worker_thread["id"] != loop_thread, (
        "simulation must execute in a worker thread, not the event loop"
    )


async def test_event_loop_stays_responsive_during_simulation() -> None:
    """A concurrent coroutine progresses while a simulation is 'running'."""
    service = SimulationService(engine_manager=MagicMock())

    sim_running = threading.Event()
    release = threading.Event()

    def _blocking_sync(_request: Any) -> Any:
        sim_running.set()
        # Block the worker thread; the event loop must NOT be blocked.
        assert release.wait(timeout=5.0)
        result = MagicMock()
        result.success = True
        return result

    service._run_simulation_sync = _blocking_sync  # type: ignore[assignment]

    sim_task = asyncio.create_task(service.run_simulation(_make_request()))

    # Wait — without ever blocking the loop — for the worker thread to enter
    # the blocking sim. If run_simulation ran inline this await would never
    # get control back until the sim finished.
    ticks = 0
    while not sim_running.is_set() and ticks < 200:
        await asyncio.sleep(0.01)
        ticks += 1
    assert sim_running.is_set(), "worker thread never started"

    # The loop kept ticking concurrently while the worker thread was blocked.
    assert ticks >= 1

    release.set()
    await sim_task
