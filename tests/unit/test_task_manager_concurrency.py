"""Regression test for issues #3506 and #4843.

Issue #3506 originally required asyncio-native locking inside an async-only
TaskManager. Issue #4843 restored a synchronous + dict-like compatibility
surface, so the data-store lock is now a ``threading.RLock`` (held only for
in-memory dict mutations, never across an ``await``) while the engine
concurrency primitive remains an ``asyncio.Semaphore``. The deadlock scenario
guarded by #3506 (mixing a sync lock with an async semaphore *across awaits*)
no longer applies because the two primitives are never composed.
"""

from __future__ import annotations

import asyncio
import threading

import pytest
from src.api.task_manager import TaskManager

_RLOCK_TYPE = type(threading.RLock())


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lock_and_semaphore_primitives() -> None:
    """Regression guard: locking primitives match the #4843 contract."""
    tm = TaskManager(max_concurrent=2)
    try:
        # Data-store lock is a threading.RLock to support sync callers (#4843).
        assert isinstance(tm._lock, _RLOCK_TYPE), (
            "TaskManager._lock must be threading.RLock for the sync API (#4843)"
        )
        assert isinstance(tm._engine_semaphore, asyncio.Semaphore), (
            "TaskManager._engine_semaphore must be asyncio.Semaphore (#3506)"
        )
        assert isinstance(tm.engine_semaphore, asyncio.Semaphore)
    finally:
        await tm.shutdown()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_concurrent_set_get_active_count_no_deadlock() -> None:
    """Many concurrent set/get/active_count calls must not deadlock."""
    tm = TaskManager(max_concurrent=2)
    try:
        n_tasks = 20

        async def writer(i: int) -> None:
            tm.set(f"task-{i}", {"status": "pending", "value": i})

        async def reader(i: int) -> dict | None:
            return tm.get(f"task-{i}")

        async def counter() -> int:
            return tm.active_count()

        # First wave: write all tasks concurrently.
        await asyncio.wait_for(
            asyncio.gather(*(writer(i) for i in range(n_tasks))),
            timeout=10.0,
        )

        # Second wave: interleave read/write/count operations.
        ops: list = []
        for i in range(n_tasks):
            ops.append(writer(i))
            ops.append(reader(i))
            ops.append(counter())

        results = await asyncio.wait_for(asyncio.gather(*ops), timeout=10.0)

        # Validate consistency: all reads return our payloads, counts are sane.
        for i in range(n_tasks):
            read_result = results[3 * i + 1]
            assert read_result is not None
            assert read_result["value"] == i

            count = results[3 * i + 2]
            assert isinstance(count, int)
            assert 0 <= count <= n_tasks

        final_count = tm.active_count()
        assert final_count == n_tasks
    finally:
        await tm.shutdown()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_semaphore_enforces_max_concurrent() -> None:
    """Semaphore must cap concurrent engine acquisitions at max_concurrent."""
    tm = TaskManager(max_concurrent=2)
    try:
        in_flight = 0
        peak = 0
        lock = asyncio.Lock()

        async def acquire_and_track() -> None:
            nonlocal in_flight, peak
            async with tm.engine_semaphore:
                async with lock:
                    in_flight += 1
                    peak = max(peak, in_flight)
                await asyncio.sleep(0.01)
                async with lock:
                    in_flight -= 1

        await asyncio.wait_for(
            asyncio.gather(*(acquire_and_track() for _ in range(10))),
            timeout=10.0,
        )

        assert peak <= 2, f"Semaphore violated max_concurrent=2 (peak={peak})"
    finally:
        await tm.shutdown()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_touches_task_ttl() -> None:
    """Reading a task refreshes retention for active polling clients (#3941)."""
    tm = TaskManager(ttl_seconds=0.05)
    try:
        tm.set("task-1", {"status": "running"})
        await asyncio.sleep(0.03)

        assert tm.get("task-1") is not None

        await asyncio.sleep(0.03)
        assert tm.get("task-1") is not None
    finally:
        await tm.shutdown()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_closes_and_clears_task_manager() -> None:
    """Shutdown prevents process-local task state from being reused (#3941)."""
    tm = TaskManager()
    tm.set("task-1", {"status": "running"})

    await tm.shutdown()

    with pytest.raises(RuntimeError, match="closed"):
        tm.set("task-2", {"status": "pending"})
    with pytest.raises(RuntimeError, match="closed"):
        tm.get("task-1")
