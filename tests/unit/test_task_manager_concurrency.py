"""Regression test for issue #3506: TaskManager must use asyncio primitives only.

Pins the contract that ``TaskManager._lock`` is an ``asyncio.Lock`` and
``TaskManager._engine_semaphore`` is an ``asyncio.Semaphore``. This guards
against re-introducing ``threading.Lock`` (which deadlocks when mixed with
``asyncio.Semaphore``) and exercises high-concurrency set/get/active_count
calls to ensure no deadlock occurs.
"""

from __future__ import annotations

import asyncio

import pytest
from src.api.task_manager import TaskManager


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lock_and_semaphore_are_asyncio_primitives() -> None:
    """Regression guard: locking primitives must be asyncio-native (#3506)."""
    tm = TaskManager(max_concurrent=2)
    try:
        assert isinstance(tm._lock, asyncio.Lock), (
            "TaskManager._lock must be asyncio.Lock to avoid deadlock with "
            "asyncio.Semaphore (#3506)"
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
            await tm.set(f"task-{i}", {"status": "pending", "value": i})

        async def reader(i: int) -> dict | None:
            return await tm.get(f"task-{i}")

        async def counter() -> int:
            return await tm.active_count()

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

        final_count = await tm.active_count()
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
        await tm.set("task-1", {"status": "running"})
        await asyncio.sleep(0.03)

        assert await tm.get("task-1") is not None

        await asyncio.sleep(0.03)
        assert await tm.get("task-1") is not None
    finally:
        await tm.shutdown()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_closes_and_clears_task_manager() -> None:
    """Shutdown prevents process-local task state from being reused (#3941)."""
    tm = TaskManager()
    await tm.set("task-1", {"status": "running"})

    await tm.shutdown()

    with pytest.raises(RuntimeError, match="closed"):
        await tm.set("task-2", {"status": "pending"})
    with pytest.raises(RuntimeError, match="closed"):
        await tm.get("task-1")
