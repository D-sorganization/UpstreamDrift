"""Unit tests for the memory TaskManager."""

import time
import pytest
import asyncio
from src.api.task_manager import TaskManager, TaskStatus, _validate_task_id


def test_validate_task_id() -> None:
    # Valid string
    _validate_task_id("task-123")

    # Invalid types
    with pytest.raises(ValueError, match="task_id must be a non-empty string"):
        _validate_task_id(None)  # type: ignore

    with pytest.raises(ValueError, match="task_id must be a non-empty string"):
        _validate_task_id("")

    with pytest.raises(ValueError, match="task_id must be a non-empty string"):
        _validate_task_id("   ")


def test_task_manager_initialization() -> None:
    # Default configs
    tm = TaskManager()
    assert tm.TTL_SECONDS == 3600
    assert tm.MAX_TASKS == 1000
    assert tm.MAX_CONCURRENT_ENGINES == 4

    # Custom configs
    tm_custom = TaskManager(ttl_seconds=100, max_tasks=10, max_concurrent=2)
    assert tm_custom.TTL_SECONDS == 100
    assert tm_custom.MAX_TASKS == 10
    assert tm_custom.MAX_CONCURRENT_ENGINES == 2


def test_task_manager_basic_lifecycle() -> None:
    tm = TaskManager()
    task_id = "task-1"
    data = {"status": "pending", "progress": 0.0}

    # Initial state
    assert not tm.exists(task_id)
    assert tm.get(task_id) is None
    assert tm.active_count() == 0

    # Set task
    tm.set(task_id, data)
    assert tm.exists(task_id)
    assert tm.get(task_id) == data
    assert tm.active_count() == 1

    # Update progress clamping
    tm.update_progress(task_id, 50.0)
    task = tm.get(task_id)
    assert task is not None
    assert task["progress"] == 50.0

    # Test underflow and overflow clamping
    tm.update_progress(task_id, -10.0)
    task = tm.get(task_id)
    assert task is not None
    assert task["progress"] == 0.0
    tm.update_progress(task_id, 150.0)
    task = tm.get(task_id)
    assert task is not None
    assert task["progress"] == 100.0

    # Update progress on non-existing task (no-op)
    tm.update_progress("non-existing", 50.0)

    # Mark completed
    result = {"output": "success"}
    tm.mark_completed(task_id, result)
    task = tm.get(task_id)
    assert task is not None
    assert task["status"] == TaskStatus.COMPLETED.value
    assert task["result"] == result
    assert task["progress"] == 100.0

    # Mark completed on non-existing task (no-op)
    tm.mark_completed("non-existing", result)

    # Mark failed
    tm.set("task-2", {"status": "running"})
    tm.mark_failed("task-2", "Something went wrong")
    task2 = tm.get("task-2")
    assert task2 is not None
    assert task2["status"] == TaskStatus.FAILED.value
    assert task2["error"] == "Something went wrong"

    # Mark failed on non-existing task (no-op)
    tm.mark_failed("non-existing", "error")


def test_task_manager_lru_eviction() -> None:
    tm = TaskManager(max_tasks=3)

    tm.set("t1", {"name": "task 1"})
    time.sleep(0.002)  # Ensure distinct timestamps
    tm.set("t2", {"name": "task 2"})
    time.sleep(0.002)
    tm.set("t3", {"name": "task 3"})
    time.sleep(0.002)

    assert tm.active_count() == 3
    # Use direct dictionary access to avoid updating the timestamp
    assert "t1" in tm._tasks

    # Add a 4th task, which should evict "t1" (oldest)
    tm.set("t4", {"name": "task 4"})
    assert tm.active_count() == 3
    assert not tm.exists("t1")
    assert tm.exists("t2")
    assert tm.exists("t3")
    assert tm.exists("t4")


def test_task_manager_ttl_expiration() -> None:
    # Set TTL to a very short duration (e.g. 0.05 seconds)
    tm = TaskManager(ttl_seconds=1)
    tm.TTL_SECONDS = 0.05  # type: ignore[assignment]

    tm.set("t1", {"name": "task 1"})
    assert tm.exists("t1")

    # Sleep so it expires
    time.sleep(0.08)

    # Checking exists/get/active_count should trigger cleanup
    assert not tm.exists("t1")
    assert tm.get("t1") is None
    assert tm.active_count() == 0


def test_task_manager_dict_compatibility() -> None:
    tm = TaskManager()
    task_id = "task-1"
    data = {"status": "pending"}

    # __setitem__ & __contains__
    tm[task_id] = data
    assert task_id in tm
    assert "non-existent" not in tm
    assert 123 not in tm  # Non-string object returns False

    # __getitem__
    assert tm[task_id] == data
    with pytest.raises(KeyError):
        _ = tm["non-existent"]

    # __len__
    assert len(tm) == 1

    # keys, values, items
    assert tm.keys() == [task_id]
    assert tm.values() == [data]
    assert tm.items() == [(task_id, data)]

    # __iter__
    assert list(iter(tm)) == [task_id]

    # __delitem__
    del tm[task_id]
    assert task_id not in tm
    with pytest.raises(KeyError):
        del tm["non-existent"]


def test_task_manager_shutdown() -> None:
    tm = TaskManager()
    tm.set("t1", {"status": "pending"})

    # Semaphore works
    assert isinstance(tm.engine_semaphore, asyncio.Semaphore)

    # Shutdown
    asyncio.run(tm.shutdown())

    # Operations raise RuntimeError after shutdown
    with pytest.raises(RuntimeError, match="TaskManager is closed"):
        tm.set("t2", {"status": "pending"})

    with pytest.raises(RuntimeError, match="TaskManager is closed"):
        _ = tm.get("t1")

    with pytest.raises(RuntimeError, match="TaskManager is closed"):
        tm.exists("t1")

    with pytest.raises(RuntimeError, match="TaskManager is closed"):
        tm.update_progress("t1", 50.0)

    with pytest.raises(RuntimeError, match="TaskManager is closed"):
        tm.mark_completed("t1", {})

    with pytest.raises(RuntimeError, match="TaskManager is closed"):
        tm.mark_failed("t1", "error")

    with pytest.raises(RuntimeError, match="TaskManager is closed"):
        tm.active_count()

    with pytest.raises(RuntimeError, match="TaskManager is closed"):
        _ = tm.engine_semaphore

    with pytest.raises(RuntimeError, match="TaskManager is closed"):
        _ = "t1" in tm

    with pytest.raises(RuntimeError, match="TaskManager is closed"):
        _ = tm["t1"]

    with pytest.raises(RuntimeError, match="TaskManager is closed"):
        del tm["t1"]

    with pytest.raises(RuntimeError, match="TaskManager is closed"):
        _ = len(tm)

    with pytest.raises(RuntimeError, match="TaskManager is closed"):
        _ = list(iter(tm))

    with pytest.raises(RuntimeError, match="TaskManager is closed"):
        tm.keys()

    with pytest.raises(RuntimeError, match="TaskManager is closed"):
        tm.values()

    with pytest.raises(RuntimeError, match="TaskManager is closed"):
        tm.items()

    # Calling shutdown twice is a no-op
    asyncio.run(tm.shutdown())
