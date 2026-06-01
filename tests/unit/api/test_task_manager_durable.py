"""Unit tests for the DurableTaskManager and SQLiteBackend."""

import os
import time
import pytest
import sqlite3
import asyncio
from pathlib import Path
from unittest.mock import patch

from src.api.task_manager_durable import (
    DurableTaskManager,
    SQLiteBackend,
    TaskRecord,
    TaskStatus,
)


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_tasks.db"


@pytest.fixture
def sqlite_backend(temp_db_path: Path) -> SQLiteBackend:
    return SQLiteBackend(db_path=temp_db_path)


def test_sqlite_backend_init(temp_db_path: Path) -> None:
    # Test initialization with fallback default DB path
    with patch.dict(os.environ, {"ARTIFACT_DIR": str(temp_db_path.parent)}):
        backend = SQLiteBackend()
        assert backend.db_path.name == "tasks.db"
        assert backend.db_path.exists()

    # Test initialization with explicit db path
    backend2 = SQLiteBackend(db_path=temp_db_path)
    assert backend2.db_path == temp_db_path
    assert temp_db_path.exists()

    # Verify tables and indexes exist
    conn = sqlite3.connect(str(temp_db_path))
    conn.row_factory = sqlite3.Row

    # Check tables
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row["name"] for row in cursor.fetchall()]
    assert "tasks" in tables
    assert "schema_version" in tables

    # Check schema version
    cursor = conn.execute("SELECT version FROM schema_version")
    row = cursor.fetchone()
    assert row["version"] == SQLiteBackend.SCHEMA_VERSION
    conn.close()


def test_sqlite_backend_transaction_rollback(sqlite_backend: SQLiteBackend) -> None:
    # Verify that a transaction rolls back if an exception occurs inside the context manager
    with (
        pytest.raises(RuntimeError, match="DB Force Fail"),
        sqlite_backend._transaction() as conn,
    ):
        conn.execute(
            "INSERT INTO tasks (task_id, created_at, updated_at, heartbeat_at) VALUES (?, ?, ?, ?)",
            ("t-fail", time.time(), time.time(), time.time()),
        )
        raise RuntimeError("DB Force Fail")

    # Verify that the task was not inserted
    assert sqlite_backend.get_task("t-fail") is None


def test_sqlite_backend_crud(sqlite_backend: SQLiteBackend) -> None:
    record = TaskRecord(
        task_id="task-1",
        run_id="run-1",
        status=TaskStatus.PENDING.value,
        task_type="simulation",
        input_data={"param": 42},
        config_hash="abc",
        code_version="1.0.0",
        progress=0.0,
        result=None,
        error=None,
        created_at=time.time(),
        updated_at=time.time(),
        heartbeat_at=time.time(),
        priority=10,
    )

    # Create task
    sqlite_backend.create_task(record)

    # Get task
    retrieved = sqlite_backend.get_task("task-1")
    assert retrieved is not None
    assert retrieved.task_id == "task-1"
    assert retrieved.run_id == "run-1"
    assert retrieved.status == TaskStatus.PENDING.value
    assert retrieved.input_data == {"param": 42}
    assert retrieved.priority == 10

    # Get non-existing task
    assert sqlite_backend.get_task("missing") is None

    # Update task
    retrieved.status = TaskStatus.RUNNING.value
    retrieved.progress = 50.0
    retrieved.result = {"output": "partial"}
    retrieved.error = "none"
    sqlite_backend.update_task(retrieved)

    updated = sqlite_backend.get_task("task-1")
    assert updated is not None
    assert updated.status == TaskStatus.RUNNING.value
    assert updated.progress == 50.0
    assert updated.result == {"output": "partial"}
    assert updated.error == "none"

    # Find by run_id
    found = sqlite_backend.find_by_run_id("run-1")
    assert found is not None
    assert found.task_id == "task-1"
    assert sqlite_backend.find_by_run_id("missing") is None

    # Delete task
    assert sqlite_backend.delete_task("task-1")
    assert sqlite_backend.get_task("task-1") is None
    assert not sqlite_backend.delete_task("task-1")


def test_sqlite_backend_acquire_and_release(sqlite_backend: SQLiteBackend) -> None:
    # Create two pending tasks with different priorities
    t1 = TaskRecord(task_id="t1", priority=0, created_at=time.time())
    t2 = TaskRecord(task_id="t2", priority=5, created_at=time.time())
    sqlite_backend.create_task(t1)
    sqlite_backend.create_task(t2)

    # Acquire task — should fetch the highest priority task (t2) first
    acquired = sqlite_backend.acquire_task(worker_id="worker-a")
    assert acquired is not None
    assert acquired.task_id == "t2"
    assert acquired.status == TaskStatus.RUNNING.value
    assert acquired.worker_id == "worker-a"

    # Next acquire should fetch t1
    acquired2 = sqlite_backend.acquire_task(worker_id="worker-a")
    assert acquired2 is not None
    assert acquired2.task_id == "t1"

    # Next acquire should be None
    assert sqlite_backend.acquire_task(worker_id="worker-a") is None

    # Release task t2 back for retry
    sqlite_backend.release_task("t2")
    t2_released = sqlite_backend.get_task("t2")
    assert t2_released is not None
    assert t2_released.status == TaskStatus.PENDING.value
    assert t2_released.worker_id is None
    assert t2_released.retry_count == 1


def test_sqlite_backend_stalled_tasks(sqlite_backend: SQLiteBackend) -> None:
    now = time.time()
    t1 = TaskRecord(
        task_id="t1",
        status=TaskStatus.RUNNING.value,
        heartbeat_at=now - 500,  # Stalled (timeout 300)
    )
    t2 = TaskRecord(
        task_id="t2",
        status=TaskStatus.RUNNING.value,
        heartbeat_at=now - 50,  # Active
    )
    t3 = TaskRecord(
        task_id="t3",
        status=TaskStatus.PENDING.value,
        heartbeat_at=now - 500,  # Pending, should not count as stalled
    )
    sqlite_backend.create_task(t1)
    sqlite_backend.create_task(t2)
    sqlite_backend.create_task(t3)

    stalled = sqlite_backend.find_stalled_tasks(heartbeat_timeout=300.0)
    assert len(stalled) == 1
    assert stalled[0].task_id == "t1"


def test_sqlite_backend_expired_tasks_and_cleanup(
    sqlite_backend: SQLiteBackend,
) -> None:
    now = time.time()
    # Expired by TTL
    t1 = TaskRecord(
        task_id="t1",
        status=TaskStatus.PENDING.value,
        created_at=now - 2000,
        ttl_seconds=1000,
    )
    # Active
    t2 = TaskRecord(
        task_id="t2",
        status=TaskStatus.PENDING.value,
        created_at=now - 50,
        ttl_seconds=1000,
    )
    # Expired by retention
    t3 = TaskRecord(
        task_id="t3",
        status=TaskStatus.COMPLETED.value,
        completed_at=now - 1000,
        retention_seconds=500,
    )
    # Active completed
    t4 = TaskRecord(
        task_id="t4",
        status=TaskStatus.COMPLETED.value,
        completed_at=now - 100,
        retention_seconds=500,
    )
    sqlite_backend.create_task(t1)
    sqlite_backend.create_task(t2)
    sqlite_backend.create_task(t3)
    sqlite_backend.create_task(t4)

    expired = sqlite_backend.find_expired_tasks()
    expired_ids = {t.task_id for t in expired}
    assert "t1" in expired_ids
    assert "t3" in expired_ids
    assert "t2" not in expired_ids
    assert "t4" not in expired_ids

    # Run cleanup
    count = sqlite_backend.cleanup()
    assert count == 2
    assert sqlite_backend.get_task("t1") is None
    assert sqlite_backend.get_task("t3") is None
    assert sqlite_backend.get_task("t2") is not None
    assert sqlite_backend.get_task("t4") is not None


def test_sqlite_backend_cancel(sqlite_backend: SQLiteBackend) -> None:
    t1 = TaskRecord(task_id="t1", status=TaskStatus.PENDING.value)
    t2 = TaskRecord(task_id="t2", status=TaskStatus.COMPLETED.value)
    sqlite_backend.create_task(t1)
    sqlite_backend.create_task(t2)

    # Cancel pending
    assert sqlite_backend.cancel_task("t1")
    task_t1 = sqlite_backend.get_task("t1")
    assert task_t1 is not None
    assert task_t1.status == TaskStatus.CANCELLED.value

    # Cancel completed (should fail / return False)
    assert not sqlite_backend.cancel_task("t2")
    task_t2 = sqlite_backend.get_task("t2")
    assert task_t2 is not None
    assert task_t2.status == TaskStatus.COMPLETED.value


def test_durable_task_manager_basic_flow(sqlite_backend: SQLiteBackend) -> None:
    # Instantiate manager without auto-cleanup loop
    manager = DurableTaskManager(backend=sqlite_backend, auto_cleanup=False)

    # Create task
    task_id = manager.create_task(
        task_type="video_render",
        input_data={"fps": 30},
        run_id="run-unique",
        ttl_seconds=500,
        retention_seconds=1000,
        priority=2,
    )
    assert task_id is not None

    # Idempotency check: creating again with same run_id returns same task_id
    task_id_dup = manager.create_task(
        task_type="video_render",
        input_data={"fps": 30},
        run_id="run-unique",
    )
    assert task_id == task_id_dup

    # Get task
    record = manager.get_task(task_id)
    assert record is not None
    assert record.task_type == "video_render"
    assert record.input_data == {"fps": 30}
    assert record.priority == 2
    assert record.progress == 0.0

    # Get non-existing task
    assert manager.get_task("missing") is None

    # Update progress
    assert manager.update_progress(task_id, 45.5)
    assert not manager.update_progress("missing", 50.0)
    task_p = manager.get_task(task_id)
    assert task_p is not None
    assert task_p.progress == 45.5

    # Heartbeat update
    assert manager.heartbeat(task_id)
    assert not manager.heartbeat("missing")

    # Mark completed
    result_data = {"url": "http://result"}
    assert manager.mark_completed(task_id, result_data)
    assert not manager.mark_completed("missing", result_data)

    completed_task = manager.get_task(task_id)
    assert completed_task is not None
    assert completed_task.status == TaskStatus.COMPLETED.value
    assert completed_task.result == result_data
    assert completed_task.progress == 100.0

    # Mark failed
    task_id2 = manager.create_task()
    assert manager.mark_failed(task_id2, "Error details")
    assert not manager.mark_failed("missing", "Error")

    failed_task = manager.get_task(task_id2)
    assert failed_task is not None
    assert failed_task.status == TaskStatus.FAILED.value
    assert failed_task.error == "Error details"

    # Cancel task
    task_id3 = manager.create_task()
    assert manager.cancel_task(task_id3)
    task_c = manager.get_task(task_id3)
    assert task_c is not None
    assert task_c.status == TaskStatus.CANCELLED.value


def test_durable_task_manager_worker_methods(sqlite_backend: SQLiteBackend) -> None:
    manager = DurableTaskManager(backend=sqlite_backend, auto_cleanup=False)

    task_id = manager.create_task(max_retries=2)

    # Acquire task
    acquired = manager.acquire_task(worker_id="my-worker")
    assert acquired is not None
    assert acquired.task_id == task_id
    assert acquired.worker_id == "my-worker"

    # Acquire with default worker_id (hostname/pid fallback)
    task_id2 = manager.create_task()
    acquired_default = manager.acquire_task()
    assert acquired_default is not None
    assert acquired_default.task_id == task_id2
    assert acquired_default.worker_id is not None

    # Release task for retry
    assert manager.release_task(task_id)
    assert not manager.release_task("missing")
    task_p = manager.get_task(task_id)
    assert task_p is not None
    assert task_p.status == TaskStatus.PENDING.value

    # Acquire and release again to hit max retries
    manager.acquire_task("my-worker")
    assert manager.release_task(task_id)  # retry count becomes 2

    manager.acquire_task("my-worker")
    assert not manager.release_task(
        task_id
    )  # retry count reaches 2 (max_retries=2), fails release
    task_r = manager.get_task(task_id)
    assert task_r is not None
    assert task_r.status == TaskStatus.RUNNING.value  # stays running / failed


def test_durable_task_manager_stalled_tasks(sqlite_backend: SQLiteBackend) -> None:
    manager = DurableTaskManager(
        backend=sqlite_backend, heartbeat_timeout=10.0, auto_cleanup=False
    )

    # Create running stalled task
    t_stalled = TaskRecord(
        task_id="stalled",
        status=TaskStatus.RUNNING.value,
        heartbeat_at=time.time() - 20.0,
    )
    sqlite_backend.create_task(t_stalled)

    stalled_list = manager.find_stalled_tasks()
    assert len(stalled_list) == 1
    assert stalled_list[0].task_id == "stalled"


@pytest.mark.asyncio
async def test_durable_task_manager_cleanup_loop(temp_db_path: Path) -> None:
    backend = SQLiteBackend(db_path=temp_db_path)

    # Create expired task
    t_expired = TaskRecord(
        task_id="expired-1",
        status=TaskStatus.PENDING.value,
        created_at=time.time() - 100,
        ttl_seconds=10,
    )
    backend.create_task(t_expired)

    # Initialize manager with short cleanup interval
    manager = DurableTaskManager(backend=backend, auto_cleanup=True, cleanup_interval=1)

    # Sleep to allow background loop to run at least once
    await asyncio.sleep(1.5)

    # Check that the task was deleted by the auto-cleanup background task
    assert backend.get_task("expired-1") is None

    # Test clean shutdown
    await manager.shutdown()
    assert manager._closed


@pytest.mark.asyncio
async def test_durable_task_manager_cleanup_loop_exception(temp_db_path: Path) -> None:
    backend = SQLiteBackend(db_path=temp_db_path)

    with patch.object(
        backend, "cleanup", side_effect=RuntimeError("Mock DB Error")
    ) as mock_cleanup:
        manager = DurableTaskManager(
            backend=backend, auto_cleanup=True, cleanup_interval=1
        )
        # Sleep to let loop run and catch error
        await asyncio.sleep(1.5)
        # Shutdown
        await manager.shutdown()
        assert mock_cleanup.called
