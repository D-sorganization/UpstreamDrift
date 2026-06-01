"""Tests for the durable task manager."""

import time
from pathlib import Path

import pytest
from src.api.task_manager_durable import (
    DurableTaskManager,
    SQLiteBackend,
    TaskRecord,
    TaskStatus,
)


@pytest.fixture
def sqlite_backend(tmp_path: Path):
    """Provide a temporary SQLite backend."""
    db_path = tmp_path / "tasks.db"
    backend = SQLiteBackend(db_path=db_path)
    return backend


@pytest.fixture
def task_manager(sqlite_backend: SQLiteBackend):
    """Provide a durable task manager with a temporary SQLite backend."""
    manager = DurableTaskManager(
        backend=sqlite_backend,
        heartbeat_timeout=1.0,
        auto_cleanup=False,  # Disable background task for cleaner tests
    )
    return manager


class TestSQLiteBackend:
    """Test the SQLite storage backend."""

    def test_init_db(self, sqlite_backend: SQLiteBackend):
        """Test database initialization creates correct schema."""
        conn = sqlite_backend._get_conn()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='tasks'"
        )
        assert cursor.fetchone() is not None

        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        )
        assert cursor.fetchone() is not None

    def test_crud_operations(self, sqlite_backend: SQLiteBackend):
        """Test creating, reading, updating, and deleting tasks."""
        # Create
        record = TaskRecord(
            task_id="test-task-1",
            status=TaskStatus.PENDING.value,
            task_type="test",
            input_data={"foo": "bar"},
        )
        sqlite_backend.create_task(record)

        # Read
        fetched = sqlite_backend.get_task("test-task-1")
        assert fetched is not None
        assert fetched.task_id == "test-task-1"
        assert fetched.task_type == "test"
        assert fetched.input_data == {"foo": "bar"}

        # Update
        fetched.status = TaskStatus.RUNNING.value
        fetched.progress = 50.0
        sqlite_backend.update_task(fetched)

        updated = sqlite_backend.get_task("test-task-1")
        assert updated is not None
        assert updated.status == TaskStatus.RUNNING.value
        assert updated.progress == 50.0

        # Delete
        assert sqlite_backend.delete_task("test-task-1") is True
        assert sqlite_backend.get_task("test-task-1") is None

    def test_find_by_run_id(self, sqlite_backend: SQLiteBackend):
        """Test finding task by idempotency key."""
        record = TaskRecord(
            task_id="test-task-2",
            run_id="run-123",
            task_type="test",
        )
        sqlite_backend.create_task(record)

        fetched = sqlite_backend.find_by_run_id("run-123")
        assert fetched is not None
        assert fetched.task_id == "test-task-2"

        missing = sqlite_backend.find_by_run_id("non-existent")
        assert missing is None

    def test_find_stalled_tasks(self, sqlite_backend: SQLiteBackend):
        """Test stalled task detection."""
        now = time.time()

        # Stalled task
        stalled = TaskRecord(
            task_id="stalled-task",
            status=TaskStatus.RUNNING.value,
            heartbeat_at=now - 100.0,
        )
        sqlite_backend.create_task(stalled)

        # Active task
        active = TaskRecord(
            task_id="active-task",
            status=TaskStatus.RUNNING.value,
            heartbeat_at=now,
        )
        sqlite_backend.create_task(active)

        # Pending task (should not be considered stalled)
        pending = TaskRecord(
            task_id="pending-task",
            status=TaskStatus.PENDING.value,
            heartbeat_at=now - 100.0,
        )
        sqlite_backend.create_task(pending)

        stalled_tasks = sqlite_backend.find_stalled_tasks(heartbeat_timeout=50.0)
        assert len(stalled_tasks) == 1
        assert stalled_tasks[0].task_id == "stalled-task"

    def test_find_expired_tasks(self, sqlite_backend: SQLiteBackend):
        """Test expired task detection."""
        now = time.time()

        # Expired by TTL
        ttl_expired = TaskRecord(
            task_id="ttl-expired",
            status=TaskStatus.PENDING.value,
            created_at=now - 4000.0,
            ttl_seconds=3600,
        )
        sqlite_backend.create_task(ttl_expired)

        # Expired by retention
        retention_expired = TaskRecord(
            task_id="retention-expired",
            status=TaskStatus.COMPLETED.value,
            completed_at=now - 90000.0,
            retention_seconds=86400,
        )
        sqlite_backend.create_task(retention_expired)

        # Active pending task
        active_pending = TaskRecord(
            task_id="active-pending",
            status=TaskStatus.PENDING.value,
            created_at=now,
            ttl_seconds=3600,
        )
        sqlite_backend.create_task(active_pending)

        expired_tasks = sqlite_backend.find_expired_tasks()
        assert len(expired_tasks) == 2
        expired_ids = [t.task_id for t in expired_tasks]
        assert "ttl-expired" in expired_ids
        assert "retention-expired" in expired_ids


class TestDurableTaskManager:
    """Test the durable task manager orchestrator."""

    def test_create_task(self, task_manager: DurableTaskManager):
        """Test task creation."""
        task_id = task_manager.create_task(
            task_type="simulation",
            input_data={"x": 10},
        )
        assert task_id is not None

        task = task_manager.get_task(task_id)
        assert task is not None
        assert task.task_type == "simulation"
        assert task.input_data == {"x": 10}
        assert task.status == TaskStatus.PENDING.value

    def test_create_task_idempotent(self, task_manager: DurableTaskManager):
        """Test task creation with run_id deduplication."""
        task_id1 = task_manager.create_task(
            task_type="simulation",
            run_id="unique-run-123",
        )

        task_id2 = task_manager.create_task(
            task_type="simulation",
            run_id="unique-run-123",
        )

        assert task_id1 == task_id2

    def test_acquire_and_release_task(self, task_manager: DurableTaskManager):
        """Test acquiring a task for a worker and releasing it."""
        task_id = task_manager.create_task()

        # Acquire
        task = task_manager.acquire_task(worker_id="worker-1")
        assert task is not None
        assert task.task_id == task_id
        assert task.status == TaskStatus.RUNNING.value
        assert task.worker_id == "worker-1"
        assert task.started_at is not None

        # Second acquire should return None (no more tasks)
        assert task_manager.acquire_task(worker_id="worker-2") is None

        # Release
        assert task_manager.release_task(task_id) is True

        released = task_manager.get_task(task_id)
        assert released is not None
        assert released.status == TaskStatus.PENDING.value
        assert released.worker_id is None
        assert released.retry_count == 1

    def test_release_task_max_retries(self, task_manager: DurableTaskManager):
        """Test releasing a task that has exceeded max retries."""
        task_id = task_manager.create_task(max_retries=1)

        task_manager.acquire_task()
        assert task_manager.release_task(task_id) is True  # retry 1

        task_manager.acquire_task()
        assert task_manager.release_task(task_id) is False  # exceeds max retries

    def test_progress_and_heartbeat(self, task_manager: DurableTaskManager):
        """Test progress updates and heartbeats."""
        task_id = task_manager.create_task()

        assert task_manager.update_progress(task_id, 50.0) is True
        task = task_manager.get_task(task_id)
        assert task is not None
        assert task.progress == 50.0

        old_heartbeat = task.heartbeat_at
        time.sleep(0.01)

        assert task_manager.heartbeat(task_id) is True
        task = task_manager.get_task(task_id)
        assert task is not None
        assert task.heartbeat_at > old_heartbeat

    def test_mark_completed_and_failed(self, task_manager: DurableTaskManager):
        """Test marking tasks terminal states."""
        # Complete
        task_id1 = task_manager.create_task()
        assert task_manager.mark_completed(task_id1, {"result": "success"}) is True

        task1 = task_manager.get_task(task_id1)
        assert task1 is not None
        assert task1.status == TaskStatus.COMPLETED.value
        assert task1.result == {"result": "success"}
        assert task1.progress == 100.0
        assert task1.completed_at is not None

        # Fail
        task_id2 = task_manager.create_task()
        assert task_manager.mark_failed(task_id2, "error message") is True

        task2 = task_manager.get_task(task_id2)
        assert task2 is not None
        assert task2.status == TaskStatus.FAILED.value
        assert task2.error == "error message"
        assert task2.completed_at is not None

    def test_cancel_task(self, task_manager: DurableTaskManager):
        """Test cancelling a task."""
        task_id = task_manager.create_task()

        assert task_manager.cancel_task(task_id) is True

        task = task_manager.get_task(task_id)
        assert task is not None
        assert task.status == TaskStatus.CANCELLED.value

        # Cannot cancel completed task
        task_manager.mark_completed(task_id, {"res": "ok"})
        assert task_manager.cancel_task(task_id) is False

    def test_find_stalled_tasks(self, task_manager: DurableTaskManager):
        """Test stalled task integration."""
        task_id = task_manager.create_task()
        task_manager.acquire_task()

        task = task_manager.get_task(task_id)
        assert task is not None
        task.heartbeat_at = time.time() - 2.0
        task_manager.backend.update_task(task)

        stalled = task_manager.find_stalled_tasks()
        assert len(stalled) == 1
        assert stalled[0].task_id == task_id

    @pytest.mark.asyncio
    async def test_shutdown(self, sqlite_backend: SQLiteBackend):
        """Test graceful shutdown."""
        manager = DurableTaskManager(
            backend=sqlite_backend,
            auto_cleanup=True,
            cleanup_interval=1,
        )
        assert manager._cleanup_task is not None
        assert not manager._cleanup_task.done()

        await manager.shutdown()

        assert manager._closed is True
        assert manager._cleanup_task.cancelled() or manager._cleanup_task.done()


class TestDurableTaskManagerAutoCleanupFallback:
    """#6979: DurableTaskManager auto-cleanup fallback when no event loop."""

    def test_warns_and_starts_daemon_thread_when_no_event_loop(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Auto-cleanup must log a warning and spawn a daemon thread outside async ctx."""
        import logging

        db_path = tmp_path / "tasks.db"
        backend = SQLiteBackend(db_path=db_path)

        with caplog.at_level(logging.WARNING):
            manager = DurableTaskManager(
                backend=backend, auto_cleanup=True, cleanup_interval=60
            )

        warning_text = " ".join(r.message for r in caplog.records).lower()
        assert "no running event loop" in warning_text or "daemon" in warning_text, (
            f"Expected warning about missing event loop, got: {caplog.text!r}"
        )

        assert manager._cleanup_thread is not None, "Expected daemon cleanup thread"
        assert manager._cleanup_thread.is_alive()
        assert manager._cleanup_thread.daemon

    def test_daemon_thread_runs_cleanup(self, tmp_path: Path) -> None:
        """Daemon cleanup thread must actually delete expired tasks."""
        import time

        db_path = tmp_path / "tasks.db"
        backend = SQLiteBackend(db_path=db_path)

        manager = DurableTaskManager(
            backend=backend, auto_cleanup=True, cleanup_interval=1
        )

        # Insert an already-expired task
        now = time.time()
        expired_record = TaskRecord(
            task_id="expired-task",
            status=TaskStatus.PENDING.value,
            created_at=now - 7200.0,
            ttl_seconds=3600,
        )
        backend.create_task(expired_record)

        assert backend.get_task("expired-task") is not None

        # Wait up to 3 s for the daemon thread's first cleanup sweep
        deadline = time.time() + 3.0
        while time.time() < deadline:
            if backend.get_task("expired-task") is None:
                break
            time.sleep(0.1)

        manager._closed = True
        if manager._cleanup_stop is not None:
            manager._cleanup_stop.set()

        assert backend.get_task("expired-task") is None, (
            "Daemon cleanup thread did not delete the expired task within 3 s"
        )
