"""Durable task manager for production environments.

This module provides a production-ready task manager that stores task state
in durable storage (SQLite by default) instead of process-local memory.

Features:
- Durable storage: Task state survives process restarts
- Multi-worker safe: Tasks accessible across multiple worker processes
- Heartbeat tracking: Detect stalled tasks via heartbeat timeouts
- Cancellation support: Tasks can be cancelled and queried for cancellation
- Result retention: Configurable retention policy for completed task results
- Replay metadata: Stores input/config/code-version for replay capability

See issue #3941 for remediation details.
"""

from __future__ import annotations

import asyncio
import enum
import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class TaskStatus(enum.Enum):
    """Lifecycle states for a background task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskRecord:
    """Complete task record with all metadata for durability and replay.

    Attributes:
        task_id: Unique task identifier (UUID4)
        run_id: Idempotency key for deduplication (optional)
        status: Current task status
        task_type: Type of task (e.g., 'simulation', 'video_render')
        input_data: JSON-serializable input parameters
        config_hash: Hash of configuration for replay verification
        code_version: Code/version identifier at task creation
        progress: Progress percentage (0-100)
        result: Task result data (JSON-serializable)
        error: Error message if failed
        created_at: Unix timestamp of creation
        updated_at: Unix timestamp of last update
        heartbeat_at: Unix timestamp of last heartbeat
        started_at: Unix timestamp when task started running
        completed_at: Unix timestamp when task completed/failed
        worker_id: ID of worker processing this task
        retry_count: Number of retry attempts
        max_retries: Maximum retry attempts allowed
        ttl_seconds: Time-to-live in seconds
        retention_seconds: How long to keep result after completion
        priority: Task priority (higher = more urgent)
    """

    task_id: str
    run_id: str | None = None
    status: str = TaskStatus.PENDING.value
    task_type: str = "generic"
    input_data: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""
    code_version: str = ""
    progress: float = 0.0
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())
    heartbeat_at: float = field(default_factory=lambda: time.time())
    started_at: float | None = None
    completed_at: float | None = None
    worker_id: str | None = None
    retry_count: int = 0
    max_retries: int = 3
    ttl_seconds: int = 3600
    retention_seconds: int = 86400  # 24 hours default
    priority: int = 0


class StorageBackend(Protocol):
    """Protocol for task storage backends."""

    def create_task(self, record: TaskRecord) -> None:
        """Create a new task record."""
        ...

    def get_task(self, task_id: str) -> TaskRecord | None:
        """Retrieve a task by ID."""
        ...

    def update_task(self, record: TaskRecord) -> None:
        """Update an existing task record."""
        ...

    def delete_task(self, task_id: str) -> bool:
        """Delete a task record."""
        ...

    def find_by_run_id(self, run_id: str) -> TaskRecord | None:
        """Find task by run_id for idempotency."""
        ...

    def find_stalled_tasks(self, heartbeat_timeout: float) -> list[TaskRecord]:
        """Find tasks that have stalled (heartbeat timeout)."""
        ...

    def find_expired_tasks(self) -> list[TaskRecord]:
        """Find tasks past their TTL or retention period."""
        ...

    def cancel_task(self, task_id: str) -> bool:
        """Mark a task as cancelled."""
        ...

    def acquire_task(self, worker_id: str) -> TaskRecord | None:
        """Acquire a pending task for processing (claim ownership)."""
        ...

    def release_task(self, task_id: str) -> None:
        """Release a task back to pending (for retry)."""
        ...

    def cleanup(self) -> int:
        """Clean up expired tasks, return count deleted."""
        ...


class SQLiteBackend:
    """SQLite-based durable storage backend.

    Provides ACID-compliant storage for task records with support for:
    - Concurrent access via threading lock
    - Automatic schema migrations
    - Efficient queries for stalled/expired tasks
    - Worker ownership tracking
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: str | Path | None = None):
        """Initialize SQLite backend.

        Args:
            db_path: Path to SQLite database. Defaults to ARTIFACT_DIR/tasks.db
                     or /tmp/upstream_drift_tasks.db as fallback.
        """
        if db_path is None:
            artifact_dir = os.environ.get(
                "ARTIFACT_DIR",
                os.path.join("/tmp", "upstream_drift_artifacts"),
            )
            os.makedirs(artifact_dir, exist_ok=True)
            db_path = os.path.join(artifact_dir, "tasks.db")

        self.db_path = Path(db_path)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                isolation_level=None,  # Autocommit mode
            )
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=30000")
        return cast(sqlite3.Connection, getattr(self._local, "conn", None))

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database transactions."""
        conn = self._get_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._lock, self._transaction() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    run_id TEXT UNIQUE,
                    status TEXT NOT NULL DEFAULT 'pending',
                    task_type TEXT NOT NULL DEFAULT 'generic',
                    input_data TEXT NOT NULL DEFAULT '{}',
                    config_hash TEXT,
                    code_version TEXT,
                    progress REAL NOT NULL DEFAULT 0.0,
                    result TEXT,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    heartbeat_at REAL NOT NULL,
                    started_at REAL,
                    completed_at REAL,
                    worker_id TEXT,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    max_retries INTEGER NOT NULL DEFAULT 3,
                    ttl_seconds INTEGER NOT NULL DEFAULT 3600,
                    retention_seconds INTEGER NOT NULL DEFAULT 86400,
                    priority INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tasks_status 
                ON tasks(status, priority DESC, created_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tasks_run_id 
                ON tasks(run_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tasks_worker 
                ON tasks(worker_id, status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tasks_heartbeat 
                ON tasks(heartbeat_at, status)
            """)
            # Schema version tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
            """)
            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
                (self.SCHEMA_VERSION,),
            )

    def _row_to_record(self, row: sqlite3.Row) -> TaskRecord:
        """Convert database row to TaskRecord."""
        return TaskRecord(
            task_id=row["task_id"],
            run_id=row["run_id"],
            status=row["status"],
            task_type=row["task_type"],
            input_data=json.loads(row["input_data"]),
            config_hash=row["config_hash"] or "",
            code_version=row["code_version"] or "",
            progress=row["progress"],
            result=json.loads(row["result"]) if row["result"] else None,
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            heartbeat_at=row["heartbeat_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            worker_id=row["worker_id"],
            retry_count=row["retry_count"],
            max_retries=row["max_retries"],
            ttl_seconds=row["ttl_seconds"],
            retention_seconds=row["retention_seconds"],
            priority=row["priority"],
        )

    def _record_to_values(self, record: TaskRecord) -> tuple:
        """Convert TaskRecord to database values tuple."""
        return (
            record.task_id,
            record.run_id,
            record.status,
            record.task_type,
            json.dumps(record.input_data),
            record.config_hash,
            record.code_version,
            record.progress,
            json.dumps(record.result) if record.result else None,
            record.error,
            record.created_at,
            record.updated_at,
            record.heartbeat_at,
            record.started_at,
            record.completed_at,
            record.worker_id,
            record.retry_count,
            record.max_retries,
            record.ttl_seconds,
            record.retention_seconds,
            record.priority,
        )

    def create_task(self, record: TaskRecord) -> None:
        """Create a new task record."""
        with self._lock, self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO tasks (
                    task_id, run_id, status, task_type, input_data,
                    config_hash, code_version, progress, result, error,
                    created_at, updated_at, heartbeat_at, started_at, completed_at,
                    worker_id, retry_count, max_retries, ttl_seconds,
                    retention_seconds, priority
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                self._record_to_values(record),
            )

    def get_task(self, task_id: str) -> TaskRecord | None:
        """Retrieve a task by ID."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
            row = cursor.fetchone()
            return self._row_to_record(row) if row else None

    def update_task(self, record: TaskRecord) -> None:
        """Update an existing task record."""
        record.updated_at = time.time()
        with self._lock, self._transaction() as conn:
            conn.execute(
                """
                UPDATE tasks SET
                    run_id = ?, status = ?, task_type = ?, input_data = ?,
                    config_hash = ?, code_version = ?, progress = ?,
                    result = ?, error = ?, updated_at = ?, heartbeat_at = ?,
                    started_at = ?, completed_at = ?, worker_id = ?,
                    retry_count = ?, max_retries = ?, ttl_seconds = ?,
                    retention_seconds = ?, priority = ?
                WHERE task_id = ?
            """,
                (
                    record.run_id,
                    record.status,
                    record.task_type,
                    json.dumps(record.input_data),
                    record.config_hash,
                    record.code_version,
                    record.progress,
                    json.dumps(record.result) if record.result else None,
                    record.error,
                    record.updated_at,
                    record.heartbeat_at,
                    record.started_at,
                    record.completed_at,
                    record.worker_id,
                    record.retry_count,
                    record.max_retries,
                    record.ttl_seconds,
                    record.retention_seconds,
                    record.priority,
                    record.task_id,
                ),
            )

    def delete_task(self, task_id: str) -> bool:
        """Delete a task record."""
        with self._lock, self._transaction() as conn:
            cursor = conn.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
            return cursor.rowcount > 0

    def find_by_run_id(self, run_id: str) -> TaskRecord | None:
        """Find task by run_id for idempotency."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute("SELECT * FROM tasks WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            return self._row_to_record(row) if row else None

    def find_stalled_tasks(self, heartbeat_timeout: float) -> list[TaskRecord]:
        """Find tasks that have stalled (heartbeat timeout)."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                """
                SELECT * FROM tasks 
                WHERE status = 'running' 
                AND heartbeat_at < ?
            """,
                (time.time() - heartbeat_timeout,),
            )
            return [self._row_to_record(row) for row in cursor.fetchall()]

    def find_expired_tasks(self) -> list[TaskRecord]:
        """Find tasks past their TTL or retention period."""
        now = time.time()
        with self._lock:
            conn = self._get_conn()
            # Expired by TTL (not yet started or pending)
            cursor = conn.execute(
                """
                SELECT * FROM tasks 
                WHERE status IN ('pending', 'running')
                AND created_at + ttl_seconds < ?
            """,
                (now,),
            )
            expired = [self._row_to_record(row) for row in cursor.fetchall()]

            # Expired by retention (completed/failed/cancelled)
            cursor = conn.execute(
                """
                SELECT * FROM tasks 
                WHERE status IN ('completed', 'failed', 'cancelled')
                AND completed_at + retention_seconds < ?
            """,
                (now,),
            )
            expired.extend(self._row_to_record(row) for row in cursor.fetchall())

            return expired

    def cancel_task(self, task_id: str) -> bool:
        """Mark a task as cancelled."""
        with self._lock, self._transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE tasks SET
                    status = 'cancelled',
                    updated_at = ?,
                    completed_at = ?
                WHERE task_id = ? AND status IN ('pending', 'running')
            """,
                (time.time(), time.time(), task_id),
            )
            return cursor.rowcount > 0

    def acquire_task(self, worker_id: str) -> TaskRecord | None:
        """Acquire a pending task for processing (claim ownership)."""
        with self._lock, self._transaction() as conn:
            # Get highest priority pending task
            cursor = conn.execute(
                """
                SELECT * FROM tasks 
                WHERE status = 'pending' 
                AND (worker_id IS NULL OR worker_id = ?)
                ORDER BY priority DESC, created_at ASC
                LIMIT 1
            """,
                (worker_id,),
            )
            row = cursor.fetchone()
            if row:
                task_id = row["task_id"]
                conn.execute(
                    """
                    UPDATE tasks SET
                        status = 'running',
                        worker_id = ?,
                        started_at = ?,
                        heartbeat_at = ?,
                        updated_at = ?
                    WHERE task_id = ?
                """,
                    (worker_id, time.time(), time.time(), time.time(), task_id),
                )
                return self._row_to_record(
                    conn.execute(
                        "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
                    ).fetchone()
                )
            return None

    def release_task(self, task_id: str) -> None:
        """Release a task back to pending (for retry)."""
        with self._lock, self._transaction() as conn:
            conn.execute(
                """
                UPDATE tasks SET
                    status = 'pending',
                    worker_id = NULL,
                    retry_count = retry_count + 1,
                    heartbeat_at = ?,
                    updated_at = ?
                WHERE task_id = ?
            """,
                (time.time(), time.time(), task_id),
            )

    def cleanup(self) -> int:
        """Clean up expired tasks, return count deleted."""
        expired = self.find_expired_tasks()
        count = 0
        for task in expired:
            if self.delete_task(task.task_id):
                count += 1
                logger.debug("Cleaned up expired task: %s", task.task_id)
        return count


class DurableTaskManager:
    """Production-ready task manager with durable storage.

    This task manager stores all task state in a durable backend (SQLite
    by default), making it safe for:
    - Multi-worker deployments
    - Process restarts and crash recovery
    - Task cancellation and retry
    - Audit and replay capabilities

    Usage:
        manager = DurableTaskManager()
        task_id = manager.create_task(
            task_type="simulation",
            input_data={"param": "value"},
            run_id="unique-run-id"  # For idempotency
        )
        manager.update_progress(task_id, 50.0)
        manager.heartbeat(task_id)  # Prevent stall detection
        manager.mark_completed(task_id, {"result": "data"})
    """

    def __init__(
        self,
        backend: StorageBackend | None = None,
        heartbeat_timeout: float = 300.0,  # 5 minutes
        auto_cleanup: bool = True,
        cleanup_interval: int = 60,  # seconds
    ):
        """Initialize durable task manager.

        Args:
            backend: Storage backend (defaults to SQLiteBackend)
            heartbeat_timeout: Seconds without heartbeat before task is stalled
            auto_cleanup: Whether to automatically clean up expired tasks
            cleanup_interval: Interval in seconds between automatic cleanup
        """
        self.backend = backend or SQLiteBackend()
        self.heartbeat_timeout = heartbeat_timeout
        self._closed = False
        self._cleanup_task: asyncio.Task | None = None

        if auto_cleanup:
            self._cleanup_interval = cleanup_interval
            # Start background cleanup in asyncio context
            try:
                loop = asyncio.get_running_loop()
                self._cleanup_task = loop.create_task(self._cleanup_loop())
            except RuntimeError:
                # No running loop, cleanup will be manual
                pass

    async def _cleanup_loop(self) -> None:
        """Background task to periodically clean up expired tasks."""
        while not self._closed:
            await asyncio.sleep(self._cleanup_interval)
            try:
                count = self.backend.cleanup()
                if count > 0:
                    logger.info("Cleaned up %d expired tasks", count)
            except Exception:  # noqa: BLE001
                logger.exception("Cleanup error")

    def create_task(
        self,
        task_type: str = "generic",
        input_data: dict[str, Any] | None = None,
        run_id: str | None = None,
        ttl_seconds: int = 3600,
        retention_seconds: int = 86400,
        priority: int = 0,
        max_retries: int = 3,
        code_version: str | None = None,
    ) -> str:
        """Create a new task.

        Args:
            task_type: Type identifier for the task
            input_data: Input parameters (JSON-serializable)
            run_id: Idempotency key - if provided and exists, returns existing task
            ttl_seconds: Time-to-live for pending/running tasks
            retention_seconds: How long to keep completed task results
            priority: Task priority (higher = more urgent)
            max_retries: Maximum retry attempts
            code_version: Code version for replay verification

        Returns:
            task_id: Unique task identifier

        Note:
            If run_id is provided and a task with that run_id exists,
            returns the existing task_id instead of creating a new one.
        """
        # Check idempotency
        if run_id:
            existing = self.backend.find_by_run_id(run_id)
            if existing:
                logger.debug(
                    "Found existing task for run_id %s: %s", run_id, existing.task_id
                )
                return existing.task_id

        # Generate task ID and config hash
        task_id = str(uuid.uuid4())
        config_hash = hashlib.sha256(
            json.dumps(input_data or {}, sort_keys=True).encode()
        ).hexdigest()[:16]

        record = TaskRecord(
            task_id=task_id,
            run_id=run_id,
            task_type=task_type,
            input_data=input_data or {},
            config_hash=config_hash,
            code_version=(
                code_version
                if code_version is not None
                else str(os.environ.get("APP_VERSION", "unknown"))
            ),
            ttl_seconds=ttl_seconds,
            retention_seconds=retention_seconds,
            priority=priority,
            max_retries=max_retries,
        )

        self.backend.create_task(record)
        logger.info(
            "Created task %s (type=%s, priority=%d)", task_id, task_type, priority
        )
        return task_id

    def get_task(self, task_id: str) -> TaskRecord | None:
        """Get task by ID."""
        return self.backend.get_task(task_id)

    def update_progress(self, task_id: str, progress: float) -> bool:
        """Update task progress.

        Args:
            task_id: Task identifier
            progress: Progress percentage (0-100)

        Returns:
            True if task was updated, False if not found
        """
        record = self.backend.get_task(task_id)
        if not record:
            return False

        record.progress = min(max(progress, 0.0), 100.0)
        record.heartbeat_at = time.time()
        self.backend.update_task(record)
        return True

    def heartbeat(self, task_id: str) -> bool:
        """Update task heartbeat to prevent stall detection.

        Args:
            task_id: Task identifier

        Returns:
            True if task was updated, False if not found
        """
        record = self.backend.get_task(task_id)
        if not record:
            return False

        record.heartbeat_at = time.time()
        self.backend.update_task(record)
        return True

    def mark_completed(self, task_id: str, result: dict[str, Any]) -> bool:
        """Mark task as completed.

        Args:
            task_id: Task identifier
            result: Result data (JSON-serializable)

        Returns:
            True if task was updated, False if not found
        """
        record = self.backend.get_task(task_id)
        if not record:
            return False

        record.status = TaskStatus.COMPLETED.value
        record.result = result
        record.progress = 100.0
        record.completed_at = time.time()
        record.heartbeat_at = time.time()
        self.backend.update_task(record)
        logger.info("Task %s completed", task_id)
        return True

    def mark_failed(self, task_id: str, error: str) -> bool:
        """Mark task as failed.

        Args:
            task_id: Task identifier
            error: Error message

        Returns:
            True if task was updated, False if not found
        """
        record = self.backend.get_task(task_id)
        if not record:
            return False

        record.status = TaskStatus.FAILED.value
        record.error = error
        record.completed_at = time.time()
        record.heartbeat_at = time.time()
        self.backend.update_task(record)
        logger.warning("Task %s failed: %s", task_id, error)
        return True

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task.

        Args:
            task_id: Task identifier

        Returns:
            True if task was cancelled, False if not found or already terminal
        """
        return self.backend.cancel_task(task_id)

    def acquire_task(self, worker_id: str | None = None) -> TaskRecord | None:
        """Acquire a pending task for processing.

        Args:
            worker_id: Worker identifier (defaults to hostname-PID)

        Returns:
            TaskRecord if a task was acquired, None otherwise
        """
        if worker_id is None:
            import socket

            worker_id = f"{socket.gethostname()}-{os.getpid()}"
        return self.backend.acquire_task(worker_id)

    def release_task(self, task_id: str) -> bool:
        """Release a task back to pending for retry.

        Args:
            task_id: Task identifier

        Returns:
            True if task was released, False if not found or exceeded retries
        """
        record = self.backend.get_task(task_id)
        if not record:
            return False

        if record.retry_count >= record.max_retries:
            logger.warning(
                "Task %s exceeded max retries (%d)", task_id, record.max_retries
            )
            return False

        self.backend.release_task(task_id)
        logger.info(
            "Released task %s for retry (attempt %d/%d)",
            task_id,
            record.retry_count + 1,
            record.max_retries,
        )
        return True

    def find_stalled_tasks(self) -> list[TaskRecord]:
        """Find tasks that have stalled."""
        return self.backend.find_stalled_tasks(self.heartbeat_timeout)

    async def shutdown(self) -> None:
        """Shutdown the task manager."""
        self._closed = True
        if self._cleanup_task:
            self._cleanup_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._cleanup_task
        logger.info("DurableTaskManager shutdown complete")
