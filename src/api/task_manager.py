"""Background task manager with TTL cleanup, concurrency limits, and job status tracking.

Extracted from server.py to follow SRP (#1485) and enhanced with:
- Concurrency semaphore for engine instances (#1488)
- Job status lifecycle (pending -> running -> completed/failed)
- Progress tracking for long-running simulations

Design by Contract:
    - Precondition: task_id must be a non-empty string
    - Postcondition: tasks are automatically cleaned up after TTL expiry
    - Invariant: at most MAX_CONCURRENT_ENGINES simulations run simultaneously
"""

from __future__ import annotations

import asyncio
import enum
import threading
import time
from typing import Any

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class TaskStatus(enum.Enum):
    """Lifecycle states for a background task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskManager:
    """Thread-safe task manager with TTL cleanup and size limits.

    Prevents memory leak from unbounded task accumulation.

    Features:
    - Tasks expire after TTL_SECONDS (default 1 hour)
    - Maximum MAX_TASKS entries with LRU eviction
    - Automatic cleanup on each access
    - Concurrency semaphore for engine instances
    """

    # Configuration constants
    TTL_SECONDS: int = 3600  # 1 hour
    MAX_TASKS: int = 1000  # Maximum stored tasks
    MAX_CONCURRENT_ENGINES: int = 4  # Concurrency limit for engine instances

    def __init__(
        self,
        *,
        ttl_seconds: int | None = None,
        max_tasks: int | None = None,
        max_concurrent: int | None = None,
    ) -> None:
        """Initialize task manager.

        Args:
            ttl_seconds: Override default TTL for task expiry.
            max_tasks: Override maximum number of stored tasks.
            max_concurrent: Override maximum concurrent engine instances.
        """
        if ttl_seconds is not None:
            self.TTL_SECONDS = ttl_seconds
        if max_tasks is not None:
            self.MAX_TASKS = max_tasks
        if max_concurrent is not None:
            self.MAX_CONCURRENT_ENGINES = max_concurrent

        self._tasks: dict[str, dict[str, Any]] = {}
        self._timestamps: dict[str, float] = {}
        self._lock = threading.Lock()
        self._engine_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_ENGINES)

    @property
    def engine_semaphore(self) -> asyncio.Semaphore:
        """Semaphore limiting concurrent engine instances.

        Usage::

            async with task_manager.engine_semaphore:
                await run_simulation(...)
        """
        return self._engine_semaphore

    def _cleanup_expired(self) -> None:
        """Remove expired tasks. Called internally under lock."""
        current_time = time.time()
        expired_keys = [
            task_id
            for task_id, timestamp in self._timestamps.items()
            if current_time - timestamp > self.TTL_SECONDS
        ]
        for task_id in expired_keys:
            self._tasks.pop(task_id, None)
            self._timestamps.pop(task_id, None)
        if expired_keys:
            logger.debug("Cleaned up %d expired tasks", len(expired_keys))

    def _enforce_size_limit(self) -> None:
        """Remove oldest tasks if over limit. Called internally under lock."""
        if len(self._tasks) <= self.MAX_TASKS:
            return

        sorted_by_age = sorted(self._timestamps.items(), key=lambda x: x[1])
        to_remove = len(self._tasks) - self.MAX_TASKS
        for task_id, _ in sorted_by_age[:to_remove]:
            self._tasks.pop(task_id, None)
            self._timestamps.pop(task_id, None)
        logger.debug("Evicted %d tasks due to size limit", to_remove)

    def set(self, task_id: str, data: dict[str, Any]) -> None:
        """Store or update a task.

        Args:
            task_id: Unique task identifier (must be non-empty).
            data: Task data dictionary.

        Raises:
            ValueError: If task_id is empty.
        """
        if not task_id or not task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        with self._lock:
            self._cleanup_expired()
            self._tasks[task_id] = data
            self._timestamps[task_id] = time.time()
            self._enforce_size_limit()

    def get(self, task_id: str) -> dict[str, Any] | None:
        """Retrieve a task by ID.

        Args:
            task_id: Task identifier.

        Returns:
            Task data or None if not found / expired.
        """
        with self._lock:
            self._cleanup_expired()
            return self._tasks.get(task_id)

    def update_progress(self, task_id: str, progress: float) -> None:
        """Update progress for a running task.

        Args:
            task_id: Task identifier.
            progress: Progress percentage (0-100).
        """
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id]["progress"] = min(max(progress, 0.0), 100.0)
                self._timestamps[task_id] = time.time()

    def mark_completed(self, task_id: str, result: dict[str, Any]) -> None:
        """Mark a task as completed with its result.

        Args:
            task_id: Task identifier.
            result: The task result data.
        """
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id]["status"] = TaskStatus.COMPLETED.value
                self._tasks[task_id]["result"] = result
                self._tasks[task_id]["progress"] = 100.0
                self._timestamps[task_id] = time.time()

    def mark_failed(self, task_id: str, error: str) -> None:
        """Mark a task as failed with error information.

        Args:
            task_id: Task identifier.
            error: Error message.
        """
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id]["status"] = TaskStatus.FAILED.value
                self._tasks[task_id]["error"] = error
                self._timestamps[task_id] = time.time()

    def active_count(self) -> int:
        """Return the number of active (non-expired) tasks."""
        with self._lock:
            self._cleanup_expired()
            return len(self._tasks)

    def __contains__(self, task_id: str) -> bool:
        """Check if task exists."""
        with self._lock:
            self._cleanup_expired()
            return task_id in self._tasks

    def __setitem__(self, task_id: str, data: dict[str, Any]) -> None:
        """Dict-like setter for backward compatibility."""
        self.set(task_id, data)

    def __getitem__(self, task_id: str) -> dict[str, Any]:
        """Dict-like getter for backward compatibility."""
        result = self.get(task_id)
        if result is None:
            raise KeyError(task_id)
        return result
