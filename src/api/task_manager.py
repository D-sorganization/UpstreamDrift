"""Background task manager with TTL cleanup, concurrency limits, and job status tracking.

Extracted from server.py to follow SRP (#1485) and enhanced with:
- Concurrency semaphore for engine instances (#1488)
- Job status lifecycle (pending -> running -> completed/failed)
- Progress tracking for long-running simulations

Design by Contract:
    - Precondition: task_id must be a non-empty string
    - Postcondition: tasks are automatically cleaned up after TTL expiry
    - Invariant: at most MAX_CONCURRENT_ENGINES simulations run simultaneously

Compatibility contract (#4843):
    Public mutation/query methods (`set`, `get`, `exists`, `update_progress`,
    `mark_completed`, `mark_failed`, `active_count`) are synchronous and
    return raw values directly. Dict-like ``tm[task_id]``,
    ``tm[task_id] = ...``, ``task_id in tm``, ``len(tm)``, ``iter(tm)``,
    ``tm.keys()``, ``tm.values()``, and ``tm.items()`` are also supported.
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


def _validate_task_id(task_id: str) -> None:
    """Precondition: task_id must be a non-empty, non-whitespace string."""
    if not isinstance(task_id, str) or not task_id or not task_id.strip():
        raise ValueError("task_id must be a non-empty string")


class TaskManager:
    """Thread-safe task manager with TTL cleanup, size limits, and dict-like access.

    Prevents memory leak from unbounded task accumulation.

    Features:
    - Tasks expire after TTL_SECONDS (default 1 hour)
    - Maximum MAX_TASKS entries with LRU eviction
    - Automatic cleanup on each access
    - Concurrency semaphore for engine instances
    - Synchronous API and dict-like access (#4843)
    """

    # Configuration constants
    TTL_SECONDS: int = 3600  # 1 hour
    MAX_TASKS: int = 1000  # Maximum stored tasks
    MAX_CONCURRENT_ENGINES: int = 4  # Concurrency limit for engine instances
    # Minimum seconds between full O(n) expiry sweeps. Read/membership ops
    # are O(1000)-hammered by status polling, so the sweep is throttled
    # rather than run on every access (issue #6992).
    CLEANUP_INTERVAL_SECONDS: float = 5.0

    def __init__(
        self,
        *,
        ttl_seconds: int | None = None,
        max_tasks: int | None = None,
        max_concurrent: int | None = None,
    ) -> None:
        """Initialize task manager.

        Note: Issue #2715 — async semaphore for engine concurrency.
              Issue #4843 — synchronous core protected by ``threading.RLock``
              so calls work whether or not an event loop is running.
        """
        if ttl_seconds is not None:
            self.TTL_SECONDS = ttl_seconds
        if max_tasks is not None:
            self.MAX_TASKS = max_tasks
        if max_concurrent is not None:
            self.MAX_CONCURRENT_ENGINES = max_concurrent

        self._tasks: dict[str, dict[str, Any]] = {}
        self._timestamps: dict[str, float] = {}
        self._lock = threading.RLock()
        self._engine_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_ENGINES)
        self._closed = False
        # Monotonic timestamp of the last full expiry sweep (issue #6992).
        self._last_cleanup: float = 0.0

    def _ensure_open(self) -> None:
        """Raise when callers try to use a shutdown manager."""
        if self._closed:
            raise RuntimeError("TaskManager is closed")

    def _purge_expired_locked(self, current_time: float) -> None:
        """Run the full O(n) expiry sweep. Caller must hold ``self._lock``.

        Records the sweep time so :meth:`_cleanup_expired_locked` can throttle
        subsequent invocations (issue #6992).
        """
        self._last_cleanup = current_time
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

    def _cleanup_expired_locked(self, *, force: bool = False) -> None:
        """Throttled expiry sweep. Caller must hold ``self._lock``.

        The underlying sweep is O(n) over every tracked task, yet it is
        invoked on every read/membership/iteration op — and status polling
        hammers those paths. To keep the hot read path effectively O(1),
        the full sweep runs at most once per ``CLEANUP_INTERVAL_SECONDS``
        unless ``force`` is set. Because tasks only *expire* (never become
        unexpired), deferring the sweep is safe: callers that depend on an
        item being absent after TTL (``get``/``exists``) re-check membership,
        and an entry lingering a few extra seconds before physical removal is
        within the TTL contract's tolerance (issue #6992).
        """
        current_time = time.time()
        if (
            not force
            and current_time - self._last_cleanup < self.CLEANUP_INTERVAL_SECONDS
        ):
            return
        self._purge_expired_locked(current_time)

    def _is_expired_locked(self, task_id: str) -> bool:
        """Return whether ``task_id`` is past its TTL. Caller holds the lock.

        Used by the read path so throttled physical cleanup never returns a
        logically-expired task (issue #6992).
        """
        ts = self._timestamps.get(task_id)
        if ts is None:
            return True
        return time.time() - ts > self.TTL_SECONDS

    def _enforce_size_limit_locked(self) -> None:
        """Evict oldest tasks if over limit. Caller must hold ``self._lock``."""
        overflow = len(self._tasks) - self.MAX_TASKS
        if overflow <= 0:
            return
        sorted_by_age = sorted(self._timestamps.items(), key=lambda x: x[1])
        for task_id, _ in sorted_by_age[:overflow]:
            self._tasks.pop(task_id, None)
            self._timestamps.pop(task_id, None)
        logger.debug("Evicted %d tasks due to size limit", overflow)

    @property
    def engine_semaphore(self) -> asyncio.Semaphore:
        """Semaphore limiting concurrent engine instances."""
        self._ensure_open()
        return self._engine_semaphore

    def set(self, task_id: str, data: dict[str, Any]) -> None:
        """Store or update a task. Raises ``ValueError`` for empty IDs."""
        _validate_task_id(task_id)
        with self._lock:
            self._ensure_open()
            # Force the sweep on writes so storage stays bounded even when
            # reads have been throttling cleanup (issue #6992).
            self._cleanup_expired_locked(force=True)
            self._tasks[task_id] = data
            self._timestamps[task_id] = time.time()
            self._enforce_size_limit_locked()

    def get(self, task_id: str) -> dict[str, Any] | None:
        """Return the task dict, or ``None`` if absent or expired."""
        with self._lock:
            self._ensure_open()
            self._cleanup_expired_locked()
            if self._is_expired_locked(task_id):
                return None
            task = self._tasks.get(task_id)
            if task is not None:
                self._timestamps[task_id] = time.time()
            return task

    def exists(self, task_id: str) -> bool:
        """Check if task exists and is not expired."""
        with self._lock:
            self._ensure_open()
            self._cleanup_expired_locked()
            present = task_id in self._tasks and not self._is_expired_locked(task_id)
            if present:
                self._timestamps[task_id] = time.time()
            return present

    def update_progress(self, task_id: str, progress: float) -> None:
        """Update progress for a running task. Progress is clamped to [0, 100]."""
        with self._lock:
            self._ensure_open()
            self._cleanup_expired_locked()
            if task_id in self._tasks and not self._is_expired_locked(task_id):
                self._tasks[task_id]["progress"] = min(max(progress, 0.0), 100.0)
                self._timestamps[task_id] = time.time()

    def mark_completed(self, task_id: str, result: dict[str, Any]) -> None:
        """Mark a task as completed with its result."""
        with self._lock:
            self._ensure_open()
            self._cleanup_expired_locked()
            if task_id in self._tasks and not self._is_expired_locked(task_id):
                self._tasks[task_id]["status"] = TaskStatus.COMPLETED.value
                self._tasks[task_id]["result"] = result
                self._tasks[task_id]["progress"] = 100.0
                self._timestamps[task_id] = time.time()

    def mark_failed(self, task_id: str, error: str) -> None:
        """Mark a task as failed with error information."""
        with self._lock:
            self._ensure_open()
            self._cleanup_expired_locked()
            if task_id in self._tasks and not self._is_expired_locked(task_id):
                self._tasks[task_id]["status"] = TaskStatus.FAILED.value
                self._tasks[task_id]["error"] = error
                self._timestamps[task_id] = time.time()

    def active_count(self) -> int:
        """Return the number of active (non-expired) tasks."""
        with self._lock:
            self._ensure_open()
            # Force an exact sweep so the reported count never includes
            # logically-expired-but-not-yet-purged tasks (issue #6992).
            self._cleanup_expired_locked(force=True)
            return len(self._tasks)

    # ── Dict-like compatibility surface (#4843) ──────────────────────

    def __contains__(self, task_id: object) -> bool:
        if not isinstance(task_id, str):
            return False
        with self._lock:
            self._ensure_open()
            self._cleanup_expired_locked()
            if task_id in self._tasks and not self._is_expired_locked(task_id):
                # Refresh the TTL on membership so dict-style polling
                # (``id in tm``) keeps long-running tasks alive,
                # consistent with ``exists()`` / ``get()``. See #4871.
                self._timestamps[task_id] = time.time()
                return True
            return False

    def __getitem__(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            self._ensure_open()
            self._cleanup_expired_locked()
            if task_id not in self._tasks or self._is_expired_locked(task_id):
                raise KeyError(task_id)
            self._timestamps[task_id] = time.time()
            return self._tasks[task_id]

    def __setitem__(self, task_id: str, data: dict[str, Any]) -> None:
        self.set(task_id, data)

    def __delitem__(self, task_id: str) -> None:
        with self._lock:
            self._ensure_open()
            if task_id not in self._tasks:
                raise KeyError(task_id)
            self._tasks.pop(task_id, None)
            self._timestamps.pop(task_id, None)

    def __len__(self) -> int:
        with self._lock:
            self._ensure_open()
            # Snapshot/aggregate ops force an exact sweep (not the hot poll
            # path) so callers see only non-expired tasks (issue #6992).
            self._cleanup_expired_locked(force=True)
            return len(self._tasks)

    def __iter__(self) -> Any:
        with self._lock:
            self._ensure_open()
            self._cleanup_expired_locked(force=True)
            return iter(list(self._tasks.keys()))

    def keys(self) -> list[str]:
        """Snapshot list of active (non-expired) task IDs."""
        with self._lock:
            self._ensure_open()
            self._cleanup_expired_locked(force=True)
            return list(self._tasks.keys())

    def values(self) -> list[dict[str, Any]]:
        """Snapshot list of active task data dicts."""
        with self._lock:
            self._ensure_open()
            self._cleanup_expired_locked(force=True)
            return list(self._tasks.values())

    def items(self) -> list[tuple[str, dict[str, Any]]]:
        """Snapshot list of (task_id, data) pairs."""
        with self._lock:
            self._ensure_open()
            self._cleanup_expired_locked(force=True)
            return list(self._tasks.items())

    async def shutdown(self) -> None:
        """Cleanup resources on TaskManager shutdown (issue #2715)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._tasks.clear()
            self._timestamps.clear()
            logger.debug("TaskManager shutdown: semaphore cleanup scheduled")
