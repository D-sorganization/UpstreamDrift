"""Tests for API architecture improvements (#1485, #1488).

Tests:
- Route registry auto-discovery and registration
- Task manager with TTL, concurrency, and lifecycle
- API versioning (routes available under /api/v1/)
- Linkage mechanisms decomposition (imports still work)
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Generator
from typing import Any

import pytest

# ── Route Registry Tests ─────────────────────────────────────────


# ── Task Manager Tests ────────────────────────────────────────────


class TestTaskManager:
    """Tests for the extracted TaskManager with TTL and concurrency (#1485, #1488)."""

    def test_api_architecture_set_and_get(self) -> None:
        """Basic set/get operations work."""
        from src.api.task_manager import TaskManager

        tm = TaskManager()
        tm.set("task-1", {"status": "running"})
        result = tm.get("task-1")
        assert result is not None
        assert result["status"] == "running"

    def test_get_nonexistent_returns_none(self) -> None:
        """Getting a nonexistent task returns None."""
        from src.api.task_manager import TaskManager

        tm = TaskManager()
        assert tm.get("nonexistent") is None

    def test_contains(self) -> None:
        """__contains__ check works."""
        from src.api.task_manager import TaskManager

        tm = TaskManager()
        tm.set("task-1", {"status": "pending"})
        assert "task-1" in tm
        assert "task-2" not in tm

    def test_dict_like_access(self) -> None:
        """Dict-like [] access works for backward compatibility."""
        from src.api.task_manager import TaskManager

        tm = TaskManager()
        tm["task-1"] = {"status": "pending"}
        assert tm["task-1"]["status"] == "pending"

    def test_dict_like_access_raises_keyerror(self) -> None:
        """Dict-like [] access raises KeyError for missing tasks."""
        from src.api.task_manager import TaskManager

        tm = TaskManager()
        with pytest.raises(KeyError):
            _ = tm["missing"]

    def test_empty_task_id_raises(self) -> None:
        """Setting a task with empty ID raises ValueError."""
        from src.api.task_manager import TaskManager

        tm = TaskManager()
        with pytest.raises(ValueError, match="non-empty"):
            tm.set("", {"status": "bad"})
        with pytest.raises(ValueError, match="non-empty"):
            tm.set("   ", {"status": "bad"})

    def test_ttl_expiry(self) -> None:
        """Tasks are cleaned up after TTL expiry."""
        from src.api.task_manager import TaskManager

        tm = TaskManager(ttl_seconds=0)  # Immediately expire
        tm.set("task-1", {"status": "running"})
        time.sleep(0.01)
        # Next access triggers cleanup
        assert tm.get("task-1") is None
        assert "task-1" not in tm

    def test_max_tasks_eviction(self) -> None:
        """Oldest tasks are evicted when MAX_TASKS is exceeded."""
        from src.api.task_manager import TaskManager

        tm = TaskManager(max_tasks=3)
        tm.set("task-1", {"status": "running"})
        tm.set("task-2", {"status": "running"})
        tm.set("task-3", {"status": "running"})
        tm.set("task-4", {"status": "running"})  # Should evict task-1
        assert "task-1" not in tm
        assert "task-4" in tm

    def test_update_progress(self) -> None:
        """update_progress updates the progress field."""
        from src.api.task_manager import TaskManager

        tm = TaskManager()
        tm.set("task-1", {"status": "running"})
        tm.update_progress("task-1", 42.5)
        result = tm.get("task-1")
        assert result is not None
        assert result["progress"] == 42.5

    def test_update_progress_clamped(self) -> None:
        """update_progress clamps to 0-100 range."""
        from src.api.task_manager import TaskManager

        tm = TaskManager()
        tm.set("task-1", {"status": "running"})
        tm.update_progress("task-1", 150.0)
        assert tm.get("task-1")["progress"] == 100.0
        tm.update_progress("task-1", -10.0)
        assert tm.get("task-1")["progress"] == 0.0

    def test_mark_completed(self) -> None:
        """mark_completed updates status and result."""
        from src.api.task_manager import TaskManager, TaskStatus

        tm = TaskManager()
        tm.set("task-1", {"status": "running"})
        tm.mark_completed("task-1", {"data": [1, 2, 3]})
        result = tm.get("task-1")
        assert result is not None
        assert result["status"] == TaskStatus.COMPLETED.value
        assert result["result"] == {"data": [1, 2, 3]}
        assert result["progress"] == 100.0

    def test_mark_failed(self) -> None:
        """mark_failed updates status and error."""
        from src.api.task_manager import TaskManager, TaskStatus

        tm = TaskManager()
        tm.set("task-1", {"status": "running"})
        tm.mark_failed("task-1", "Engine crashed")
        result = tm.get("task-1")
        assert result is not None
        assert result["status"] == TaskStatus.FAILED.value
        assert result["error"] == "Engine crashed"

    def test_active_count(self) -> None:
        """active_count returns correct count of non-expired tasks."""
        from src.api.task_manager import TaskManager

        tm = TaskManager()
        assert tm.active_count() == 0
        tm.set("task-1", {"status": "running"})
        tm.set("task-2", {"status": "running"})
        assert tm.active_count() == 2

    def test_engine_semaphore_property(self) -> None:
        """engine_semaphore returns an asyncio.Semaphore with correct limit."""
        from src.api.task_manager import TaskManager

        tm = TaskManager(max_concurrent=2)
        sem = tm.engine_semaphore
        assert isinstance(sem, asyncio.Semaphore)
        # Semaphore should allow 2 acquisitions
        assert sem._value == 2  # noqa: SLF001

    def test_task_status_enum(self) -> None:
        """TaskStatus enum has expected values."""
        from src.api.task_manager import TaskStatus

        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"


# ── Dict-like Compatibility Tests (#4843) ────────────────────────


# ── API Versioning Tests ──────────────────────────────────────────


# ── Linkage Decomposition Tests ──────────────────────────────────


# ── OpenAPI Enhancement Tests ─────────────────────────────────────
