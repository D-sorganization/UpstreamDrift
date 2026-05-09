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


# ── Dict-like Compatibility Tests (#4843) ────────────────────────


class TestTaskManagerDictContract:
    """Pins the synchronous/dict-like compatibility contract (#4843)."""

    def test_getitem_returns_task_data(self) -> None:
        """``tm[task_id]`` returns the stored dict."""
        from src.api.task_manager import TaskManager

        tm = TaskManager()
        tm.set("task-1", {"status": "running", "progress": 10})
        assert tm["task-1"] == {"status": "running", "progress": 10}

    def test_getitem_missing_raises_keyerror(self) -> None:
        """``tm[missing]`` raises ``KeyError``."""
        from src.api.task_manager import TaskManager

        tm = TaskManager()
        with pytest.raises(KeyError):
            _ = tm["missing"]

    def test_setitem_delegates_to_set(self) -> None:
        """``tm[task_id] = data`` validates and stores."""
        from src.api.task_manager import TaskManager

        tm = TaskManager()
        tm["task-1"] = {"status": "pending"}
        assert tm["task-1"]["status"] == "pending"
        with pytest.raises(ValueError, match="non-empty"):
            tm[""] = {"status": "bad"}

    def test_delitem_removes_task(self) -> None:
        """``del tm[task_id]`` removes the task."""
        from src.api.task_manager import TaskManager

        tm = TaskManager()
        tm.set("task-1", {"status": "pending"})
        del tm["task-1"]
        assert "task-1" not in tm
        with pytest.raises(KeyError):
            del tm["task-1"]

    def test_contains_only_matches_strings(self) -> None:
        """``__contains__`` returns False for non-string keys without raising."""
        from src.api.task_manager import TaskManager

        tm = TaskManager()
        tm.set("task-1", {})
        assert "task-1" in tm
        assert 42 not in tm
        assert None not in tm

    def test_len_reflects_active_tasks(self) -> None:
        """``len(tm)`` matches the count of non-expired tasks."""
        from src.api.task_manager import TaskManager

        tm = TaskManager()
        assert len(tm) == 0
        tm.set("task-1", {})
        tm.set("task-2", {})
        assert len(tm) == 2

    def test_iter_yields_task_ids(self) -> None:
        """Iterating yields a snapshot of currently active task IDs."""
        from src.api.task_manager import TaskManager

        tm = TaskManager()
        tm.set("task-1", {})
        tm.set("task-2", {})
        assert sorted(tm) == ["task-1", "task-2"]

    def test_keys_values_items_snapshots(self) -> None:
        """``keys``/``values``/``items`` return snapshot lists."""
        from src.api.task_manager import TaskManager

        tm = TaskManager()
        tm.set("task-1", {"status": "running"})
        tm.set("task-2", {"status": "pending"})

        keys = tm.keys()
        values = tm.values()
        items = tm.items()

        assert sorted(keys) == ["task-1", "task-2"]
        assert {"status": "running"} in values
        assert ("task-1", {"status": "running"}) in items

        # Snapshot is detached: mutating tm afterward does not change result.
        tm.set("task-3", {})
        assert "task-3" not in keys

    def test_synchronous_calls_do_not_return_coroutines(self) -> None:
        """Regression: methods must NOT return coroutine objects (#4843)."""
        import inspect

        from src.api.task_manager import TaskManager

        tm = TaskManager()
        assert not inspect.iscoroutine(tm.set("task-1", {"status": "running"}))
        assert not inspect.iscoroutine(tm.get("task-1"))
        assert not inspect.iscoroutine(tm.exists("task-1"))
        assert not inspect.iscoroutine(tm.update_progress("task-1", 50.0))
        assert not inspect.iscoroutine(tm.active_count())
        assert isinstance(tm.active_count(), int)
        assert isinstance(tm.exists("task-1"), bool)


# ── API Versioning Tests ──────────────────────────────────────────


# ── Linkage Decomposition Tests ──────────────────────────────────


# ── OpenAPI Enhancement Tests ─────────────────────────────────────
