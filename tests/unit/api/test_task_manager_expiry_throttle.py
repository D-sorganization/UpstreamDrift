"""Regression tests for TaskManager throttled expiry sweep (issue #6992).

The in-memory ``TaskManager`` used to run an O(n) expiry sweep over every
tracked task on *every* read/membership/iteration op. Status polling hammers
those paths, so the sweep is now throttled: the full scan runs at most once
per ``CLEANUP_INTERVAL_SECONDS`` on the hot read path, while writes and
aggregate ops force an exact sweep. These tests pin that behavior.
"""

from __future__ import annotations

import time

import pytest

from src.api.task_manager import TaskManager


def test_read_path_does_not_sweep_all_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """get/exists/contains must not run the full O(n) purge on every call."""
    tm = TaskManager()
    for i in range(50):
        tm.set(f"task-{i}", {"status": "running"})

    purge_calls = {"count": 0}
    original = tm._purge_expired_locked

    def _counting_purge(current_time: float) -> None:
        purge_calls["count"] += 1
        original(current_time)

    monkeypatch.setattr(tm, "_purge_expired_locked", _counting_purge)

    # Hammer the read path the way status polling does.
    for _ in range(100):
        tm.exists("task-0")
        _ = "task-0" in tm
        tm.get("task-0")

    # Throttled: the hot path triggers at most a single (or zero) full sweep
    # within the cleanup interval, NOT one per call.
    assert purge_calls["count"] <= 1


def test_expired_task_not_returned_despite_throttle() -> None:
    """A logically-expired task is never served, even if not yet purged."""
    # Long cleanup interval so the throttled read path never force-purges;
    # tiny TTL so the entry is logically expired after a brief wait.
    tm = TaskManager(ttl_seconds=0.01)
    tm.CLEANUP_INTERVAL_SECONDS = 1000.0
    tm.set("gone", {"status": "running"})
    time.sleep(0.05)

    # Read path is throttled (no forced purge), but TTL correctness holds.
    assert tm.get("gone") is None
    assert not tm.exists("gone")
    assert "gone" not in tm
    with pytest.raises(KeyError):
        _ = tm["gone"]


def test_active_count_forces_exact_sweep() -> None:
    """active_count/len report only non-expired tasks (force a sweep)."""
    tm = TaskManager(ttl_seconds=0.01)
    tm.set("a", {"status": "running"})
    time.sleep(0.05)
    # active_count/len force an exact sweep, so the expired entry is gone.
    assert tm.active_count() == 0
    assert len(tm) == 0


def test_fresh_tasks_survive() -> None:
    """Non-expired tasks remain readable across throttled reads."""
    tm = TaskManager(ttl_seconds=3600)
    tm.set("live", {"status": "running"})
    for _ in range(20):
        assert tm.exists("live")
        assert tm.get("live") == {"status": "running"}
