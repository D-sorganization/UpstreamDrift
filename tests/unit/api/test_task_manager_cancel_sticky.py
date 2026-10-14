"""A cancelled task stays cancelled (issue #9545, slice 4).

Cancellation is cooperative, so a worker can finish the step it was on
after the cancel landed. Its late result must not turn the task back into
``completed`` (or ``failed``): a cancelled run would then masquerade as a
fresh result. ``mark_running`` only starts a task that is still pending.
"""

from __future__ import annotations

from src.api.task_manager import TaskManager, TaskStatus


def _pending(tm: TaskManager, task_id: str = "t1") -> None:
    tm.admit_active(task_id, {"status": TaskStatus.PENDING.value})


def test_late_completion_does_not_overwrite_a_cancellation() -> None:
    tm = TaskManager()
    _pending(tm)
    assert tm.cancel_task("t1") is True
    assert tm.mark_completed("t1", {"value": 1}) is False
    task = tm.get("t1")
    assert task is not None
    assert task["status"] == TaskStatus.CANCELLED.value
    assert "result" not in task


def test_late_failure_does_not_overwrite_a_cancellation() -> None:
    tm = TaskManager()
    _pending(tm)
    tm.cancel_task("t1")
    assert tm.mark_failed("t1", "boom") is False
    task = tm.get("t1")
    assert task is not None
    assert task["status"] == TaskStatus.CANCELLED.value


def test_completion_of_a_live_task_is_reported() -> None:
    tm = TaskManager()
    _pending(tm)
    assert tm.mark_completed("t1", {"value": 1}) is True
    task = tm.get("t1")
    assert task is not None
    assert task["status"] == TaskStatus.COMPLETED.value


def test_mark_running_starts_only_a_pending_task() -> None:
    tm = TaskManager()
    _pending(tm)
    assert tm.mark_running("t1") is True
    assert tm.get("t1")["status"] == TaskStatus.RUNNING.value  # type: ignore[index]
    assert tm.mark_running("t1") is False
    _pending(tm, "t2")
    tm.cancel_task("t2")
    assert tm.mark_running("t2") is False
    assert tm.mark_running("absent") is False
