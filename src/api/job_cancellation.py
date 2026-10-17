"""Shared cancel step of the cooperative job routes (bunker workbench, NSX-40)."""

from __future__ import annotations

from fastapi import HTTPException, status

from src.api.task_manager import TaskManager, TaskStatus

__all__ = ["cancel_unsettled_job"]

_SETTLED = {TaskStatus.COMPLETED.value, TaskStatus.FAILED.value}


def cancel_unsettled_job(tm: TaskManager, job_id: str, state: str) -> None:
    """Cancel a pending, running or already cancelled job.

    Raises:
        HTTPException: 409 ``job_not_cancellable`` when ``state`` is
            ``completed`` or ``failed``.
    """
    if state in _SETTLED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "job_not_cancellable",
                "message": f"the job already {state}",
            },
        )
    tm.cancel_task(job_id)
