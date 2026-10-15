"""Async, cancellable loop-nullspace report jobs (NSX-40, issue #12572).

* ``POST   /analysis/loop-nullspace/jobs``       -- submit; returns the job id.
* ``GET    /analysis/loop-nullspace/jobs/{id}``  -- status, progress and, while
  running, the **partial invariants** (they are computed before the envelopes).
* ``DELETE /analysis/loop-nullspace/jobs/{id}``  -- cancel.

The job machinery is the API's one registry, not a new framework: the shared
:class:`~src.api.task_manager.TaskManager` tracks the lifecycle and
:class:`~src.api.routes.bunker_workbench_jobs.BunkerJobPool` bounds the worker
threads (a slot is held until the thread exits).  The work itself is
``loop_nullspace_service.run`` (service -> facade).

Contract
--------

* **Refusals are up front.** An unknown source or preset, or invalid options,
  is a ``422`` before any job exists.
* **A cache hit is a finished job.** An identical request (same trajectory and
  model provenance, registry version, preset and options) returns ``202`` with
  a new job id already ``completed`` and ``cache_hit: true``; nothing recomputes.
* **Cancellation is cooperative at stage boundaries.** ``DELETE`` marks the job
  ``cancelled`` at once; the worker stops before its next stage and a late
  result is never attached to a cancelled job nor cached.
* **A failed job carries its error and no result.**
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Mapping
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.api.dependencies import get_task_manager
from src.api.job_cancellation import cancel_unsettled_job
from src.api.routes.bunker_workbench_jobs import BunkerJobPool
from src.api.services import loop_nullspace_service as service
from src.api.task_manager import TaskManager, TaskStatus
from src.shared.python.loop_nullspace.facade import AnalysisCancelledError

logger = logging.getLogger(__name__)

__all__ = [
    "JOB_POOL",
    "MAX_LOOP_NULLSPACE_JOBS",
    "LoopNullspaceJobStatusV1",
    "router",
]

MAX_LOOP_NULLSPACE_JOBS = 2
_KIND = "loop-nullspace/analysis"
_JOB_ID = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
_RUN_ERRORS: tuple[type[Exception], ...] = (
    ValueError,
    TypeError,
    RuntimeError,
    ImportError,
    ArithmeticError,
    LookupError,
    OSError,
)
"""What a run can raise that is a failed job rather than a defect here."""

router = APIRouter(prefix="/analysis/loop-nullspace", tags=["loop-nullspace"])
JOB_POOL = BunkerJobPool(MAX_LOOP_NULLSPACE_JOBS)


class LoopNullspaceJobStatusV1(BaseModel):
    """One job; ``result`` only when ``completed``, ``partial`` only before."""

    job_id: str
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    stage: str | None = None
    progress_percent: float = Field(ge=0.0, le=100.0)
    cache_key: str
    cache_hit: bool
    worker_active: bool
    error: str | None = None
    partial: dict[str, Any] | None = None
    result: dict[str, Any] | None = None


def _refuse(code: int, detail_code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=code, detail={"code": detail_code, "message": message}
    )


def _status_view(tm: TaskManager, job_id: str) -> dict[str, Any]:
    """The public status of a job (404 when no loop-nullspace job has this id)."""
    task = tm.get(job_id)
    if task is None or task.get("kind") != _KIND:
        raise _refuse(status.HTTP_404_NOT_FOUND, "job_not_found", "no such job")
    state = task["status"]
    completed = state == TaskStatus.COMPLETED.value
    return {
        "job_id": job_id,
        "status": state,
        "stage": task.get("stage"),
        "progress_percent": float(task.get("progress", 0.0)),
        "cache_key": task["cache_key"],
        "cache_hit": bool(task.get("cache_hit", False)),
        "worker_active": JOB_POOL.is_live(job_id),
        "error": task.get("error") if state == TaskStatus.FAILED.value else None,
        "partial": None if completed else task.get("partial"),
        "result": task.get("result") if completed else None,
    }


def _running(tm: TaskManager, job_id: str) -> bool:
    task = tm.get(job_id)
    return task is not None and task.get("status") == TaskStatus.RUNNING.value


def _stage_hook(tm: TaskManager, job_id: str):
    """Record each finished stage; its partial result is shown while it runs."""

    def on_stage(stage: str, percent: float, partial: Mapping[str, Any] | None) -> None:
        task = tm.get(job_id)
        if task is None or task.get("status") != TaskStatus.RUNNING.value:
            return
        task["stage"] = stage
        if partial is not None:
            task["partial"] = dict(partial)
        tm.update_progress(job_id, percent)

    return on_stage


def _run_job(
    tm: TaskManager, job_id: str, request: service.LoopNullspaceRequest
) -> None:
    """Worker body: run the service; a job never stays ``running`` after it."""
    if not tm.mark_running(job_id):
        return
    try:
        report = service.run(
            request,
            on_stage=_stage_hook(tm, job_id),
            should_cancel=lambda: not _running(tm, job_id),
        )
        tm.mark_completed(job_id, report)
    except AnalysisCancelledError:
        logger.info("loop-nullspace job %s stopped after cancellation", job_id)
    except service.RequestRefusedError as refusal:
        tm.mark_failed(job_id, f"{refusal.code}: {refusal}")
    except _RUN_ERRORS as error:
        logger.exception("loop-nullspace job %s failed", job_id)
        tm.mark_failed(job_id, f"{type(error).__name__}: {error}")
    finally:
        if _running(tm, job_id):
            tm.mark_failed(job_id, "the worker exited without a result")


def _new_task(key: str) -> dict[str, Any]:
    return {
        "status": TaskStatus.PENDING.value,
        "kind": _KIND,
        "progress": 0.0,
        "cache_key": key,
        "cache_hit": False,
    }


def _admit_hit(tm: TaskManager, key: str, report: dict[str, Any]) -> dict[str, Any]:
    job_id = str(uuid.uuid4())
    task = _new_task(key)
    task.update(
        status=TaskStatus.COMPLETED.value, progress=100.0, cache_hit=True, result=report
    )
    tm.set(job_id, task)
    return _status_view(tm, job_id)


def _admit_run(
    tm: TaskManager, key: str, request: service.LoopNullspaceRequest
) -> dict[str, Any]:
    job_id = str(uuid.uuid4())
    if not JOB_POOL.try_reserve(job_id):
        raise _refuse(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "job_capacity_saturated",
            f"all {MAX_LOOP_NULLSPACE_JOBS} loop-nullspace job slots are busy; retry later",
        )
    try:
        tm.admit_active(job_id, _new_task(key))
    except RuntimeError as saturated:
        JOB_POOL.release(job_id)
        raise _refuse(
            status.HTTP_429_TOO_MANY_REQUESTS, "job_capacity_saturated", str(saturated)
        ) from saturated
    JOB_POOL.start(job_id, lambda: _run_job(tm, job_id, request))
    return _status_view(tm, job_id)


@router.post(
    "/jobs",
    response_model=LoopNullspaceJobStatusV1,
    status_code=status.HTTP_202_ACCEPTED,
)
def submit_job(
    request: service.LoopNullspaceRequest,
    tm: TaskManager = Depends(get_task_manager),
) -> dict[str, Any]:
    """Start the analysis as a job (or answer from the cache); poll by ``job_id``.

    Raises:
        HTTPException: 422 for a refused request, 429 when every worker slot or
            the task manager's active capacity is busy.
    """
    try:
        key = service.cache_key(request)
    except service.RequestRefusedError as refusal:
        raise _refuse(
            status.HTTP_422_UNPROCESSABLE_ENTITY, refusal.code, str(refusal)
        ) from refusal
    cached = service.RESULT_CACHE.get(key)
    if cached is not None:
        return _admit_hit(tm, key, cached)
    return _admit_run(tm, key, request)


def _job_id(job_id: str) -> str:
    """A job id path parameter; anything but a UUID is an unknown job."""
    if not re.fullmatch(_JOB_ID, job_id):
        raise _refuse(status.HTTP_404_NOT_FOUND, "job_not_found", "no such job")
    return job_id


@router.get("/jobs/{job_id}", response_model=LoopNullspaceJobStatusV1)
def get_job(job_id: str, tm: TaskManager = Depends(get_task_manager)) -> dict[str, Any]:
    """Status and progress; partial invariants while running, the report once done."""
    return _status_view(tm, _job_id(job_id))


@router.delete("/jobs/{job_id}", response_model=LoopNullspaceJobStatusV1)
def cancel_job(
    job_id: str, tm: TaskManager = Depends(get_task_manager)
) -> dict[str, Any]:
    """Cancel a pending or running job (a no-op if already cancelled).

    Raises:
        HTTPException: 404 for an unknown job, 409 ``job_not_cancellable`` for
            one that already completed or failed.
    """
    view = _status_view(tm, _job_id(job_id))
    cancel_unsettled_job(tm, job_id, view["status"])
    return _status_view(tm, job_id)
