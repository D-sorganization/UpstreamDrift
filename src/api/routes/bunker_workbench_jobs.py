"""Bounded, cancellable jobs for the BunkerShot3D workbench (issue #9545).

The synchronous ``/v1/evaluate`` and ``/v1/compare`` routes hold a request
open for the whole solve; the comparison is 52 shots at the defaults. These
routes run the **same** validated solve (:class:`BoundedSolve`, built by the
same functions) on a worker thread and track it in the shared
:class:`~src.api.task_manager.TaskManager` -- the API's one job registry --
rather than a second framework.

Contract
--------

* **Submit is immediate and refuses up front.** A submit returns ``202``
  with the job id, the shot budget and the input digest. Every refusal of
  the synchronous route (left-handed, predictive, duplicate names, an
  unconstructible sole) is the same ``422``, before any job exists.
* **Bounded.** The payload is the synchronous route's strict, range-bounded
  request model and the shot budget is fixed by it. At most
  :data:`MAX_BUNKER_JOBS` workers exist; a slot is held until the worker
  thread exits, not merely until the job is marked cancelled, so a cancelled
  job still finishing its shot cannot let a new one oversubscribe the pool.
  The task manager's shared active capacity also applies. Either limit is a
  ``429``.
* **Progress and cancellation are per shot.** The model calls the job's hook
  before every shot; the hook reports progress and, once the job is
  cancelled, raises so no further shot runs.
* **No stale result masquerades as fresh.** Only a ``completed`` job carries
  a ``result``, and the result names its ``job_id`` and the ``input_digest``
  of the request it answers. A cancelled job stays cancelled even when the
  shot in flight finishes afterwards (the task manager refuses to settle a
  cancelled task); a failed job carries its error and no result.
"""

from __future__ import annotations

import logging
import re
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.api.dependencies import get_task_manager
from src.api.routes.bunker_workbench import (
    BoundedSolve,
    BunkerWorkbenchCompareRequestV1,
    BunkerWorkbenchRequestV1,
    ShotHook,
    comparison_solve,
    evaluation_solve,
    record_digest,
)
from src.api.task_manager import TaskManager, TaskStatus
from src.shared.python.core.contracts import require

logger = logging.getLogger(__name__)

__all__ = [
    "JOB_POOL",
    "MAX_BUNKER_JOBS",
    "BunkerJobPool",
    "BunkerJobStatusV1",
    "input_digest",
    "router",
]

MAX_BUNKER_JOBS = 2
"""Concurrent bunker solves; a comparison is 52 shots at the defaults."""

_KIND_PREFIX = "bunker-workbench/"
"""Marks a task-manager entry as a bunker job, so other tasks are not served."""

_JOB_ID = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"

_SOLVE_ERRORS: tuple[type[Exception], ...] = (
    ValueError,
    RuntimeError,
    ImportError,
    ArithmeticError,
    LookupError,
)
"""What a solve can raise that is a failed job, not a defect in this module."""

router = APIRouter(prefix="/tools/bunker-workbench", tags=["bunker-workbench"])


class JobCancelledError(Exception):
    """Raised from the shot hook to stop a cancelled job before its next shot."""


class BunkerJobResultV1(BaseModel):
    """A completed job's record, naming the job and the request it answers."""

    job_id: str
    input_digest: str
    record: dict[str, Any]


class BunkerJobStatusV1(BaseModel):
    """The state of one bunker job; ``result`` only when ``completed``."""

    job_id: str
    kind: Literal["evaluate", "compare"]
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    progress_percent: float = Field(ge=0.0, le=100.0)
    shots_planned: int = Field(ge=1)
    input_digest: str
    worker_active: bool
    error: str | None = None
    result: BunkerJobResultV1 | None = None


class BunkerJobPool:
    """A fixed number of worker slots, each held until its thread exits."""

    def __init__(self, max_jobs: int) -> None:
        """Create the pool.

        Raises:
            ValueError: If ``max_jobs`` is not positive.
        """
        require(max_jobs >= 1, "a job pool needs at least one slot", max_jobs)
        self._max_jobs = max_jobs
        self._live: set[str] = set()
        self._idle = threading.Condition()
        self._executor: ThreadPoolExecutor | None = None

    def try_reserve(self, job_id: str) -> bool:
        """Take a slot for ``job_id``; ``False`` when every slot is held."""
        with self._idle:
            if len(self._live) >= self._max_jobs:
                return False
            self._live.add(job_id)
            return True

    def release(self, job_id: str) -> None:
        """Free the slot of ``job_id``."""
        with self._idle:
            self._live.discard(job_id)
            self._idle.notify_all()

    def is_live(self, job_id: str) -> bool:
        """Whether a worker for ``job_id`` holds a slot."""
        with self._idle:
            return job_id in self._live

    def start(self, job_id: str, work: Callable[[], None]) -> None:
        """Run ``work`` for a reserved job; its slot is freed when it exits."""
        with self._idle:
            require(job_id in self._live, "start needs a reserved slot", job_id)
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self._max_jobs, thread_name_prefix="bunker-job"
                )
            future = self._executor.submit(work)
        future.add_done_callback(lambda done: self._finished(job_id, done))

    def _finished(self, job_id: str, future: Future[None]) -> None:
        """Log an escaped defect and free the slot."""
        error = future.exception()
        if error is not None:
            logger.error("bunker job %s crashed", job_id, exc_info=error)
        self.release(job_id)

    def drain(self, timeout_s: float) -> bool:
        """Wait until no worker holds a slot; ``False`` on timeout."""
        with self._idle:
            return self._idle.wait_for(lambda: not self._live, timeout=timeout_s)


JOB_POOL = BunkerJobPool(MAX_BUNKER_JOBS)


def input_digest(kind: str, request: BaseModel) -> str:
    """SHA-256 naming a job's kind and request, independent of the run.

    The client compares it with the digest of the inputs it shows, so a
    result for other inputs is recognisably stale.
    """
    return record_digest({"kind": kind, "request": request.model_dump(mode="json")})


def _refuse(code: int, detail_code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=code, detail={"code": detail_code, "message": message}
    )


def _shot_hook(tm: TaskManager, job_id: str, planned: int) -> ShotHook:
    """Report progress before each shot and stop a job no longer running."""
    done = 0

    def before_shot() -> None:
        nonlocal done
        task = tm.get(job_id)
        if task is None or task.get("status") != TaskStatus.RUNNING.value:
            raise JobCancelledError(job_id)
        tm.update_progress(job_id, 100.0 * min(done, planned) / planned)
        done += 1

    return before_shot


def _run_job(tm: TaskManager, job_id: str, solve: BoundedSolve, digest: str) -> None:
    """Worker body: run the solve; a job never stays ``running`` after it."""
    if not tm.mark_running(job_id):
        return
    try:
        _solve_and_settle(tm, job_id, solve, digest)
    finally:
        task = tm.get(job_id)
        if task is not None and task.get("status") == TaskStatus.RUNNING.value:
            tm.mark_failed(job_id, "the worker exited without a result")


def _solve_and_settle(
    tm: TaskManager, job_id: str, solve: BoundedSolve, digest: str
) -> None:
    """Run the solve and settle the task as completed, failed or cancelled."""
    try:
        record = solve.run(_shot_hook(tm, job_id, solve.planned_shots))
    except JobCancelledError:
        logger.info("bunker job %s stopped after cancellation", job_id)
        return
    except HTTPException as refusal:
        tm.mark_failed(job_id, str(refusal.detail))
        return
    except _SOLVE_ERRORS as error:
        logger.exception("bunker job %s failed", job_id)
        tm.mark_failed(job_id, f"{type(error).__name__}: {error}")
        return
    tm.mark_completed(
        job_id, {"job_id": job_id, "input_digest": digest, "record": record}
    )


def _status_view(tm: TaskManager, job_id: str) -> dict[str, Any]:
    """The public status of a bunker job.

    Raises:
        HTTPException: 404 when no bunker job has this id.
    """
    task = tm.get(job_id)
    if task is None or not str(task.get("kind", "")).startswith(_KIND_PREFIX):
        raise _refuse(status.HTTP_404_NOT_FOUND, "job_not_found", "no such job")
    completed = task["status"] == TaskStatus.COMPLETED.value
    result = task.get("result") if completed else None
    if result is not None:
        require(
            result["job_id"] == job_id and result["input_digest"] == task["digest"],
            "a result must name its own job and inputs",
            job_id,
        )
    return {
        "job_id": job_id,
        "kind": task["kind"].removeprefix(_KIND_PREFIX),
        "status": task["status"],
        "progress_percent": float(task.get("progress", 0.0)),
        "shots_planned": task["shots_planned"],
        "input_digest": task["digest"],
        "worker_active": JOB_POOL.is_live(job_id),
        "error": task.get("error") if task["status"] == "failed" else None,
        "result": result,
    }


def _submit(tm: TaskManager, solve: BoundedSolve, request: BaseModel) -> dict[str, Any]:
    """Admit a validated solve as a job and start its worker.

    Raises:
        HTTPException: 429 when the worker slots or the shared capacity are
            saturated; nothing is left reserved.
    """
    job_id = str(uuid.uuid4())
    digest = input_digest(solve.kind, request)
    if not JOB_POOL.try_reserve(job_id):
        raise _refuse(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "job_capacity_saturated",
            f"all {MAX_BUNKER_JOBS} bunker job slots are busy; retry later",
        )
    task = {
        "status": TaskStatus.PENDING.value,
        "kind": f"{_KIND_PREFIX}{solve.kind}",
        "progress": 0.0,
        "shots_planned": solve.planned_shots,
        "digest": digest,
    }
    try:
        tm.admit_active(job_id, task)
    except RuntimeError as saturated:
        JOB_POOL.release(job_id)
        raise _refuse(
            status.HTTP_429_TOO_MANY_REQUESTS, "job_capacity_saturated", str(saturated)
        ) from saturated
    JOB_POOL.start(job_id, lambda: _run_job(tm, job_id, solve, digest))
    return _status_view(tm, job_id)


@router.post(
    "/v1/jobs/evaluate",
    response_model=BunkerJobStatusV1,
    status_code=status.HTTP_202_ACCEPTED,
)
def submit_evaluate_job(
    request: BunkerWorkbenchRequestV1,
    tm: TaskManager = Depends(get_task_manager),
) -> dict[str, Any]:
    """Start the ``/v1/evaluate`` solve as a job; poll it by ``job_id``."""
    return _submit(tm, evaluation_solve(request), request)


@router.post(
    "/v1/jobs/compare",
    response_model=BunkerJobStatusV1,
    status_code=status.HTTP_202_ACCEPTED,
)
def submit_compare_job(
    request: BunkerWorkbenchCompareRequestV1,
    tm: TaskManager = Depends(get_task_manager),
) -> dict[str, Any]:
    """Start the ``/v1/compare`` solve as a job; poll it by ``job_id``."""
    return _submit(tm, comparison_solve(request), request)


def _job_id(job_id: str) -> str:
    """A job id path parameter; anything but a UUID is an unknown job."""
    if not re.fullmatch(_JOB_ID, job_id):
        raise _refuse(status.HTTP_404_NOT_FOUND, "job_not_found", "no such job")
    return job_id


@router.get("/v1/jobs/{job_id}", response_model=BunkerJobStatusV1)
def get_job(job_id: str, tm: TaskManager = Depends(get_task_manager)) -> dict[str, Any]:
    """The job's status and progress; its result only once ``completed``."""
    return _status_view(tm, _job_id(job_id))


@router.post("/v1/jobs/{job_id}/cancel", response_model=BunkerJobStatusV1)
def cancel_job(
    job_id: str, tm: TaskManager = Depends(get_task_manager)
) -> dict[str, Any]:
    """Cancel a pending or running job; it stops before its next shot.

    Cancelling an already cancelled job is a no-op.

    Raises:
        HTTPException: 404 for an unknown job, 409 ``job_not_cancellable``
            for one that already completed or failed.
    """
    view = _status_view(tm, _job_id(job_id))
    if view["status"] in {TaskStatus.COMPLETED.value, TaskStatus.FAILED.value}:
        raise _refuse(
            status.HTTP_409_CONFLICT,
            "job_not_cancellable",
            f"the job already {view['status']}",
        )
    tm.cancel_task(job_id)
    return _status_view(tm, job_id)
