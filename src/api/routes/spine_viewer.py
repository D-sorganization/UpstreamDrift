"""Spine Viewer web routes (SPN-15c, #12815): React parity with the PyQt viewer.

* ``GET /tools/spine-viewer/receipts`` lists study receipts under the allowed
  study root (``UPSTREAMDRIFT_SPINE_STUDY_ROOT``, default ``data/spine_studies``).
* ``GET /tools/spine-viewer/study?receipt=<relative path>`` opens one receipt
  through ``study_aggregation.load_verified_run`` (artifact SHA-256 verified
  server-side) and returns the core view model, which always carries
  ``scientific_status`` and ``distribution_source``.

Routes are local-only (they read local evidence). Receipt paths must be
relative, may not contain ``..`` and may not traverse symlinks.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Literal
import uuid

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from pydantic import BaseModel

from src.api.dependencies import get_task_manager
from src.api.routes.bunker_workbench_jobs import BunkerJobPool
from src.api.routes.matched_swings import require_local_client
from src.api.services import spine_viewer_service as service
from src.api.task_manager import TaskManager, TaskStatus
from src.api.utils.path_validation import resolve_contained_path
from src.engines.physics_engines.opensim.python.spine.study import (
    SpineStudyRequest,
    run_spine_study,
)
from src.engines.physics_engines.opensim.python.spine.study_aggregation import (
    AggregationIntegrityError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools/spine-viewer", tags=["spine-viewer"])

MAX_SPINE_STUDY_JOBS = 2
_SPINE_STUDY_KIND = "spine-viewer/study"
JOB_POOL = BunkerJobPool(MAX_SPINE_STUDY_JOBS)

_STUDY_ERRORS: tuple[type[Exception], ...] = (
    ValueError,
    TypeError,
    RuntimeError,
    ImportError,
    ArithmeticError,
    LookupError,
    OSError,
)


class SpineStudyJobStatusV1(BaseModel):
    """Lifecycle and result information for a spine study background job."""

    job_id: str
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    receipt_path: str | None = None
    error: str | None = None


def _resolve_receipt(receipt: str) -> Path:
    """Map a client path to a file inside the study root (400/404 otherwise)."""
    try:
        relative = service.safe_relative_receipt(receipt)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    root = service.study_root()
    return resolve_contained_path(root.joinpath(*relative.parts), [root])


def _run_study_job(
    tm: TaskManager,
    job_id: str,
    request: SpineStudyRequest,
    root: Path,
) -> None:
    """Worker task executing the study pipeline in the background."""
    if not tm.mark_running(job_id):
        return
    try:
        run_spine_study(request)
        receipt_file = request.output_dir / f"{request.study_id}_receipt.json"
        if not receipt_file.is_file():
            candidates = sorted(request.output_dir.glob("*_receipt.json"))
            if candidates:
                receipt_file = candidates[0]
        try:
            rel_path = receipt_file.relative_to(root).as_posix()
        except ValueError:
            rel_path = receipt_file.name
        tm.mark_completed(job_id, {"receipt_path": rel_path})
    except _STUDY_ERRORS as exc:
        logger.exception("Spine study job %s failed: %s", job_id, exc)
        tm.mark_failed(job_id, f"{type(exc).__name__}: {exc}")
    finally:
        task = tm.get(job_id)
        if task is not None and task.get("status") == TaskStatus.RUNNING.value:
            tm.mark_failed(job_id, "the worker exited without a result")


@router.get("/receipts")
def list_spine_receipts(
    _local: None = Depends(require_local_client),
) -> dict[str, object]:
    """Relative paths of the receipts available under the study root."""
    return {"receipts": service.list_receipts(service.study_root())}


@router.get("/assets")
def list_spine_assets(
    _local: None = Depends(require_local_client),
) -> dict[str, Any]:
    """Asset manifest entries with admission status and the fixture model."""
    return {"assets": service.list_spine_assets()}


@router.post(
    "/studies",
    response_model=SpineStudyJobStatusV1,
    status_code=status.HTTP_202_ACCEPTED,
)
def submit_spine_study(
    payload: dict[str, Any] = Body(...),
    tm: TaskManager = Depends(get_task_manager),
    _local: None = Depends(require_local_client),
) -> dict[str, Any]:
    """Accept and schedule a spine study background job."""
    root = service.study_root()
    try:
        request = service.parse_study_request(payload, root)
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    # Ensure study root and output directory are created on first write
    root.mkdir(parents=True, exist_ok=True)
    request.output_dir.mkdir(parents=True, exist_ok=True)

    job_id = str(uuid.uuid4())
    if not JOB_POOL.try_reserve(job_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"all {MAX_SPINE_STUDY_JOBS} spine study slots are busy; retry later",
        )

    task = {
        "status": TaskStatus.PENDING.value,
        "kind": _SPINE_STUDY_KIND,
        "study_id": request.study_id,
        "created_at": time.time(),
    }
    try:
        tm.admit_active(job_id, task)
    except RuntimeError as saturated:
        JOB_POOL.release(job_id)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(saturated),
        ) from saturated

    JOB_POOL.start(job_id, lambda: _run_study_job(tm, job_id, request, root))
    return {
        "job_id": job_id,
        "status": TaskStatus.PENDING.value,
        "receipt_path": None,
        "error": None,
    }


@router.get("/studies/{job_id}", response_model=SpineStudyJobStatusV1)
def get_spine_study_job(
    job_id: str,
    tm: TaskManager = Depends(get_task_manager),
    _local: None = Depends(require_local_client),
) -> dict[str, Any]:
    """Status and receipt path of a spine study job."""
    task = tm.get(job_id)
    if task is None or task.get("kind") != _SPINE_STUDY_KIND:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="no such job",
        )
    status_val = str(task.get("status", TaskStatus.PENDING.value))
    result = task.get("result") if status_val == TaskStatus.COMPLETED.value else None
    receipt_path = result.get("receipt_path") if isinstance(result, dict) else None
    error = task.get("error") if status_val == TaskStatus.FAILED.value else None
    return {
        "job_id": job_id,
        "status": status_val,
        "receipt_path": receipt_path,
        "error": error,
    }


@router.get("/study")
def get_spine_study(
    receipt: str = Query(..., description="Receipt path relative to the study root"),
    _local: None = Depends(require_local_client),
) -> dict[str, object]:
    """Hash-verified view model of one study receipt."""
    path = _resolve_receipt(receipt)
    try:
        return service.load_study_payload(path)
    except AggregationIntegrityError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"Malformed receipt: {exc}"
        ) from exc
    except OSError as exc:
        logger.warning("Spine receipt unreadable: %s", exc)
        raise HTTPException(status_code=404, detail="Receipt not readable") from exc
