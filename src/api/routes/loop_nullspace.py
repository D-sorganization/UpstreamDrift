"""FastAPI routes for Closed-Loop Null-Space Exploration (NSX-40, #12572).

Provides:
- ``POST /analysis/loop-nullspace/jobs``: creates an async/cancellable NSX job
  and returns ``job_id`` and initial status.
- ``GET /analysis/loop-nullspace/jobs/{job_id}``: returns job progress, partial
  invariants first, or the completed report.
- ``DELETE /analysis/loop-nullspace/jobs/{job_id}``: cancels the job and sets
  its status to ``"cancelled"``.
"""

from __future__ import annotations

from typing import Any, Literal, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from src.api.services.loop_nullspace_service import LoopNullspaceService

__all__ = [
    "LoopNullspaceJobRequest",
    "LoopNullspaceJobStatus",
    "get_loop_nullspace_service",
    "router",
]

router = APIRouter(tags=["loop-nullspace"])


class LoopNullspaceJobRequest(BaseModel):
    """Request payload for creating an NSX analysis job."""

    trajectory: dict[str, Any] | None = Field(
        default=None,
        description="Optional closed-loop trajectory payload; defaults to canonical fixture.",
    )
    engine: str = Field(
        default="pinocchio",
        min_length=1,
        description="Physics engine identifier.",
    )
    preset: str = Field(
        default="capacity_min_effort",
        min_length=1,
        description="Registry hypothesis preset name.",
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Analysis options (e.g. registry_version, include_restriction_sensitivity).",
    )


class LoopNullspaceJobStatus(BaseModel):
    """Status response for an NSX analysis job."""

    job_id: str
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    progress_percent: float = Field(ge=0.0, le=100.0)
    compute_cache_key: str
    cache_hit: bool
    partial_invariants: dict[str, Any] | None = None
    report: dict[str, Any] | None = None
    error: str | None = None


def get_loop_nullspace_service(request: Request) -> LoopNullspaceService:
    """Resolve the :class:`LoopNullspaceService` attached to ``request.app.state``."""
    app_state = request.app.state
    service = getattr(app_state, "loop_nullspace_service", None)
    if service is None:
        service = LoopNullspaceService()
        app_state.loop_nullspace_service = service
    return cast(LoopNullspaceService, service)


@router.post(
    "/analysis/loop-nullspace/jobs",
    response_model=LoopNullspaceJobStatus,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_loop_nullspace_job(
    payload: LoopNullspaceJobRequest,
    service: LoopNullspaceService = Depends(get_loop_nullspace_service),
) -> dict[str, Any]:
    """Create an async/cancellable NSX job and return its initial status."""
    try:
        job_id = service.create_job(payload)
        return service.get_job(job_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc


@router.get(
    "/analysis/loop-nullspace/jobs/{job_id}",
    response_model=LoopNullspaceJobStatus,
)
def get_loop_nullspace_job(
    job_id: str,
    service: LoopNullspaceService = Depends(get_loop_nullspace_service),
) -> dict[str, Any]:
    """Poll an NSX job for progress, partial invariants, or completed report."""
    try:
        return service.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"No such NSX job: {job_id}"
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc


@router.delete(
    "/analysis/loop-nullspace/jobs/{job_id}",
    response_model=LoopNullspaceJobStatus,
)
def cancel_loop_nullspace_job(
    job_id: str,
    service: LoopNullspaceService = Depends(get_loop_nullspace_service),
) -> dict[str, Any]:
    """Cancel an NSX job and set its status to ``'cancelled'``."""
    try:
        return service.cancel_job(job_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"No such NSX job: {job_id}"
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
