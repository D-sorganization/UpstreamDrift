"""Matched-swing results API routes (MS-85, #10358).

Read-only, local-only endpoints exposing the matched-swing run ledger,
individual receipts, candidate NPZ packages, parity reports, and GIF
animations for the web/Tauri Results page.

Routes
------
- ``GET /matched-swings`` — ledger index
- ``GET /matched-swings/{id}`` — receipt JSON
- ``GET /matched-swings/{id}/candidate`` — NPZ stream or preview JSON
- ``GET /matched-swings/{id}/parity`` — parity report JSON
- ``GET /matched-swings/{id}/animation.gif`` — GIF stream
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse

from src.api.services.matched_swings_service import (
    MatchedSwingJobError,
    MatchedSwingsService,
)
from src.shared.python.contracts import precondition

router = APIRouter(prefix="/matched-swings", tags=["matched-swings"])

_LOCAL_CLIENT_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})


def _is_local_request_client(request: Request) -> bool:
    """Return True when the request originates from the server's own machine."""
    client = request.client
    if client is None:
        return True
    return client.host in _LOCAL_CLIENT_HOSTS


@precondition(lambda request: request is not None)
def require_local_client(request: Request) -> None:
    """Fail closed when a remote client attempts to read local evidence."""
    if not _is_local_request_client(request):
        raise HTTPException(
            status_code=403,
            detail=(
                "Matched-swing evidence routes are local-only. "
                "Connect from the machine running the API server."
            ),
        )


@lru_cache(maxsize=1)
def _default_service() -> MatchedSwingsService:
    return MatchedSwingsService()


def get_matched_swings_service() -> MatchedSwingsService:
    """FastAPI dependency returning the process-level matched-swings service."""
    return _default_service()


def _raise_job_error(error: MatchedSwingJobError, status_code: int) -> None:
    raise HTTPException(
        status_code=status_code,
        detail={"message": error.message, "error": error.to_dict()},
    )


@router.get("")
async def list_matched_swings(
    _local: None = Depends(require_local_client),
    service: MatchedSwingsService = Depends(get_matched_swings_service),
) -> dict[str, Any]:
    """Return the matched-swing ledger as public run summaries."""
    runs = service.list_runs()
    return {
        "schema_version": "matched-swing-api/1",
        "total": len(runs),
        "runs": [run.to_dict() for run in runs],
    }


@router.get("/{run_id}")
async def get_matched_swing_receipt(
    run_id: str,
    _local: None = Depends(require_local_client),
    service: MatchedSwingsService = Depends(get_matched_swings_service),
) -> dict[str, Any]:
    """Return the receipt JSON for a single matched-swing run."""
    try:
        summary = service.get_run_summary(run_id)
        receipt = service.get_receipt(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (FileNotFoundError, ValueError) as exc:
        _raise_job_error(
            MatchedSwingJobError(code="receipt_unavailable", message=str(exc)),
            status_code=404,
        )
    return {
        "id": summary.id,
        "receipt": receipt,
        "summary": summary.to_dict(),
        "candidate_sha256": summary.candidate_sha256,
        "capabilities": summary.capabilities.to_dict(),
    }


@router.get("/{run_id}/candidate", response_model=None)
async def get_matched_swing_candidate(
    run_id: str,
    preview_frame: int | None = Query(
        default=None,
        ge=0,
        description="When set, return JSON marker joints for MocapSkeleton3D preview.",
    ),
    _local: None = Depends(require_local_client),
    service: MatchedSwingsService = Depends(get_matched_swings_service),
) -> FileResponse | dict[str, Any]:
    """Stream the candidate NPZ or return a JSON preview frame for 3D replay."""
    if preview_frame is not None:
        try:
            joints = service.candidate_preview_joints(run_id, preview_frame)
            frame_count = service.candidate_frame_count(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (FileNotFoundError, ValueError, IndexError) as exc:
            _raise_job_error(
                MatchedSwingJobError(code="candidate_unavailable", message=str(exc)),
                status_code=404,
            )
        return {
            "id": run_id,
            "frame_index": preview_frame,
            "frame_count": frame_count,
            "joints": joints,
        }

    try:
        path = service.resolve_artifact_path(run_id, "candidate")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        _raise_job_error(
            MatchedSwingJobError(code="candidate_unavailable", message=str(exc)),
            status_code=404,
        )
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=f"{run_id[:12]}_candidate.npz",
    )


@router.get("/{run_id}/parity")
async def get_matched_swing_parity(
    run_id: str,
    _local: None = Depends(require_local_client),
    service: MatchedSwingsService = Depends(get_matched_swings_service),
) -> dict[str, Any]:
    """Return the cross-engine parity report JSON for a run."""
    try:
        path = service.resolve_artifact_path(run_id, "parity")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        _raise_job_error(
            MatchedSwingJobError(code="parity_unavailable", message=str(exc)),
            status_code=404,
        )
    import json

    data: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise HTTPException(
            status_code=500, detail="Parity report is not a JSON object"
        )
    return dict(data)


@router.get("/{run_id}/animation.gif")
async def get_matched_swing_animation(
    run_id: str,
    _local: None = Depends(require_local_client),
    service: MatchedSwingsService = Depends(get_matched_swings_service),
) -> FileResponse:
    """Stream the kinematic GIF animation for a run."""
    try:
        path = service.resolve_artifact_path(run_id, "gif")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type="image/gif", filename=f"{run_id[:12]}.gif")
