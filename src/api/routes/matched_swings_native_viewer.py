"""Native-viewer handoff routes for matched-swing runs (issue #11987).

Sibling to ``src/api/routes/matched_swings.py`` (same router prefix and tag,
same local-only guard and error shape): the web Matched Swing Browser page
has no equivalent of the desktop's "Open Native Viewer" button
(``gui.py::_on_open_native_viewer``), which lets a user pick a backend
(MuJoCo/MeshCat/Gepetto/OpenSim/MATLAB) and launch the selected run's NPZ
trajectory in it. These routes expose that handoff to the local API host,
sharing the Qt-free ``src/tools/matched_swing_browser/native_handoff.py``
helper with the desktop button so both paths load the NPZ and launch the
viewer through one implementation.

Routes
------
- ``GET /matched-swings/{run_id}/native-viewers`` — backend availability list
- ``POST /matched-swings/{run_id}/native-viewer`` — launch a backend
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.routes.matched_swings import (
    _raise_job_error,
    get_matched_swings_service,
    require_local_client,
)
from src.api.services.matched_swings_service import (
    MatchedSwingJobError,
    MatchedSwingsService,
)
from src.shared.python.motion_matching.native_viewers import (
    ViewerUnavailableError,
    get_backend_adapter,
)
from src.tools.matched_swing_browser.native_handoff import (
    launch_native_viewer,
    list_native_viewer_backends,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/matched-swings", tags=["matched-swings"])


class NativeViewerLaunchRequest(BaseModel):
    """Request body for launching a native viewer backend on a run's NPZ."""

    backend: str
    speed: float = Field(default=1.0, gt=0, le=4)


def _resolve_trajectory_path(service: MatchedSwingsService, run_id: str) -> Path:
    """Resolve the NPZ trajectory path for ``run_id``, raising HTTPException.

    Raises:
        HTTPException: 404 when ``run_id`` is unknown.
    """
    try:
        return service.resolve_artifact_path(run_id, "candidate")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{run_id}/native-viewers")
async def list_matched_swing_native_viewers(
    run_id: str,
    _local: None = Depends(require_local_client),
    service: MatchedSwingsService = Depends(get_matched_swings_service),
) -> dict[str, Any]:
    """Return native-viewer backend availability for a matched-swing run."""
    try:
        _resolve_trajectory_path(service, run_id)
        has_trajectory = True
    except FileNotFoundError:
        has_trajectory = False

    return {
        "schema_version": "matched-swing-native-viewer/1",
        "run_id": run_id,
        "has_trajectory": has_trajectory,
        "backends": list_native_viewer_backends(),
    }


@router.post("/{run_id}/native-viewer")
def launch_matched_swing_native_viewer(
    run_id: str,
    payload: NativeViewerLaunchRequest,
    _local: None = Depends(require_local_client),
    service: MatchedSwingsService = Depends(get_matched_swings_service),
) -> dict[str, Any]:
    """Launch a native viewer backend on a matched-swing run's NPZ trajectory.

    A plain (synchronous) route so FastAPI runs the blocking viewer launch in
    its threadpool instead of the event loop.
    """
    try:
        npz_path = _resolve_trajectory_path(service, run_id)
    except FileNotFoundError as exc:
        _raise_job_error(
            MatchedSwingJobError(code="trajectory_unavailable", message=str(exc)),
            status_code=404,
        )

    try:
        get_backend_adapter(payload.backend)
    except ValueError as exc:
        _raise_job_error(
            MatchedSwingJobError(code="unsupported_backend", message=str(exc)),
            status_code=422,
        )

    try:
        result = launch_native_viewer(npz_path, payload.backend, speed=payload.speed)
    except ViewerUnavailableError as exc:
        _raise_job_error(
            MatchedSwingJobError(code="viewer_unavailable", message=str(exc)),
            status_code=503,
        )
    except ValueError as exc:
        _raise_job_error(
            MatchedSwingJobError(code="trajectory_invalid", message=str(exc)),
            status_code=422,
        )
    except FileNotFoundError as exc:
        # Rare race: the NPZ existed when resolved above but is gone by launch.
        _raise_job_error(
            MatchedSwingJobError(code="trajectory_unavailable", message=str(exc)),
            status_code=404,
        )
    except (RuntimeError, OSError):
        # The backend failed after passing its availability check. The
        # exception text can name server paths, so it stays in the log.
        logger.exception("Native viewer %s failed for run %s", payload.backend, run_id)
        _raise_job_error(
            MatchedSwingJobError(
                code="viewer_launch_failed",
                message=(
                    f"Native viewer '{payload.backend}' failed to launch; "
                    "see the API server log."
                ),
            ),
            status_code=502,
        )

    return {
        "schema_version": "matched-swing-native-viewer/1",
        "run_id": run_id,
        "success": result.success,
        "backend": result.backend,
        "url": result.url,
        "message": result.message,
    }
