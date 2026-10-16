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

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.routes.matched_swings import require_local_client
from src.api.services import spine_viewer_service as service
from src.api.utils.path_validation import resolve_contained_path
from src.engines.physics_engines.opensim.python.spine.study_aggregation import (
    AggregationIntegrityError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools/spine-viewer", tags=["spine-viewer"])


def _resolve_receipt(receipt: str) -> Path:
    """Map a client path to a file inside the study root (400/404 otherwise)."""
    try:
        relative = service.safe_relative_receipt(receipt)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    root = service.study_root()
    return resolve_contained_path(root.joinpath(*relative.parts), [root])


@router.get("/receipts")
def list_spine_receipts(
    _local: None = Depends(require_local_client),
) -> dict[str, object]:
    """Relative paths of the receipts available under the study root."""
    return {"receipts": service.list_receipts(service.study_root())}


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
