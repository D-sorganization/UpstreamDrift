"""LIFT-1 cross-engine baseline API routes (LIFT-8, #11748).

Read-only routes over the committed
``docs/development/lifting/pack_parity_baseline.json`` receipt via
``src.shared.python.lifting.baseline_view``. These are the data source the
PyQt and web lift-viewing surfaces (later LIFT-8 slices) will consume.

Routes
------
- ``GET /lifting/baseline`` -- receipt metadata (schema, generated_utc,
  anthropometry, tolerances, packs, lifts, gap count)
- ``GET /lifting/baseline/lifts/{lift}`` -- per-lift view (see
  :func:`src.shared.python.lifting.baseline_view.lift_view`)
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from src.shared.python.lifting.baseline_view import (
    available_lifts,
    lift_view,
    load_baseline,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/lifting", tags=["lifting"])


@lru_cache(maxsize=1)
def _cached_baseline_receipt() -> dict[str, Any]:
    """Process-cached receipt load; a missing/invalid file is never cached."""
    return load_baseline()


def get_lifting_baseline_receipt() -> dict[str, Any]:
    """FastAPI dependency: the LIFT-1 baseline receipt, or a 503."""
    try:
        return _cached_baseline_receipt()
    except (FileNotFoundError, ValueError) as exc:
        # The loader message carries a server path; log it, don't return it.
        logger.warning("lift baseline receipt unavailable: %s", exc)
        raise HTTPException(
            status_code=503, detail="lift baseline receipt unavailable"
        ) from exc


@router.get("/baseline")
async def get_baseline_metadata(
    receipt: dict[str, Any] = Depends(get_lifting_baseline_receipt),
) -> dict[str, Any]:
    """Return LIFT-1 baseline receipt metadata (not the full per-lift data)."""
    return {
        "schema": receipt["schema"],
        "generated_utc": receipt["generated_utc"],
        "anthropometry": receipt["anthropometry"],
        "tolerances": receipt["tolerances"],
        "packs": receipt["packs"],
        "lifts": available_lifts(receipt),
        "gap_count": len(receipt.get("gaps", [])),
    }


@router.get("/baseline/lifts/{lift}")
async def get_baseline_lift(
    lift: str,
    receipt: dict[str, Any] = Depends(get_lifting_baseline_receipt),
) -> dict[str, Any]:
    """Return the per-lift baseline view for *lift*."""
    try:
        return lift_view(receipt, lift)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
