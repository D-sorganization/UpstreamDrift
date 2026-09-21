"""Analysis routes.

Provides endpoints for biomechanical analysis and counterfactual
(ZTCF/ZVCF/induced-acceleration) analyses (issue #7450).
All dependencies are injected via FastAPI's Depends() mechanism.
No module-level mutable state.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from src.shared.python.core.contracts import precondition

from ..dependencies import (
    get_analysis_service,
    get_logger,
    get_simulation_service,
    get_task_manager,
)
from ..models.requests import AnalysisRequest, CounterfactualRequest
from ..models.responses import AnalysisResponse
from ..utils.datetime_compat import UTC

if TYPE_CHECKING:
    from ..services.analysis_service import AnalysisService
    from ..services.simulation_service import SimulationService

router = APIRouter()


@router.post("/analyze/biomechanics", response_model=AnalysisResponse)
@precondition(
    lambda request, service=None, logger=None: request is not None,
    "Analysis request must not be None",
)
async def analyze_biomechanics(
    request: AnalysisRequest,
    service: AnalysisService = Depends(get_analysis_service),
    logger: Any = Depends(get_logger),
) -> AnalysisResponse:
    """Perform biomechanical analysis on simulation data.

    Args:
        request: Analysis parameters.
        service: Injected analysis service.
        logger: Injected logger.

    Returns:
        Analysis results.

    Raises:
        HTTPException: On analysis failure.
    """
    try:
        result = await service.analyze_biomechanics(request)
        return result
    except (RuntimeError, TypeError, AttributeError) as exc:
        if logger:
            logger.exception("Analysis error")
        raise HTTPException(
            status_code=500, detail=f"Analysis failed: {str(exc)}"
        ) from exc


# ──────────────────────────────────────────────────────────────
#  Counterfactual / induced-acceleration analyses (issue #7450)
# ──────────────────────────────────────────────────────────────


@router.get("/analysis/counterfactual/kinds")
async def get_counterfactual_kinds(
    service: SimulationService = Depends(get_simulation_service),
) -> dict[str, Any]:
    """Report which counterfactual kinds the current session supports.

    Capability gating is data-driven from the active engine's surface
    (``supported_counterfactual_kinds`` in the analysis orchestrator —
    single source), never hardcoded per engine in the frontend.

    Returns:
        ``{"kinds": [...], "engine": str | None, "session_available": bool}``
    """
    result: dict[str, Any] = service.describe_counterfactual_support()
    return result


@router.post("/analysis/counterfactual")
async def run_counterfactual(
    payload: CounterfactualRequest,
    background_tasks: BackgroundTasks,
    service: SimulationService = Depends(get_simulation_service),
    task_manager: Any = Depends(get_task_manager),
) -> dict[str, str]:
    """Start an asynchronous counterfactual analysis (ZTCF/ZVCF/induced).

    Reuses the ``/simulate/async`` task machinery: poll
    ``GET /simulate/status/{task_id}`` until ``status`` is ``completed``
    (serialized ``CounterfactualResult`` under ``result``) or ``failed``.

    Args:
        payload: Kind and options (kind validity enforced by the model).
        background_tasks: FastAPI background task manager.
        service: Injected simulation service (owns the session recorder).
        task_manager: Injected task manager for tracking.

    Returns:
        Task ID and initial status.

    Raises:
        HTTPException: 409 when no completed simulation session exists or
            the session engine does not support the requested kind.
    """
    support = service.describe_counterfactual_support()
    if not support["session_available"]:
        raise HTTPException(
            status_code=409,
            detail=(
                "No completed simulation session; run a simulation before "
                "requesting a counterfactual analysis"
            ),
        )
    if payload.kind not in support["kinds"]:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Engine '{support['engine']}' does not support "
                f"counterfactual kind '{payload.kind}'. "
                f"Supported kinds: {support['kinds']}"
            ),
        )

    task_id = str(uuid.uuid4())
    task_manager.set(
        task_id,
        {
            "status": "started",
            "kind": payload.kind,
            "created_at": datetime.now(UTC),
        },
    )
    background_tasks.add_task(
        service.run_counterfactual_background,
        task_id,
        payload.kind,
        payload.run_post_hoc,
        task_manager,
    )
    return {"task_id": task_id, "status": "started", "kind": payload.kind}
