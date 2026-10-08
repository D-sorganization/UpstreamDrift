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

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query


from src.shared.python.core.contracts import precondition

from ..dependencies import (
    get_analysis_service,
    get_logger,
    get_simulation_service,
    get_task_manager,
)
from ..models.requests import (
    AnalysisRequest,
    CandidateCounterfactualRequest,
    CounterfactualRequest,
)
from ..models.responses import (
    AnalysisResponse,
    GroundReactionResponse,
    ImpactParametersResponse,
)
from ..services.ground_reaction_service import compute_ground_reaction_plot
from ..services.impact_parameters_service import compute_impact_card
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
    run_id: str | None = Query(None, description="Optional run ID"),
    service: SimulationService = Depends(get_simulation_service),
) -> dict[str, Any]:
    """Report which counterfactual kinds the current session supports.

    Capability gating is data-driven from the active engine's surface
    (``supported_counterfactual_kinds`` in the analysis orchestrator —
    single source), never hardcoded per engine in the frontend.

    Returns:
        ``{"kinds": [...], "engine": str | None, "session_available": bool, "run_id": str | None}``
    """
    try:
        result: dict[str, Any] = (
            service.describe_counterfactual_support(run_id=run_id)
            if run_id
            else service.describe_counterfactual_support()
        )
    except TypeError:
        result = service.describe_counterfactual_support()
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
    try:
        support = (
            service.describe_counterfactual_support(run_id=payload.run_id)
            if payload.run_id
            else service.describe_counterfactual_support()
        )
    except TypeError:
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
            "run_id": payload.run_id,
            "created_at": datetime.now(UTC),
        },
    )
    if payload.run_id:
        try:
            background_tasks.add_task(
                service.run_counterfactual_background,
                task_id,
                payload.kind,
                payload.run_post_hoc,
                task_manager,
                payload.run_id,
            )
        except TypeError:
            background_tasks.add_task(
                service.run_counterfactual_background,
                task_id,
                payload.kind,
                payload.run_post_hoc,
                task_manager,
            )
    else:
        background_tasks.add_task(
            service.run_counterfactual_background,
            task_id,
            payload.kind,
            payload.run_post_hoc,
            task_manager,
        )
    return {"task_id": task_id, "status": "started", "kind": payload.kind}


# ──────────────────────────────────────────────────────────────
#  Candidate Session Force & Counterfactual Analyses (MV-06, #10482)
# ──────────────────────────────────────────────────────────────


@router.get("/analysis/candidate/forces")
async def get_candidate_forces(
    service: SimulationService = Depends(get_simulation_service),
) -> dict[str, Any]:
    """Inspect synchronized force/torque, GRF, and CoP telemetry from active candidate session.

    Fails closed with 409 Conflict if no candidate session has been loaded.
    """
    if service.active_candidate_session is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "No active candidate session loaded; ingest or load a candidate session "
                "before inspecting force/torque telemetry"
            ),
        )
    return service.get_candidate_forces()


@router.post("/analysis/candidate/counterfactual")
async def run_candidate_counterfactual(
    payload: CandidateCounterfactualRequest,
    service: SimulationService = Depends(get_simulation_service),
) -> dict[str, Any]:
    """Execute counterfactual fork rollout on the active candidate session.

    Fails closed with 409 Conflict if no candidate session has been loaded or if the
    session is not a dynamic rollout supporting force/torque channels.
    """
    session = service.active_candidate_session
    if session is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "No active candidate session loaded; ingest or load a candidate session "
                "before requesting counterfactual analysis"
            ),
        )
    if not session.supports_counterfactuals:
        raise HTTPException(
            status_code=409,
            detail=(
                "Active candidate session does not support counterfactual rollouts "
                "(requires accepted dynamic candidate with force channels)"
            ),
        )

    try:
        result = service.run_candidate_counterfactual(
            fork_frame_idx=payload.fork_frame_idx,
            strategy=payload.strategy,
            duration_frames=payload.duration_frames,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ──────────────────────────────────────────────────────────────
#  Impact Parameters (GCV-17, #11723)
# ──────────────────────────────────────────────────────────────


@router.get("/analysis/impact-parameters", response_model=ImpactParametersResponse)
async def get_impact_parameters(
    run_id: str | None = Query(None, description="Run id; defaults to the active run"),
    target_dir: str | None = Query(
        None, description="Target direction 'x,y[,z]' (horizontal); default -Y"
    ),
    handedness: str = Query("right", pattern="^(right|left)$"),
    units: str = Query("mph", pattern="^(mph|m/s)$"),
    impact_index: int | None = Query(None, ge=0),
    service: SimulationService = Depends(get_simulation_service),
) -> ImpactParametersResponse:
    """Launch-monitor-style impact parameters relative to a target line.

    Unavailable quantities are returned as ``null`` with a reason, never zero.
    Responds 404 for an unknown run and 400 for malformed inputs.
    """
    run = service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="No such simulation run")
    try:
        card = compute_impact_card(
            run,
            target_dir=target_dir,
            handedness=handedness,
            units=units,
            impact_index=impact_index,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    payload = card.to_dict()
    time_s = payload["impact_time_s"]
    payload["impact_time_s"] = time_s if time_s == time_s else None
    return ImpactParametersResponse(
        run_id=run.run_id, engine=run.engine_type, **payload
    )


# ──────────────────────────────────────────────────────────────
#  Ground-reaction plots (GCV-5, #11711)
# ──────────────────────────────────────────────────────────────


@router.get("/analysis/ground-reaction", response_model=GroundReactionResponse)
async def get_ground_reaction(
    run_id: str | None = Query(None, description="Run id; defaults to the active run"),
    impact_time_s: float | None = Query(None, description="Impact event marker (s)"),
    service: SimulationService = Depends(get_simulation_service),
) -> GroundReactionResponse:
    """Per-foot and net ground reaction on the body (world frame) over time.

    Force (N and body weights), centre of pressure, free moment, moment about
    the centre of mass and vertical load share.  Unavailable samples are
    ``null``, never zero.  404 for an unknown run, 400 for malformed data.
    """
    run = service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="No such simulation run")
    try:
        payload = compute_ground_reaction_plot(run, impact_time_s=impact_time_s)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return GroundReactionResponse(run_id=run.run_id, engine=run.engine_type, **payload)
