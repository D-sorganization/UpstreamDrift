"""Simulation routes.

Provides endpoints for running physics simulations synchronously and asynchronously.
All dependencies are injected via FastAPI's Depends() mechanism.
No module-level mutable state.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

from src.api.utils.datetime_compat import UTC
from src.shared.python.core.contracts import precondition

from ..dependencies import get_logger, get_simulation_service, get_task_manager
from ..models.requests import SimulationRequest
from ..models.responses import SimulationResponse
from ..rate_limit import get_limit, limiter

if TYPE_CHECKING:
    from ..services.simulation_service import SimulationService

router = APIRouter()


@router.post("/simulate", response_model=SimulationResponse)
@limiter.limit(get_limit("API_LIMIT_SIMULATE", "5/minute"))
@precondition(
    lambda http_request, request, service=None, logger=None: request is not None,
    "Simulation request must not be None",
)
async def run_simulation(
    http_request: Request,
    request: SimulationRequest,
    service: SimulationService = Depends(get_simulation_service),
    logger: Any = Depends(get_logger),
) -> SimulationResponse:
    """Run a physics simulation.

    Args:
        http_request: FastAPI request object (used by the rate limiter).
        request: Simulation parameters.
        service: Injected simulation service.
        logger: Injected logger.

    Returns:
        Simulation results.

    Raises:
        HTTPException: On simulation failure.
    """
    try:
        result = await service.run_simulation(request)
        return result
    except TimeoutError as exc:
        if logger:
            logger.warning("Simulation timeout: %s", exc)
        raise HTTPException(status_code=504, detail="Simulation timed out") from exc
    except ValueError as exc:
        if logger:
            logger.warning("Invalid simulation parameters: %s", exc)
        raise HTTPException(
            status_code=400, detail=f"Invalid parameters: {str(exc)}"
        ) from exc
    except RuntimeError as exc:
        if logger:
            logger.error("Simulation runtime error: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Simulation failed: {str(exc)}"
        ) from exc
    except ImportError as exc:
        if logger:
            logger.exception("Unexpected simulation error: %s", exc)
        raise HTTPException(
            status_code=500, detail="Internal simulation error"
        ) from exc


@router.post("/simulate/async")
@limiter.limit(get_limit("API_LIMIT_SIMULATE_ASYNC", "10/minute"))
async def run_simulation_async(
    http_request: Request,
    request: SimulationRequest,
    background_tasks: BackgroundTasks,
    service: SimulationService = Depends(get_simulation_service),
    task_manager: Any = Depends(get_task_manager),
) -> dict[str, str]:
    """Start an asynchronous simulation.

    Args:
        http_request: FastAPI request object (used by the rate limiter).
        request: Simulation parameters.
        background_tasks: FastAPI background task manager.
        service: Injected simulation service.
        task_manager: Injected task manager for tracking.

    Returns:
        Task ID and initial status.
    """
    if not (request is not None):
        raise ValueError("request must be provided")
    task_id = str(uuid.uuid4())

    await task_manager.set(
        task_id,
        {
            "status": "started",
            "created_at": datetime.now(UTC),
        },
    )

    background_tasks.add_task(
        service.run_simulation_background,
        task_id,
        request,
        task_manager,
    )

    return {"task_id": task_id, "status": "started"}


@router.get("/simulate/status/{task_id}")
@precondition(
    lambda task_id, task_manager=None: task_id is not None and len(task_id.strip()) > 0,
    "Task ID must be a non-empty string",
)
async def get_simulation_status(
    task_id: str,
    task_manager: Any = Depends(get_task_manager),
) -> dict[str, Any]:
    """Get status of an asynchronous simulation.

    Args:
        task_id: The task identifier.
        task_manager: Injected task manager.

    Returns:
        Current task status and data.

    Raises:
        HTTPException: If task not found.
    """
    if not await task_manager.exists(task_id):
        raise HTTPException(status_code=404, detail="Task not found")

    task_data = await task_manager.get(task_id)
    return dict(task_data) if task_data else {}
