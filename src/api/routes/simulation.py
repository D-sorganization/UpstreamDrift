"""Simulation routes.

Provides endpoints for running physics simulations synchronously and asynchronously.
Uses FastAPI's Depends() for dependency injection.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from src.api.utils.datetime_compat import UTC

from ..dependencies import get_logger, get_simulation_service, get_task_manager
from ..models.requests import SimulationRequest
from ..models.responses import SimulationResponse

if TYPE_CHECKING:
    from ..services.simulation_service import SimulationService

router = APIRouter()

# Legacy globals for backward compatibility during migration
_simulation_service: SimulationService | None = None
_active_tasks: dict[str, dict[str, Any]] = {}
_logger: Any = None


def configure(
    simulation_service: SimulationService | None,
    active_tasks: dict[str, dict[str, Any]],
    logger: Any,
) -> None:
    """Configure dependencies for simulation routes (legacy).

    Note: This function is deprecated. New code should use Depends() instead.
    """
    global _simulation_service, _active_tasks, _logger
    _simulation_service = simulation_service
    _active_tasks = active_tasks
    _logger = logger


@router.post("/simulate", response_model=SimulationResponse)
async def run_simulation(
    request: SimulationRequest,
    service: SimulationService = Depends(get_simulation_service),
    logger: Any = Depends(get_logger),
) -> SimulationResponse:
    """Run a physics simulation.

    Args:
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
    except Exception as exc:
        if logger:
            logger.exception("Unexpected simulation error: %s", exc)
        raise HTTPException(
            status_code=500, detail="Internal simulation error"
        ) from exc


@router.post("/simulate/async")
async def run_simulation_async(
    request: SimulationRequest,
    background_tasks: BackgroundTasks,
    service: SimulationService = Depends(get_simulation_service),
    task_manager: Any = Depends(get_task_manager),
) -> dict[str, str]:
    """Start an asynchronous simulation.

    Args:
        request: Simulation parameters.
        background_tasks: FastAPI background task manager.
        service: Injected simulation service.
        task_manager: Injected task manager for tracking.

    Returns:
        Task ID and initial status.
    """
    task_id = str(uuid.uuid4())

    task_manager[task_id] = {
        "status": "started",
        "created_at": datetime.now(UTC),
    }

    background_tasks.add_task(
        service.run_simulation_background,
        task_id,
        request,
        task_manager,
    )

    return {"task_id": task_id, "status": "started"}


@router.get("/simulate/status/{task_id}")
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
    if task_id not in task_manager:
        raise HTTPException(status_code=404, detail="Task not found")

    return dict(task_manager[task_id])
