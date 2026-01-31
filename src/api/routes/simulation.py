"""Simulation routes."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException

from ..models.requests import SimulationRequest
from ..models.responses import SimulationResponse
from ..services.simulation_service import SimulationService

router = APIRouter()

_simulation_service: SimulationService | None = None
_active_tasks: dict[str, dict[str, Any]] = {}
_logger: Any = None


def configure(
    simulation_service: SimulationService | None,
    active_tasks: dict[str, dict[str, Any]],
    logger: Any,
) -> None:
    """Configure dependencies for simulation routes."""
    global _simulation_service, _active_tasks, _logger
    _simulation_service = simulation_service
    _active_tasks = active_tasks
    _logger = logger


@router.post("/simulate", response_model=SimulationResponse)
async def run_simulation(request: SimulationRequest) -> SimulationResponse:
    """Run a physics simulation."""
    if not _simulation_service:
        raise HTTPException(
            status_code=500, detail="Simulation service not initialized"
        )

    try:
        result = await _simulation_service.run_simulation(request)
        return result
    except Exception as exc:
        if _logger:
            _logger.error("Simulation error: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Simulation failed: {str(exc)}"
        ) from exc


@router.post("/simulate/async")
async def run_simulation_async(
    request: SimulationRequest, background_tasks: BackgroundTasks
) -> dict[str, str]:
    """Start an asynchronous simulation."""
    if not _simulation_service:
        raise HTTPException(
            status_code=500, detail="Simulation service not initialized"
        )

    task_id = str(uuid.uuid4())

    _active_tasks[task_id] = {
        "status": "started",
        "created_at": datetime.now(UTC),
    }

    background_tasks.add_task(
        _simulation_service.run_simulation_background,
        task_id,
        request,
        _active_tasks,  # type: ignore[arg-type]
    )

    return {"task_id": task_id, "status": "started"}


@router.get("/simulate/status/{task_id}")
async def get_simulation_status(task_id: str) -> dict[str, Any]:
    """Get status of an asynchronous simulation."""
    if task_id not in _active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    return dict(_active_tasks[task_id])
