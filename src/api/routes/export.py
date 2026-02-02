"""Export routes.

Provides endpoints for exporting analysis results.
Uses FastAPI's Depends() for dependency injection.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from src.api.config import VALID_EXPORT_FORMATS

from ..dependencies import get_task_manager

router = APIRouter()

# Legacy global for backward compatibility during migration
_active_tasks: dict[str, dict[str, Any]] = {}


def configure(active_tasks: dict[str, dict[str, Any]]) -> None:
    """Configure dependencies for export routes (legacy).

    Note: This function is deprecated. New code should use Depends() instead.
    """
    global _active_tasks
    _active_tasks = active_tasks


@router.get("/export/{task_id}")
async def export_results(
    task_id: str,
    format: str = "json",
    task_manager: Any = Depends(get_task_manager),
) -> JSONResponse:
    """Export analysis results in specified format.

    Args:
        task_id: The task identifier.
        format: Export format (default: json).
        task_manager: Injected task manager.

    Returns:
        Exported results as JSON response.

    Raises:
        HTTPException: If format invalid, task not found, or task incomplete.
    """
    if format not in VALID_EXPORT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format '{format}'. "
            f"Must be one of: {', '.join(sorted(VALID_EXPORT_FORMATS))}",
        )

    if task_id not in task_manager:
        raise HTTPException(status_code=404, detail="Task not found")

    task = task_manager[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed")

    return JSONResponse(content=task["result"])
