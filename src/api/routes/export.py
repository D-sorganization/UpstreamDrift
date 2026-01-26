"""Export routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from src.api.config import VALID_EXPORT_FORMATS

router = APIRouter()

_active_tasks: dict[str, dict[str, Any]] = {}


def configure(active_tasks: dict[str, dict[str, Any]]) -> None:
    """Configure dependencies for export routes."""
    global _active_tasks
    _active_tasks = active_tasks


@router.get("/export/{task_id}")
async def export_results(task_id: str, format: str = "json") -> JSONResponse:
    """Export analysis results in specified format."""
    if format not in VALID_EXPORT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format '{format}'. "
            f"Must be one of: {', '.join(sorted(VALID_EXPORT_FORMATS))}",
        )

    if task_id not in _active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = _active_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed")

    return JSONResponse(content=task["result"])
