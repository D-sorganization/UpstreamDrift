"""FastAPI dependency injection providers.

This module provides dependency functions for FastAPI's Depends() mechanism,
enabling proper separation of concerns and testability.

All services are stored in app.state during startup and retrieved via
these dependency functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, Request

if TYPE_CHECKING:
    from src.shared.python.engine_manager import EngineManager
    from src.shared.python.gui_pkg.video_pose_pipeline import VideoPosePipeline

    from .services.analysis_service import AnalysisService
    from .services.simulation_service import SimulationService


def get_engine_manager(request: Request) -> EngineManager:
    """Retrieve the EngineManager from app state.

    Args:
        request: FastAPI request object.

    Returns:
        EngineManager instance.

    Raises:
        HTTPException: If engine manager not initialized.
    """
    manager = getattr(request.app.state, "engine_manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="Engine manager not initialized")
    return manager


def get_simulation_service(request: Request) -> SimulationService:
    """Retrieve the SimulationService from app state.

    Args:
        request: FastAPI request object.

    Returns:
        SimulationService instance.

    Raises:
        HTTPException: If simulation service not initialized.
    """
    service = getattr(request.app.state, "simulation_service", None)
    if service is None:
        raise HTTPException(
            status_code=503, detail="Simulation service not initialized"
        )
    return service


def get_analysis_service(request: Request) -> AnalysisService:
    """Retrieve the AnalysisService from app state.

    Args:
        request: FastAPI request object.

    Returns:
        AnalysisService instance.

    Raises:
        HTTPException: If analysis service not initialized.
    """
    service = getattr(request.app.state, "analysis_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Analysis service not initialized")
    return service


def get_video_pipeline(request: Request) -> VideoPosePipeline:
    """Retrieve the VideoPosePipeline from app state.

    Args:
        request: FastAPI request object.

    Returns:
        VideoPosePipeline instance.

    Raises:
        HTTPException: If video pipeline not initialized.
    """
    pipeline = getattr(request.app.state, "video_pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Video pipeline not initialized (MediaPipe may not be installed)",
        )
    return pipeline


def get_task_manager(request: Request) -> Any:
    """Retrieve the TaskManager from app state.

    Args:
        request: FastAPI request object.

    Returns:
        TaskManager instance.

    Raises:
        HTTPException: If task manager not initialized.
    """
    manager = getattr(request.app.state, "task_manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="Task manager not initialized")
    return manager


def get_logger(request: Request) -> Any:
    """Retrieve the logger from app state.

    Args:
        request: FastAPI request object.

    Returns:
        Logger instance.
    """
    return getattr(request.app.state, "logger", None)
