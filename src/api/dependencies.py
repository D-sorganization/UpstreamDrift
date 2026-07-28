"""FastAPI dependency injection providers.

This module provides dependency functions for FastAPI's Depends() mechanism,
enabling proper separation of concerns and testability.

All services are stored in app.state during startup and retrieved via
these dependency functions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from fastapi import HTTPException, Request, WebSocket

if TYPE_CHECKING:
    from src.shared.python.engine_core.engine_manager import EngineManager
    from src.shared.python.gui_pkg.video_pose_pipeline import VideoPosePipeline

    from .services.analysis_service import AnalysisService
    from .services.simulation_service import SimulationService
    from .task_manager import TaskManager

VideoPipelineFactory = Callable[[str, float, bool], Any]


def _load_video_pipeline_classes() -> tuple[type[Any], type[Any]]:
    """Lazily import the video pose pipeline classes.

    The video stack transitively imports optional packages such as ``cv2`` and
    ``mediapipe``. Keeping the import behind this dependency boundary preserves
    route discovery in slim runtime images while still returning an actionable
    503 when a request needs unavailable video extras.

    Returns:
        ``(VideoPosePipeline, VideoProcessingConfig)`` classes.

    Raises:
        HTTPException: 503 when optional video dependencies are missing.
    """
    try:
        from src.shared.python.gui_pkg.video_pose_pipeline import (
            VideoPosePipeline,
            VideoProcessingConfig,
        )
    except ImportError as exc:  # pragma: no cover - exercised only in slim image
        raise HTTPException(
            status_code=503,
            detail=(
                "Video analysis is unavailable in this runtime image: optional "
                "dependency missing ("
                f"{exc.name or exc}"
                "). Install the 'video' extras (opencv-python, mediapipe) or "
                "use the video-enabled runtime image."
            ),
        ) from exc
    return VideoPosePipeline, VideoProcessingConfig


def make_video_pipeline(
    estimator_type: str,
    min_confidence: float,
    enable_smoothing: bool = True,
) -> VideoPosePipeline:
    """Create a video pipeline with request-scoped estimator settings.

    Returns:
        VideoPosePipeline configured for the provided estimator parameters.

    Raises:
        HTTPException: 503 when optional video dependencies are missing.
    """
    video_pipeline_cls, video_config_cls = _load_video_pipeline_classes()
    config = video_config_cls(
        estimator_type=estimator_type,
        min_confidence=min_confidence,
        enable_temporal_smoothing=enable_smoothing,
    )
    return cast("VideoPosePipeline", video_pipeline_cls(config))


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
    return cast("EngineManager", manager)


def get_ws_engine_manager(websocket: WebSocket) -> EngineManager | None:
    """Retrieve the EngineManager from app state for WebSocket routes.

    WebSocket handlers cannot surface FastAPI ``HTTPException`` responses after
    the upgrade path starts, so this accessor returns ``None`` when app state is
    missing and lets the route send an explicit WebSocket error frame.

    Args:
        websocket: FastAPI WebSocket object.

    Returns:
        EngineManager instance, or ``None`` when not initialized.
    """
    app_state = getattr(getattr(websocket, "app", None), "state", None)
    manager = getattr(app_state, "engine_manager", None)
    if manager is None:
        return None
    return cast("EngineManager", manager)


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
    return cast("SimulationService", service)


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
    return cast("AnalysisService", service)


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
    return cast("VideoPosePipeline", pipeline)


def get_video_pipeline_factory(request: Request) -> VideoPipelineFactory:
    """Retrieve the configured video pipeline factory from app state.

    Args:
        request: FastAPI request object.

    Returns:
        Callable that creates a VideoPosePipeline for request-scoped settings.
    """
    factory = getattr(request.app.state, "video_pipeline_factory", None)
    if factory is None:
        return make_video_pipeline
    return cast(VideoPipelineFactory, factory)


def get_task_manager(request: Request) -> TaskManager:
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
    return cast("TaskManager", manager)


def get_logger(request: Request) -> Any:
    """Retrieve the logger from app state.

    Args:
        request: FastAPI request object.

    Returns:
        Logger instance.
    """
    return getattr(request.app.state, "logger", None)
