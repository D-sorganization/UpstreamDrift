"""Video analysis routes."""

from __future__ import annotations

import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from src.api.config import (
    MAX_CONFIDENCE,
    MAX_POSE_DATA_ENTRIES,
    MIN_CONFIDENCE,
    VALID_ESTIMATOR_TYPES,
)
from src.shared.python.video_pose_pipeline import (
    VideoPosePipeline,
    VideoProcessingConfig,
)

from ..models.responses import VideoAnalysisResponse

router = APIRouter()

_video_pipeline: VideoPosePipeline | None = None
_active_tasks: dict[str, dict[str, Any]] = {}
_logger: Any = None


def configure(
    video_pipeline: VideoPosePipeline | None,
    active_tasks: dict[str, dict[str, Any]],
    logger: Any,
) -> None:
    """Configure route dependencies from the server startup."""
    global _video_pipeline, _active_tasks, _logger
    _video_pipeline = video_pipeline
    _active_tasks = active_tasks
    _logger = logger


@router.post("/analyze/video", response_model=VideoAnalysisResponse)
async def analyze_video(
    file: UploadFile = File(...),
    estimator_type: str = "mediapipe",
    min_confidence: float = 0.5,
    enable_smoothing: bool = True,
) -> VideoAnalysisResponse:
    """Analyze golf swing from uploaded video."""
    if not _video_pipeline:
        raise HTTPException(status_code=500, detail="Video pipeline not initialized")

    if estimator_type not in VALID_ESTIMATOR_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid estimator_type '{estimator_type}'. "
            f"Must be one of: {', '.join(sorted(VALID_ESTIMATOR_TYPES))}",
        )

    if not (MIN_CONFIDENCE <= min_confidence <= MAX_CONFIDENCE):
        raise HTTPException(
            status_code=400,
            detail=f"min_confidence must be between {MIN_CONFIDENCE} and {MAX_CONFIDENCE}",
        )

    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        config = VideoProcessingConfig(
            estimator_type=estimator_type,
            min_confidence=min_confidence,
            enable_temporal_smoothing=enable_smoothing,
        )
        pipeline = VideoPosePipeline(config)
        result = pipeline.process_video(temp_path)

        response = VideoAnalysisResponse(
            filename=file.filename or "unknown",
            total_frames=result.total_frames,
            valid_frames=result.valid_frames,
            average_confidence=result.average_confidence,
            quality_metrics=result.quality_metrics,
            pose_data=[
                {
                    "timestamp": pose.timestamp,
                    "confidence": pose.confidence,
                    "joint_angles": pose.joint_angles,
                    "keypoints": pose.raw_keypoints or {},
                }
                for pose in result.pose_results[:MAX_POSE_DATA_ENTRIES]
            ],
        )

        return response

    except Exception as e:
        if _logger:
            _logger.error("Video analysis error: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Video analysis failed: {str(e)}"
        ) from e
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError as cleanup_error:
                if _logger:
                    _logger.warning(
                        "Failed to clean up temp file %s: %s",
                        temp_path,
                        cleanup_error,
                    )


@router.post("/analyze/video/async")
async def analyze_video_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    estimator_type: str = "mediapipe",
    min_confidence: float = 0.5,
) -> dict[str, str]:
    """Start asynchronous video analysis."""
    if not _video_pipeline:
        raise HTTPException(status_code=500, detail="Video pipeline not initialized")

    if estimator_type not in VALID_ESTIMATOR_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid estimator_type '{estimator_type}'. "
            f"Must be one of: {', '.join(sorted(VALID_ESTIMATOR_TYPES))}",
        )

    if not (MIN_CONFIDENCE <= min_confidence <= MAX_CONFIDENCE):
        raise HTTPException(
            status_code=400,
            detail=f"min_confidence must be between {MIN_CONFIDENCE} and {MAX_CONFIDENCE}",
        )

    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    task_id = str(uuid.uuid4())

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = Path(temp_file.name)

    _active_tasks[task_id] = {
        "status": "started",
        "created_at": datetime.now(UTC),
    }

    background_tasks.add_task(
        _process_video_background,
        task_id,
        temp_path,
        file.filename or "unknown",
        estimator_type,
        min_confidence,
    )

    return {"task_id": task_id, "status": "started"}


async def _process_video_background(
    task_id: str,
    video_path: Path,
    filename: str,
    estimator_type: str,
    min_confidence: float,
) -> None:
    """Background task for video processing."""
    try:
        task_data = _active_tasks.get(task_id) or {}
        created_at = task_data.get("created_at", datetime.now(UTC))

        _active_tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "created_at": created_at,
        }

        config = VideoProcessingConfig(
            estimator_type=estimator_type, min_confidence=min_confidence
        )
        pipeline = VideoPosePipeline(config)

        result = pipeline.process_video(video_path)

        task_data = _active_tasks.get(task_id) or {}
        created_at = task_data.get("created_at", datetime.now(UTC))

        _active_tasks[task_id] = {
            "status": "completed",
            "created_at": created_at,
            "result": {
                "filename": filename,
                "total_frames": result.total_frames,
                "valid_frames": result.valid_frames,
                "average_confidence": result.average_confidence,
                "quality_metrics": result.quality_metrics,
            },
        }

    except Exception as e:
        task_data = _active_tasks.get(task_id) or {}
        created_at = task_data.get("created_at", datetime.now(UTC))

        _active_tasks[task_id] = {
            "status": "failed",
            "error": str(e),
            "created_at": created_at,
        }
    finally:
        if video_path.exists():
            video_path.unlink()
