"""Video analysis routes.

All dependencies are injected via FastAPI's Depends() mechanism.
No module-level mutable state.
"""

from __future__ import annotations

import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile

from src.api.config import (
    MAX_CONFIDENCE,
    MAX_POSE_DATA_ENTRIES,
    MIN_CONFIDENCE,
    VALID_ESTIMATOR_TYPES,
)
from src.api.utils.datetime_compat import UTC
from src.shared.python.video_pose_pipeline import (
    VideoPosePipeline,
    VideoProcessingConfig,
)

from ..dependencies import get_logger, get_task_manager, get_video_pipeline
from ..models.responses import VideoAnalysisResponse

router = APIRouter()


@router.post("/analyze/video", response_model=VideoAnalysisResponse)
async def analyze_video(
    file: UploadFile = File(...),
    estimator_type: str = "mediapipe",
    min_confidence: float = 0.5,
    enable_smoothing: bool = True,
    video_pipeline: VideoPosePipeline = Depends(get_video_pipeline),
    logger: Any = Depends(get_logger),
) -> VideoAnalysisResponse:
    """Analyze golf swing from uploaded video.

    Args:
        file: Uploaded video file.
        estimator_type: Pose estimation backend.
        min_confidence: Minimum confidence threshold.
        enable_smoothing: Enable temporal smoothing.
        video_pipeline: Injected video pipeline.
        logger: Injected logger.

    Returns:
        Video analysis results.

    Raises:
        HTTPException: On validation or processing failure.
    """
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

    except HTTPException:
        raise
    except Exception as e:
        if logger:
            logger.error("Video analysis error: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Video analysis failed: {str(e)}"
        ) from e
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError as cleanup_error:
                if logger:
                    logger.warning(
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
    video_pipeline: VideoPosePipeline = Depends(get_video_pipeline),
    task_manager: Any = Depends(get_task_manager),
) -> dict[str, str]:
    """Start asynchronous video analysis.

    Args:
        background_tasks: FastAPI background task manager.
        file: Uploaded video file.
        estimator_type: Pose estimation backend.
        min_confidence: Minimum confidence threshold.
        video_pipeline: Injected video pipeline (validates initialization).
        task_manager: Injected task manager for tracking.

    Returns:
        Task ID and initial status.

    Raises:
        HTTPException: On validation failure.
    """
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

    task_manager[task_id] = {
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
        task_manager,
    )

    return {"task_id": task_id, "status": "started"}


async def _process_video_background(
    task_id: str,
    video_path: Path,
    filename: str,
    estimator_type: str,
    min_confidence: float,
    task_manager: Any,
) -> None:
    """Background task for video processing.

    Args:
        task_id: Unique task identifier.
        video_path: Path to temporary video file.
        filename: Original filename.
        estimator_type: Pose estimation backend.
        min_confidence: Minimum confidence threshold.
        task_manager: Task manager for status updates.
    """
    try:
        task_data = task_manager.get(task_id) or {}
        created_at = task_data.get("created_at", datetime.now(UTC))

        task_manager[task_id] = {
            "status": "processing",
            "progress": 0,
            "created_at": created_at,
        }

        config = VideoProcessingConfig(
            estimator_type=estimator_type, min_confidence=min_confidence
        )
        pipeline = VideoPosePipeline(config)

        result = pipeline.process_video(video_path)

        task_data = task_manager.get(task_id) or {}
        created_at = task_data.get("created_at", datetime.now(UTC))

        task_manager[task_id] = {
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
        task_data = task_manager.get(task_id) or {}
        created_at = task_data.get("created_at", datetime.now(UTC))

        task_manager[task_id] = {
            "status": "failed",
            "error": str(e),
            "created_at": created_at,
        }
    finally:
        if video_path.exists():
            video_path.unlink()
