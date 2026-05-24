"""Video analysis routes.

All dependencies are injected via FastAPI's Depends() mechanism.
No module-level mutable state.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile

from src.api.config import (
    MAX_CONFIDENCE,
    MAX_POSE_DATA_ENTRIES,
    MIN_CONFIDENCE,
    VALID_ESTIMATOR_TYPES,
)
from src.api.middleware.upload_limits import write_upload_file_to_path
from src.api.utils.datetime_compat import UTC
from src.shared.python.core.contracts import precondition
from src.shared.python.logging_pkg.logging_config import get_logger as get_module_logger

from ..dependencies import get_logger, get_task_manager, get_video_pipeline
from ..models.responses import VideoAnalysisResponse

if TYPE_CHECKING:
    from src.shared.python.gui_pkg.video_pose_pipeline import VideoPosePipeline

router = APIRouter()
logger = get_module_logger(__name__)
VIDEO_UPLOAD_MAX_BYTES = 100 * 1024 * 1024
VIDEO_SIGNATURE_READ_BYTES = 32


def _load_video_pipeline_classes() -> tuple[type, type]:
    """Lazily import the video pose pipeline classes.

    The pipeline transitively requires ``cv2`` (and ``mediapipe``). In the slim
    runtime image these are intentionally absent to keep the image small
    (see issue #2809). Importing them eagerly at module load time caused the
    entire route module to be skipped during route discovery, which produced
    404s on ``/api/v1/video/*`` instead of a meaningful error.

    By deferring the import to request time we can keep the route registered
    and return a 503 with a clear message when the optional dependencies are
    not installed.

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


def _looks_like_supported_video_bytes(header: bytes) -> bool:
    """Return True when the upload header matches a supported video container."""
    if len(header) >= 12 and header[4:8] == b"ftyp":
        return True
    if header.startswith(b"\x1a\x45\xdf\xa3"):
        return True
    return len(header) >= 12 and header.startswith(b"RIFF") and header[8:12] == b"AVI "


async def _validate_video_upload(file: UploadFile) -> None:
    """Reject uploads whose metadata or leading bytes do not look like video."""
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    if file.size and file.size > VIDEO_UPLOAD_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Video file too large. Maximum size is 100MB.",
        )

    header = await file.read(VIDEO_SIGNATURE_READ_BYTES)
    await file.seek(0)
    if not header or not _looks_like_supported_video_bytes(header):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file content does not match a supported video format.",
        )


@router.post("/analyze/video", response_model=VideoAnalysisResponse)
@precondition(
    lambda file=None, estimator_type="mediapipe", min_confidence=0.5, enable_smoothing=True, video_pipeline=None, logger=None: (
        estimator_type is not None
        and len(estimator_type.strip()) > 0
        and 0.0 <= min_confidence <= 1.0
    ),
    "Estimator type must be non-empty and min_confidence must be in [0.0, 1.0]",
)
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

    await _validate_video_upload(file)

    temp_path: Path | None = None
    try:
        temp_fd, temp_file_name = tempfile.mkstemp(suffix=".mp4")
        os.close(temp_fd)
        temp_path = Path(temp_file_name)
        await write_upload_file_to_path(
            file,
            temp_path,
            max_bytes=VIDEO_UPLOAD_MAX_BYTES,
        )

        video_pipeline_cls, video_config_cls = _load_video_pipeline_classes()
        config = video_config_cls(
            estimator_type=estimator_type,
            min_confidence=min_confidence,
            enable_temporal_smoothing=enable_smoothing,
        )
        pipeline = video_pipeline_cls(config)
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
    except (FileNotFoundError, OSError) as e:
        if logger:
            logger.exception("Video analysis error")
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
@precondition(
    lambda background_tasks=None, file=None, estimator_type="mediapipe", min_confidence=0.5, video_pipeline=None, task_manager=None: (
        estimator_type is not None
        and len(estimator_type.strip()) > 0
        and 0.0 <= min_confidence <= 1.0
    ),
    "Estimator type must be non-empty and min_confidence must be in [0.0, 1.0]",
)
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

    await _validate_video_upload(file)

    task_id = str(uuid.uuid4())

    # Use managed artifact directory for temp files (issue #3938 alignment)
    artifact_dir = os.environ.get(
        "ARTIFACT_DIR",
        os.path.join(tempfile.gettempdir(), "upstream_drift_artifacts", "video"),
    )
    os.makedirs(artifact_dir, exist_ok=True)

    temp_fd, temp_file_name = tempfile.mkstemp(
        suffix=".mp4",
        dir=artifact_dir,
    )
    os.close(temp_fd)
    temp_path = Path(temp_file_name)
    await write_upload_file_to_path(
        file,
        temp_path,
        max_bytes=VIDEO_UPLOAD_MAX_BYTES,
    )

    # Compute input hash for reproducibility
    input_hash = hashlib.sha256(temp_path.read_bytes()).hexdigest()[:16]

    # Store task with durable metadata for replay (issue #3941 alignment)
    task_manager.set(
        task_id,
        {
            "status": "pending",
            "created_at": datetime.now(UTC),
            "input_hash": input_hash,
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size": file.size,
            "estimator_type": estimator_type,
            "min_confidence": min_confidence,
        },
    )

    background_tasks.add_task(
        _process_video_background,
        task_id,
        temp_path,
        file.filename or "unknown",
        estimator_type,
        min_confidence,
        input_hash,
        task_manager,
    )

    return {"task_id": task_id, "status": "pending"}


async def _process_video_background(
    task_id: str,
    video_path: Path,
    filename: str,
    estimator_type: str,
    min_confidence: float,
    input_hash: str,
    task_manager: Any,
) -> None:
    """Background task for video processing.

    Runs blocking video processing in thread pool to avoid blocking
    the asyncio event loop (issue #3942).

    Args:
        task_id: Unique task identifier.
        video_path: Path to temporary video file.
        filename: Original filename.
        estimator_type: Pose estimation backend.
        min_confidence: Minimum confidence threshold.
        input_hash: Hash of input video for reproducibility.
        task_manager: Task manager for status updates.
    """
    try:
        task_data = task_manager.get(task_id) or {}
        created_at = task_data.get("created_at", datetime.now(UTC))

        task_manager.set(
            task_id,
            {
                "status": "processing",
                "progress": 0,
                "created_at": created_at,
                "input_hash": input_hash,
            },
        )

        video_pipeline_cls, video_config_cls = _load_video_pipeline_classes()
        config = video_config_cls(
            estimator_type=estimator_type, min_confidence=min_confidence
        )
        pipeline = video_pipeline_cls(config)

        result = await asyncio.to_thread(pipeline.process_video, video_path)

        task_data = task_manager.get(task_id) or {}
        created_at = task_data.get("created_at", datetime.now(UTC))

        # Store comprehensive result metadata for reproducibility
        task_manager.set(
            task_id,
            {
                "status": "completed",
                "created_at": created_at,
                "completed_at": datetime.now(UTC),
                "input_hash": input_hash,
                "result": {
                    "filename": filename,
                    "total_frames": result.total_frames,
                    "valid_frames": result.valid_frames,
                    "average_confidence": result.average_confidence,
                    "quality_metrics": result.quality_metrics,
                    "estimator_type": estimator_type,
                    "min_confidence": min_confidence,
                },
            },
        )

    except (RuntimeError, ValueError, OSError, ImportError, HTTPException) as e:
        task_data = task_manager.get(task_id) or {}
        created_at = task_data.get("created_at", datetime.now(UTC))
        error = e.detail if isinstance(e, HTTPException) else str(e)

        task_manager.set(
            task_id,
            {
                "status": "failed",
                "error": error,
                "created_at": created_at,
                "completed_at": datetime.now(UTC),
                "input_hash": input_hash,
            },
        )
    finally:
        # Hardened cleanup with error handling (issue #3942)
        if video_path.exists():
            try:
                video_path.unlink()
            except OSError as cleanup_error:
                logger.warning(
                    "Failed to clean up temp video %s: %s",
                    video_path,
                    cleanup_error,
                )
