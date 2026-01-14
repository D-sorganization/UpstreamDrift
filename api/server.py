"""FastAPI server for Golf Modeling Suite.

Provides REST API endpoints for:
- Physics engine management and simulation
- Video-based pose estimation
- Biomechanical analysis
- Data export and visualization

Built on top of the existing EngineManager and PhysicsEngine protocol.
"""

import logging
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from shared.python.engine_manager import EngineManager
from shared.python.engine_registry import EngineType
from shared.python.video_pose_pipeline import VideoPosePipeline, VideoProcessingConfig

from .models.requests import (
    AnalysisRequest,
    SimulationRequest,
)
from .models.responses import (
    AnalysisResponse,
    EngineStatusResponse,
    SimulationResponse,
    VideoAnalysisResponse,
)
from .services.analysis_service import AnalysisService
from .services.simulation_service import SimulationService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app
app = FastAPI(
    title="Golf Modeling Suite API",
    description="Professional biomechanical analysis and physics simulation API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.golfmodelingsuite.com"],
)

# CORS middleware with restricted origins and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React development
        "http://localhost:8080",  # Vue development
        "https://app.golfmodelingsuite.com",  # Production frontend
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    # SECURITY: Restrict headers - do NOT use "*"
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)

# Maximum file upload size (10 MB)
MAX_UPLOAD_SIZE_MB = 10
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]


# SECURITY: Path validation to prevent path traversal attacks
ALLOWED_MODEL_DIRS = [
    Path("shared/models").resolve(),
    Path("models").resolve(),
    Path("data").resolve(),
]


def _validate_model_path(model_path: str) -> str:
    """Validate model path to prevent path traversal attacks.

    Args:
        model_path: User-provided model path

    Returns:
        Validated absolute path string

    Raises:
        HTTPException: If path is invalid or contains traversal attempts
    """
    # SECURITY: Sanitize user input to prevent path traversal
    try:
        user_path = Path(model_path)
    except TypeError as e:
        raise HTTPException(
            status_code=400,
            detail="Invalid path format",
        ) from e

    # SECURITY: Reject absolute paths - user input must be relative
    if user_path.is_absolute():
        raise HTTPException(
            status_code=400,
            detail="Invalid path: absolute paths are not allowed",
        )

    # SECURITY: Reject paths with parent directory references
    if ".." in user_path.parts:
        raise HTTPException(
            status_code=400,
            detail="Invalid path: parent directory references not allowed",
        )

    # Build candidate paths under each allowed directory and check them
    for allowed_dir in ALLOWED_MODEL_DIRS:
        try:
            # SECURITY: Resolve to absolute path and validate it stays within allowed_dir
            candidate = (allowed_dir / user_path).resolve()
        except (ValueError, OSError) as e:
            raise HTTPException(
                status_code=400,
                detail="Invalid path format",
            ) from e

        # SECURITY: Ensure the resolved path is still within the allowed directory
        try:
            candidate.relative_to(allowed_dir)
        except ValueError:
            # Path escaped the allowed directory (traversal attempt)
            continue

        # Check if this valid candidate exists
        if candidate.exists():
            # SECURITY: Return only the validated, sanitized path
            return str(candidate)

    raise HTTPException(
        status_code=404,
        detail="Model file not found in allowed directories",
    )


# Global services
engine_manager: EngineManager | None = None
simulation_service: SimulationService | None = None
analysis_service: AnalysisService | None = None
video_pipeline: VideoPosePipeline | None = None


class TaskManager:
    """Thread-safe task manager with TTL cleanup and size limits.

    Prevents memory leak from unbounded task accumulation.

    Performance Fix for Issue #1:
    - Tasks expire after TTL_SECONDS (default 1 hour)
    - Maximum MAX_TASKS entries with LRU eviction
    - Automatic cleanup on each access
    """

    # Configuration constants
    TTL_SECONDS: int = 3600  # 1 hour
    MAX_TASKS: int = 1000  # Maximum stored tasks

    def __init__(self) -> None:
        """Initialize task manager with empty storage."""
        import threading
        import time

        self._tasks: dict[str, dict[str, Any]] = {}
        self._timestamps: dict[str, float] = {}
        self._lock = threading.Lock()
        self._time = time  # Store reference for use in methods

    def _cleanup_expired(self) -> None:
        """Remove expired tasks. Called internally under lock."""
        current_time = self._time.time()
        expired_keys = [
            task_id
            for task_id, timestamp in self._timestamps.items()
            if current_time - timestamp > self.TTL_SECONDS
        ]
        for task_id in expired_keys:
            self._tasks.pop(task_id, None)
            self._timestamps.pop(task_id, None)

    def _enforce_size_limit(self) -> None:
        """Remove oldest tasks if over limit. Called internally under lock."""
        if len(self._tasks) <= self.MAX_TASKS:
            return

        # Sort by timestamp and remove oldest entries
        sorted_by_age = sorted(self._timestamps.items(), key=lambda x: x[1])
        to_remove = len(self._tasks) - self.MAX_TASKS
        for task_id, _ in sorted_by_age[:to_remove]:
            self._tasks.pop(task_id, None)
            self._timestamps.pop(task_id, None)

    def set(self, task_id: str, data: dict[str, Any]) -> None:
        """Store or update a task.

        Args:
            task_id: Unique task identifier
            data: Task data dictionary
        """
        with self._lock:
            self._cleanup_expired()
            self._tasks[task_id] = data
            self._timestamps[task_id] = self._time.time()
            self._enforce_size_limit()

    def get(self, task_id: str) -> dict[str, Any] | None:
        """Retrieve a task by ID.

        Args:
            task_id: Task identifier

        Returns:
            Task data or None if not found
        """
        with self._lock:
            self._cleanup_expired()
            return self._tasks.get(task_id)

    def __contains__(self, task_id: str) -> bool:
        """Check if task exists."""
        with self._lock:
            self._cleanup_expired()
            return task_id in self._tasks

    def __setitem__(self, task_id: str, data: dict[str, Any]) -> None:
        """Dict-like setter for backward compatibility."""
        self.set(task_id, data)

    def __getitem__(self, task_id: str) -> dict[str, Any]:
        """Dict-like getter for backward compatibility."""
        result = self.get(task_id)
        if result is None:
            raise KeyError(task_id)
        return result


# Background task storage with TTL cleanup (Performance Issue #1 fix)
active_tasks = TaskManager()


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize services on startup."""
    global engine_manager, simulation_service, analysis_service, video_pipeline

    try:
        # Initialize engine manager
        engine_manager = EngineManager()
        logger.info("Engine manager initialized")

        # Initialize services
        simulation_service = SimulationService(engine_manager)
        analysis_service = AnalysisService(engine_manager)

        # Initialize video pipeline with default config
        video_config = VideoProcessingConfig(
            estimator_type="mediapipe",
            min_confidence=0.5,
            enable_temporal_smoothing=True,
        )
        video_pipeline = VideoPosePipeline(video_config)

        logger.info("Golf Modeling Suite API started successfully")

    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with API information."""
    return {
        "message": "Golf Modeling Suite API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running",
    }


@app.get("/health")
async def health_check() -> dict[str, str | int]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "engines_available": (
            len(engine_manager.get_available_engines()) if engine_manager else 0
        ),
        "timestamp": "2026-01-12T00:00:00Z",
    }


# Engine Management Endpoints


@app.get("/engines", response_model=list[EngineStatusResponse])
async def get_engines() -> list[EngineStatusResponse]:
    """Get status of all available physics engines."""
    if not engine_manager:
        raise HTTPException(status_code=500, detail="Engine manager not initialized")

    engines = []
    for engine_type in EngineType:
        status = engine_manager.get_engine_status(engine_type)
        # Check if engine is available by trying to get available engines
        available_engines = engine_manager.get_available_engines()
        is_available = engine_type in available_engines

        engines.append(
            EngineStatusResponse(
                engine_type=engine_type.value,
                status=status.value,
                is_available=is_available,
                description=f"{engine_type.value} physics engine",
            )
        )

    return engines


@app.post("/engines/{engine_type}/load")
async def load_engine(
    engine_type: str, model_path: str | None = None
) -> dict[str, str]:
    """Load a specific physics engine with optional model."""
    if not engine_manager:
        raise HTTPException(status_code=500, detail="Engine manager not initialized")

    try:
        engine_enum = EngineType(engine_type.upper())
        engine_manager._load_engine(
            engine_enum
        )  # This method doesn't return success status

        # Check if engine was loaded successfully
        engine = engine_manager.get_active_physics_engine()
        if not engine:
            raise HTTPException(
                status_code=400, detail=f"Failed to load engine: {engine_type}"
            )

        # SECURITY: Validate model path to prevent path traversal
        if model_path:
            engine = engine_manager.get_active_physics_engine()
            if engine:
                # Validate path: no parent directory traversal, must exist
                validated_path = _validate_model_path(model_path)
                engine.load_from_path(validated_path)

        return {"message": f"Engine {engine_type} loaded successfully"}

    except ValueError as e:
        raise HTTPException(
            status_code=400, detail=f"Unknown engine type: {engine_type}"
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error loading engine: {str(e)}"
        ) from e


# Simulation Endpoints


@app.post("/simulate", response_model=SimulationResponse)
async def run_simulation(request: SimulationRequest) -> SimulationResponse:
    """Run a physics simulation."""
    if not simulation_service:
        raise HTTPException(
            status_code=500, detail="Simulation service not initialized"
        )

    try:
        result = await simulation_service.run_simulation(request)
        return result
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Simulation failed: {str(e)}"
        ) from e


@app.post("/simulate/async")
async def run_simulation_async(
    request: SimulationRequest, background_tasks: BackgroundTasks
) -> dict[str, str]:
    """Start an asynchronous simulation."""
    if not simulation_service:
        raise HTTPException(
            status_code=500, detail="Simulation service not initialized"
        )

    task_id = str(uuid.uuid4())

    # Initialize task with timestamp
    active_tasks[task_id] = {
        "status": "started",
        "created_at": datetime.now(UTC),
    }

    # Add background task
    background_tasks.add_task(
        simulation_service.run_simulation_background, task_id, request, active_tasks  # type: ignore[arg-type]
    )

    return {"task_id": task_id, "status": "started"}


@app.get("/simulate/status/{task_id}")
async def get_simulation_status(task_id: str) -> dict[str, Any]:
    """Get status of an asynchronous simulation."""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    return dict(active_tasks[task_id])


# Video Analysis Endpoints


@app.post("/analyze/video", response_model=VideoAnalysisResponse)
async def analyze_video(
    file: UploadFile = File(...),
    estimator_type: str = "mediapipe",
    min_confidence: float = 0.5,
    enable_smoothing: bool = True,
) -> VideoAnalysisResponse:
    """Analyze golf swing from uploaded video."""
    if not video_pipeline:
        raise HTTPException(status_code=500, detail="Video pipeline not initialized")

    # Validate file type
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        # Configure pipeline for this request
        config = VideoProcessingConfig(
            estimator_type=estimator_type,
            min_confidence=min_confidence,
            enable_temporal_smoothing=enable_smoothing,
        )
        pipeline = VideoPosePipeline(config)

        # Process video
        result = pipeline.process_video(temp_path)

        # Clean up temp file
        temp_path.unlink()

        # Convert to response format
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
                for pose in result.pose_results[:100]  # Limit response size
            ],
        )

        return response

    except Exception as e:
        logger.error(f"Video analysis error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Video analysis failed: {str(e)}"
        ) from e


@app.post("/analyze/video/async")
async def analyze_video_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    estimator_type: str = "mediapipe",
    min_confidence: float = 0.5,
) -> dict[str, str]:
    """Start asynchronous video analysis."""
    if not video_pipeline:
        raise HTTPException(status_code=500, detail="Video pipeline not initialized")

    task_id = str(uuid.uuid4())

    # Save file and add background task
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = Path(temp_file.name)

    # Initialize task with timestamp
    active_tasks[task_id] = {
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
        task_data = active_tasks.get(task_id)
        if task_data is None:
            task_data = {}
        created_at = task_data.get("created_at", datetime.now(UTC))

        active_tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "created_at": created_at,
        }

        config = VideoProcessingConfig(
            estimator_type=estimator_type, min_confidence=min_confidence
        )
        pipeline = VideoPosePipeline(config)

        result = pipeline.process_video(video_path)

        # Store result
        task_data = active_tasks.get(task_id)
        if task_data is None:
            task_data = {}
        created_at = task_data.get("created_at", datetime.now(UTC))

        active_tasks[task_id] = {
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
        task_data = active_tasks.get(task_id)
        if task_data is None:
            task_data = {}
        created_at = task_data.get("created_at", datetime.now(UTC))

        active_tasks[task_id] = {
            "status": "failed",
            "error": str(e),
            "created_at": created_at,
        }
    finally:
        # Clean up temp file
        if video_path.exists():
            video_path.unlink()


# Analysis Endpoints


@app.post("/analyze/biomechanics", response_model=AnalysisResponse)
async def analyze_biomechanics(request: AnalysisRequest) -> AnalysisResponse:
    """Perform biomechanical analysis on simulation data."""
    if not analysis_service:
        raise HTTPException(status_code=500, detail="Analysis service not initialized")

    try:
        result = await analysis_service.analyze_biomechanics(request)
        return result
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}") from e


# Export Endpoints


@app.get("/export/{task_id}")
async def export_results(task_id: str, format: str = "json") -> JSONResponse:
    """Export analysis results in specified format."""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = active_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed")

    # Generate export file
    # Implementation depends on specific export requirements
    return JSONResponse(content=task["result"])


if __name__ == "__main__":
    uvicorn.run(
        "api.server:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
