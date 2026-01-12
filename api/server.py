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
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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

# Initialize FastAPI app
app = FastAPI(
    title="Golf Modeling Suite API",
    description="Research-grade biomechanical analysis and physics simulation API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services
engine_manager: EngineManager | None = None
simulation_service: SimulationService | None = None
analysis_service: AnalysisService | None = None
video_pipeline: VideoPosePipeline | None = None

# Background task storage
active_tasks: dict[str, Any] = {}


@app.on_event("startup")
async def startup_event():
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
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Golf Modeling Suite API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running",
    }


@app.get("/health")
async def health_check():
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
async def get_engines():
    """Get status of all available physics engines."""
    if not engine_manager:
        raise HTTPException(status_code=500, detail="Engine manager not initialized")

    engines = []
    for engine_type in EngineType:
        status = engine_manager.get_engine_status(engine_type)
        engines.append(
            EngineStatusResponse(
                engine_type=engine_type.value,
                status=status.value,
                is_available=engine_manager.is_engine_available(engine_type),
                description=f"{engine_type.value} physics engine",
            )
        )

    return engines


@app.post("/engines/{engine_type}/load")
async def load_engine(engine_type: str, model_path: str | None = None):
    """Load a specific physics engine with optional model."""
    if not engine_manager:
        raise HTTPException(status_code=500, detail="Engine manager not initialized")

    try:
        engine_enum = EngineType(engine_type.upper())
        success = engine_manager.load_engine(engine_enum)

        if not success:
            raise HTTPException(
                status_code=400, detail=f"Failed to load engine: {engine_type}"
            )

        # Load model if provided
        if model_path:
            engine = engine_manager.get_active_engine()
            if engine:
                engine.load_from_path(model_path)

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
async def run_simulation(request: SimulationRequest):
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
):
    """Start an asynchronous simulation."""
    if not simulation_service:
        raise HTTPException(
            status_code=500, detail="Simulation service not initialized"
        )

    task_id = str(uuid.uuid4())

    # Add background task
    background_tasks.add_task(
        simulation_service.run_simulation_background, task_id, request, active_tasks
    )

    return {"task_id": task_id, "status": "started"}


@app.get("/simulate/status/{task_id}")
async def get_simulation_status(task_id: str):
    """Get status of an asynchronous simulation."""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    return active_tasks[task_id]


# Video Analysis Endpoints


@app.post("/analyze/video", response_model=VideoAnalysisResponse)
async def analyze_video(
    file: UploadFile = File(...),
    estimator_type: str = "mediapipe",
    min_confidence: float = 0.5,
    enable_smoothing: bool = True,
):
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
):
    """Start asynchronous video analysis."""
    if not video_pipeline:
        raise HTTPException(status_code=500, detail="Video pipeline not initialized")

    task_id = str(uuid.uuid4())

    # Save file and add background task
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = Path(temp_file.name)

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
):
    """Background task for video processing."""
    try:
        active_tasks[task_id] = {"status": "processing", "progress": 0}

        config = VideoProcessingConfig(
            estimator_type=estimator_type, min_confidence=min_confidence
        )
        pipeline = VideoPosePipeline(config)

        result = pipeline.process_video(video_path)

        # Store result
        active_tasks[task_id] = {
            "status": "completed",
            "result": {
                "filename": filename,
                "total_frames": result.total_frames,
                "valid_frames": result.valid_frames,
                "average_confidence": result.average_confidence,
                "quality_metrics": result.quality_metrics,
            },
        }

    except Exception as e:
        active_tasks[task_id] = {"status": "failed", "error": str(e)}
    finally:
        # Clean up temp file
        if video_path.exists():
            video_path.unlink()


# Analysis Endpoints


@app.post("/analyze/biomechanics", response_model=AnalysisResponse)
async def analyze_biomechanics(request: AnalysisRequest):
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
async def export_results(task_id: str, format: str = "json"):
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
