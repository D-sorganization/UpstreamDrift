"""FastAPI server for UpstreamDrift.

Provides REST API endpoints for:
- Physics engine management and simulation
- Video-based pose estimation
- Biomechanical analysis
- Data export and visualization

Built on top of the existing EngineManager and PhysicsEngine protocol.
"""

import os
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.shared.python.engine_manager import EngineManager

# Configure logging - use centralized logging config
from src.shared.python.logging_config import get_logger, setup_logging

from .config import (
    get_allowed_hosts,
    get_cors_origins,
    get_server_host,
    get_server_port,
)
from .database import init_db
from .middleware.security_headers import add_security_headers
from .middleware.upload_limits import validate_upload_size
from .routes import analysis as analysis_routes
from .routes import analysis_tools as analysis_tools_routes
from .routes import auth as auth_routes
from .routes import core as core_routes
from .routes import dataset as dataset_routes
from .routes import engines as engine_routes
from .routes import export as export_routes
from .routes import launcher as launcher_routes
from .routes import models as model_routes
from .routes import physics as physics_routes
from .routes import simulation as simulation_routes
from .routes import terrain as terrain_routes
from .routes import video as video_routes
from .services.analysis_service import AnalysisService
from .services.simulation_service import SimulationService
from .utils.tracing import RequestTracer

setup_logging()
logger = get_logger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app
app = FastAPI(
    title="UpstreamDrift API",
    description="Professional biomechanical analysis and physics simulation API",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Security middleware
allowed_hosts = get_allowed_hosts()
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=allowed_hosts,
)

# CORS middleware with restricted origins and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    # SECURITY: Restrict headers - do NOT use "*"
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]


# SECURITY: middleware registration
app.middleware("http")(add_security_headers)
app.middleware("http")(validate_upload_size)

# TRACEABILITY: Request tracing middleware for diagnostics
_tracer = RequestTracer()
app.middleware("http")(_tracer.trace_request)


# All services are stored in app.state and accessed via Depends() in routes.
# No module-level mutable state in route modules.


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
    """Initialize services on startup.

    All services are stored in app.state for proper dependency injection
    via FastAPI's Depends() mechanism. This enables:
    - Better testability (dependencies can be overridden)
    - Cleaner separation of concerns
    - Type-safe dependency resolution
    """
    try:
        # Initialize database (Issue #544)
        logger.info("Initializing database...")
        init_db()
        logger.info("Database initialized successfully")

        # Initialize engine manager
        engine_manager = EngineManager()
        app.state.engine_manager = engine_manager
        logger.info("Engine manager initialized")

        # Initialize services and store in app.state for dependency injection
        app.state.simulation_service = SimulationService(engine_manager)
        app.state.analysis_service = AnalysisService(engine_manager)
        app.state.task_manager = active_tasks
        app.state.logger = logger

        # Initialize video pipeline with default config
        video_pipeline = None
        try:
            from src.shared.python.video_pose_pipeline import (
                VideoPosePipeline,
                VideoProcessingConfig,
            )

            video_config = VideoProcessingConfig(
                estimator_type="mediapipe",
                min_confidence=0.5,
                enable_temporal_smoothing=True,
            )
            video_pipeline = VideoPosePipeline(video_config)
        except ImportError as e:
            logger.info("MediaPipe not installed, video features disabled: %s", e)
        except AttributeError as e:
            logger.warning(
                "MediaPipe installed but incompatible, video features disabled: %s",
                e,
            )
        except OSError as e:
            logger.warning(
                "Video pipeline failed to initialize (camera/device issue): %s", e
            )
        except RuntimeError as e:
            logger.warning("Video pipeline runtime initialization failed: %s", e)

        app.state.video_pipeline = video_pipeline

        # All routes now use FastAPI Depends() for dependency injection.
        # No legacy configure() calls needed.

        logger.info("Golf Modeling Suite API started successfully")

    except OSError as e:
        logger.error("Database or file system error during initialization: %s", e)
        raise
    except ImportError as e:
        logger.error("Missing required dependency: %s", e)
        raise
    except RuntimeError as e:
        logger.error("Engine initialization failed: %s", e)
        raise
    except Exception as e:
        logger.exception("Unexpected error during API initialization: %s", e)
        raise


# Include routers
app.include_router(auth_routes.router)
app.include_router(core_routes.router)
app.include_router(engine_routes.router)
app.include_router(simulation_routes.router)
app.include_router(video_routes.router)
app.include_router(analysis_routes.router)
app.include_router(export_routes.router)
app.include_router(launcher_routes.router)
app.include_router(terrain_routes.router)
app.include_router(dataset_routes.router)
app.include_router(physics_routes.router)
app.include_router(model_routes.router)
app.include_router(analysis_tools_routes.router)


if __name__ == "__main__":
    # SECURITY FIX: Only enable auto-reload in development mode
    # Auto-reload in production can enable code injection attacks
    environment = os.getenv("ENVIRONMENT", "development").lower()
    enable_reload = environment == "development"

    if enable_reload:
        logger.warning("⚠️  Running with auto-reload enabled (development mode)")

    uvicorn.run(
        "api.server:app",
        host=get_server_host(),
        port=get_server_port(),
        reload=enable_reload,
        log_level="info",
    )
