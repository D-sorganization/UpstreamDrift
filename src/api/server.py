"""FastAPI server for UpstreamDrift.

Provides REST API endpoints for:
- Physics engine management and simulation
- Video-based pose estimation
- Biomechanical analysis
- Data export and visualization

Built on top of the existing EngineManager and PhysicsEngine protocol.

Architecture (#1485):
    Route loading uses a registry/plugin pattern via ``route_registry.py``.
    Adding a new route module requires only creating the file in
    ``src/api/routes/`` with a top-level ``router`` attribute.

API Versioning (#1488):
    All routes are served under ``/api/v1/`` prefix for forward compatibility.
    Legacy un-prefixed routes are also registered for backward compatibility.
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

from src.shared.python.engine_core.engine_manager import EngineManager

# Configure logging - use centralized logging config
from src.shared.python.logging_pkg.logging_config import get_logger, setup_logging

from .config import (
    get_allowed_hosts,
    get_cors_origins,
    get_server_host,
    get_server_port,
)
from .database import init_db
from .middleware.security_headers import add_security_headers
from .middleware.upload_limits import validate_upload_size
from .route_registry import register_routes
from .services.analysis_service import AnalysisService
from .services.simulation_service import SimulationService
from .task_manager import TaskManager
from .utils.tracing import RequestTracer

setup_logging()
logger = get_logger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# API version constant
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"

# Initialize FastAPI app with enhanced OpenAPI metadata (#1488)
app = FastAPI(
    title="UpstreamDrift API",
    description=(
        "Professional biomechanical analysis and physics simulation API.\n\n"
        "## Features\n"
        "- Multi-engine physics simulation (MuJoCo, Drake, Pinocchio, OpenSim, MyoSuite)\n"
        "- Video-based pose estimation and motion capture\n"
        "- Biomechanical analysis (kinematics, kinetics, energetics)\n"
        "- Asynchronous simulation with job status tracking\n"
        "- Real-time WebSocket streaming\n\n"
        "## Versioning\n"
        f"Current API version: **{API_VERSION}**. "
        f"All endpoints are available under `{API_PREFIX}/` prefix.\n"
        "Legacy un-prefixed routes are maintained for backward compatibility."
    ),
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "engines",
            "description": "Physics engine lifecycle management",
        },
        {
            "name": "simulation",
            "description": "Synchronous and asynchronous simulation execution",
        },
        {
            "name": "analysis",
            "description": "Biomechanical analysis and metrics",
        },
        {
            "name": "video",
            "description": "Video-based pose estimation and motion capture",
        },
        {
            "name": "export",
            "description": "Data export in multiple formats",
        },
        {
            "name": "models",
            "description": "URDF/MJCF model management and exploration",
        },
    ],
    responses={
        503: {"description": "Service not initialized"},
        429: {"description": "Rate limit exceeded"},
    },
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

# Background task storage with TTL cleanup and concurrency limits
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
        video_pipeline = _init_video_pipeline()
        app.state.video_pipeline = video_pipeline

        # All routes now use FastAPI Depends() for dependency injection.
        # No legacy configure() calls needed.

        logger.info("Golf Modeling Suite API %s started successfully", API_PREFIX)

    except OSError as e:
        logger.error("Database or file system error during initialization: %s", e)
        raise
    except ImportError as e:
        logger.error("Missing required dependency: %s", e)
        raise
    except RuntimeError as e:
        logger.error("Engine initialization failed: %s", e)
        raise
    except (TypeError, AttributeError) as e:
        logger.exception("Unexpected error during API initialization: %s", e)
        raise


def _init_video_pipeline() -> Any:
    """Initialize the video pose pipeline, returning None on failure.

    Extracted from startup_event for SRP and testability.
    """
    try:
        from src.shared.python.gui_pkg.video_pose_pipeline import (
            VideoPosePipeline,
            VideoProcessingConfig,
        )

        video_config = VideoProcessingConfig(
            estimator_type="mediapipe",
            min_confidence=0.5,
            enable_temporal_smoothing=True,
        )
        return VideoPosePipeline(video_config)
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
    return None


# ── Route Registration ──────────────────────────────────────────
# Use plugin-style auto-discovery instead of 20+ explicit imports (#1485).
# Routes are registered both at root (backward compat) and under /api/v1/ (#1488).

# Register all routes at root level (backward compatibility)
_root_count = register_routes(app, prefix="")
logger.info("Registered %d route modules at root prefix", _root_count)

# Register all routes under /api/v1/ prefix (versioned API)
_versioned_count = register_routes(app, prefix=API_PREFIX)
logger.info("Registered %d route modules under %s", _versioned_count, API_PREFIX)


if __name__ == "__main__":
    # SECURITY FIX: Only enable auto-reload in development mode
    # Auto-reload in production can enable code injection attacks
    environment = os.getenv("ENVIRONMENT", "development").lower()
    enable_reload = environment == "development"

    if enable_reload:
        logger.warning("Running with auto-reload enabled (development mode)")

    uvicorn.run(
        "api.server:app",
        host=get_server_host(),
        port=get_server_port(),
        reload=enable_reload,
        log_level="info",
    )
