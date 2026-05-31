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
    Legacy un-prefixed and ``/api/`` routes are also registered for backward
    compatibility.
"""

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded


from src.shared.python.config.environment import get_environment
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
from .rate_limit import limiter
from .route_registry import register_routes, ws_compatible_auth_dependency
from .services.analysis_service import AnalysisService
from .services.chat_service import ChatService
from .services.simulation_service import SimulationService
from .task_manager import TaskManager
from .utils.tracing import RequestTracer
from .versioning import get_app_version
from .routes import chat_ws, realtime as realtime_route, simulation_ws
from src.shared.python.app_state import agent_context, get_agent_state_store

setup_logging()
logger = get_logger(__name__)

# API version constant
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"


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


# Background task storage with TTL cleanup and concurrency limits
active_tasks = TaskManager()


def rate_limit_exceeded_handler(request: Request, exc: Exception) -> Response:
    """Build a JSON response with Retry-After header when rate limit is exceeded."""
    detail = getattr(exc, "detail", "Too Many Requests")
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {detail}"},
        headers={"Retry-After": "60"},
    )


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan: startup and shutdown.

    All services are stored in app.state for proper dependency injection
    via FastAPI's Depends() mechanism. This enables:
    - Better testability (dependencies can be overridden)
    - Cleaner separation of concerns
    - Type-safe dependency resolution
    """
    task_manager: TaskManager | None = None
    try:
        # Validate environment variables before any service initialisation.
        # env_validator checks GOLF_API_SECRET_KEY, DATABASE_URL, and the
        # production checklist; it raises on critical failures so misconfigured
        # deployments fail fast instead of silently using insecure defaults.
        # _assert_production_secrets() is a hard gate: it raises RuntimeError if
        # production mode is active with .env.example placeholder credentials.
        from src.shared.python.security.env_validator import (
            _assert_production_secrets,
            validate_environment,
        )

        _assert_production_secrets()
        validate_environment(raise_on_error=get_environment() == "production")

        # Initialize database (Issue #544)
        logger.info("Initializing database...")
        init_db()
        logger.info("Database initialized successfully")

        # Initialize engine manager
        engine_manager = EngineManager()
        fastapi_app.state.engine_manager = engine_manager
        logger.info("Engine manager initialized")

        # Initialize services and store in app.state for dependency injection
        fastapi_app.state.simulation_service = SimulationService(engine_manager)
        fastapi_app.state.analysis_service = AnalysisService(engine_manager)
        task_manager = TaskManager()
        fastapi_app.state.task_manager = task_manager
        fastapi_app.state.logger = logger
        fastapi_app.state.api_started_at = time.time()
        fastapi_app.state.static_files_mounted = False

        # Initialize video pipeline with default config
        video_pipeline = _init_video_pipeline()
        fastapi_app.state.video_pipeline = video_pipeline

        # Initialize chat service wired to app state (issue #5470)
        fastapi_app.state.chat_service = ChatService(
            app_state_provider=lambda: agent_context(get_agent_state_store())
        )

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

    try:
        yield
    finally:
        if task_manager is not None:
            await task_manager.shutdown()


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
        "Legacy un-prefixed and `/api/` routes are maintained for backward "
        "compatibility."
    ),
    version=get_app_version(),
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
    lifespan=lifespan,
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
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)


# SECURITY: middleware registration
app.middleware("http")(add_security_headers)
app.middleware("http")(validate_upload_size)

# TRACEABILITY: Request tracing middleware for diagnostics
_tracer = RequestTracer()
app.middleware("http")(_tracer.trace_request)


# All services are stored in app.state and accessed via Depends() in routes.
# No module-level mutable state in route modules.


# ── Route Registration ──────────────────────────────────────────
# Use plugin-style auto-discovery instead of 20+ explicit imports (#1485).
# Routes are registered at root, /api, and /api/v1 (#1488).

# Register all routes at root level (backward compatibility)
_root_count = register_routes(app, prefix="")
logger.info("Registered %d route modules at root prefix", _root_count)

# Register all routes under /api prefix (legacy API compatibility)
_legacy_api_count = register_routes(app, prefix="/api")
logger.info("Registered %d route modules under /api", _legacy_api_count)

# Register all routes under /api/v1/ prefix (versioned API)
_versioned_count = register_routes(app, prefix=API_PREFIX)
logger.info("Registered %d route modules under %s", _versioned_count, API_PREFIX)

# Register explicitly excluded WebSocket routes.
#
# These routers mix WebSocket endpoints (which self-authenticate via
# resolve_ws_user) with HTTP endpoints. The HTTP endpoints were previously
# unauthenticated in cloud mode (issues #6888, #6889), allowing unauthenticated
# broadcast injection (POST /realtime/publish) and chat-session enumeration
# (GET /chat/sessions, /history). ws_compatible_auth_dependency enforces the
# bearer header on HTTP requests only; WebSocket connections fall through to the
# route handler's own auth check, and local/auth-disabled mode remains open.
_excluded_ws_auth = [Depends(ws_compatible_auth_dependency)]
app.include_router(chat_ws.router, prefix=API_PREFIX, dependencies=_excluded_ws_auth)
app.include_router(simulation_ws.router, prefix=API_PREFIX)
app.include_router(chat_ws.router, prefix="/api", dependencies=_excluded_ws_auth)
app.include_router(simulation_ws.router, prefix="/api")
app.include_router(chat_ws.router, prefix="", dependencies=_excluded_ws_auth)
app.include_router(simulation_ws.router, prefix="")

# Realtime IPC layer (issue #4997) — combined HTTP + WS endpoints under
# /realtime; mounted at root so cross-process clients (WSPubSub) can use the
# canonical "/realtime/publish" and "/realtime/subscribe" paths.
app.include_router(realtime_route.router, prefix="", dependencies=_excluded_ws_auth)
app.include_router(
    realtime_route.router, prefix=API_PREFIX, dependencies=_excluded_ws_auth
)


if __name__ == "__main__":
    # SECURITY FIX: Only enable auto-reload in development mode
    # Auto-reload in production can enable code injection attacks
    # Use canonical get_environment() from shared config (normalises "dev", "prod", etc.)
    enable_reload = get_environment() == "development"

    if enable_reload:
        logger.warning("Running with auto-reload enabled (development mode)")

    uvicorn.run(
        "src.api.server:app",
        host=get_server_host(),
        port=get_server_port(),
        reload=enable_reload,
        log_level="info",
    )
