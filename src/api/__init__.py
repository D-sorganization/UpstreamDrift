"""API package for Golf Modeling Suite."""

from .auth import (
    APIKeyAuth,
    OAuth2Config,
    RateLimitConfig,
    TokenManager,
    generate_token,
    get_current_user,
    refresh_token,
    require_auth,
    validate_token,
)
from .middleware import (
    add_security_headers,
    get_upload_limit_mb,
    handle_api_errors,
)
from .models import (
    ChatMessageCreate,
    ChatRequest,
    ChatResponse,
    PhysicsAnalysisResponse,
    PhysicsRequest,
)
from .routes import (
    launcher_router,
    models_router,
    physics_router,
    simulation_router,
)
from .services import (
    AnalysisService,
    ChatService,
    LauncherService,
    SimulationService,
    SimulationStats,
)

__all__: list[str] = [
    # auth
    "APIKeyAuth",
    "OAuth2Config",
    "RateLimitConfig",
    "TokenManager",
    "generate_token",
    "get_current_user",
    "refresh_token",
    "require_auth",
    "validate_token",
    # middleware
    "add_security_headers",
    "get_upload_limit_mb",
    "handle_api_errors",
    # models
    "ChatMessageCreate",
    "ChatRequest",
    "ChatResponse",
    "PhysicsAnalysisResponse",
    "PhysicsRequest",
    # routes
    "launcher_router",
    "models_router",
    "physics_router",
    "simulation_router",
    # services
    "AnalysisService",
    "ChatService",
    "LauncherService",
    "SimulationService",
    "SimulationStats",
]
