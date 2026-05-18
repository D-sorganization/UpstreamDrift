"""API package for Golf Modeling Suite."""

from .auth import (
    get_current_user,
)
from .middleware import (
    add_security_headers,
    handle_api_errors,
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
    "get_current_user",
    # middleware
    "add_security_headers",
    "handle_api_errors",
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
