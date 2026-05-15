"""API route registration."""

from .launcher import router as launcher_router
from .models import router as models_router
from .physics import router as physics_router
from .simulation import router as simulation_router

__all__: list[str] = [
    "launcher_router",
    "models_router",
    "physics_router",
    "simulation_router",
]
