"""Core health and root routes.

All dependencies are injected via FastAPI's Depends() mechanism.
No module-level mutable state.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends

from src.api.utils.datetime_compat import iso_format, utc_now
from src.shared.python.core.contracts import precondition

from ..dependencies import get_engine_manager

if TYPE_CHECKING:
    from src.shared.python.engine_core.engine_manager import EngineManager

router = APIRouter()


@router.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with API information."""
    return {
        "message": "Golf Modeling Suite API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running",
    }


@router.get(
    "/health",
    responses={
        200: {
            "description": "Server is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "engines_available": 4,
                        "timestamp": "2026-02-18T00:00:00+00:00",
                    }
                }
            },
        }
    },
)
@precondition(
    lambda engine_manager=None: engine_manager is not None,
    "Engine manager must be injected",
)
async def health_check(
    engine_manager: EngineManager = Depends(get_engine_manager),
) -> dict[str, str | int]:
    """Health check endpoint.

    Args:
        engine_manager: Injected engine manager.

    Returns:
        Health status with engine count and timestamp.
    """
    return {
        "status": "healthy",
        "engines_available": len(engine_manager.get_available_engines()),
        "timestamp": iso_format(utc_now()),
    }


@router.get("/diagnostics", response_model=None)
async def get_diagnostics() -> dict:  # type: ignore[type-arg]
    """Get comprehensive diagnostic information for browser mode."""
    repo_root = Path(__file__).parent.parent.parent.parent

    return {
        "backend": {
            "running": True,
            "pid": None,  # Not available in browser mode
            "port": 8001,  # Docker backend port
            "error": None,
        },
        "python_found": True,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "repo_root": str(repo_root),
        "local_server_found": True,  # If this endpoint is hit, server is found
    }
