"""Core health and root routes."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter

from src.api.utils.datetime_compat import iso_format, utc_now
from src.shared.python.engine_manager import EngineManager

router = APIRouter()

_engine_manager: EngineManager | None = None


def configure(engine_manager: EngineManager | None) -> None:
    """Configure dependencies for core routes."""
    global _engine_manager
    _engine_manager = engine_manager


@router.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with API information."""
    return {
        "message": "Golf Modeling Suite API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running",
    }


@router.get("/health")
async def health_check() -> dict[str, str | int]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "engines_available": (
            len(_engine_manager.get_available_engines()) if _engine_manager else 0
        ),
        "timestamp": iso_format(utc_now()),
    }


@router.get("/api/diagnostics", response_model=None)
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
