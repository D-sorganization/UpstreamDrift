"""Core health and root routes."""

from __future__ import annotations

from fastapi import APIRouter

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
        "timestamp": "2026-01-12T00:00:00Z",
    }
