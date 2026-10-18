"""Core health and root routes.

All dependencies are injected via FastAPI's Depends() mechanism.
No module-level mutable state.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, ContextManager

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from src.api.auth.dependencies import authenticate_bearer_request
from src.api.database import get_db_factory
from src.api.utils.datetime_compat import iso_format, utc_now
from src.shared.python.config.environment import (
    get_env_int,
    is_auth_disabled,
    is_production,
)
from src.shared.python.config.typed_settings import DEFAULT_SERVER_PORT
from src.shared.python.core.contracts import precondition

from .._version import __version__
from ..dependencies import get_engine_manager

if TYPE_CHECKING:
    from src.shared.python.engine_core.engine_manager import EngineManager

router = APIRouter()


async def _guard_diagnostics_access(
    request: Request,
    db_factory: Callable[[], ContextManager[Session]] = Depends(get_db_factory),
) -> None:
    """Enforce production gating and remote authentication for diagnostics (#13001).

    In production mode diagnostics are completely disabled (404).
    In remote mode (when authentication is enabled), an authenticated Bearer
    token is required (401 if missing or invalid).
    In local mode (when auth is disabled), access is permitted.
    """
    if is_production():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    if not is_auth_disabled():
        with db_factory() as db:
            await authenticate_bearer_request(request, db)


@router.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with API information."""
    return {
        "message": "Golf Modeling Suite API",
        "version": __version__,
        "docs": "/docs",
        "status": "running",
    }


@router.get("/health")
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


@router.get(
    "/api/diagnostics",
    response_model=None,
    dependencies=[Depends(_guard_diagnostics_access)],
)
@router.get(
    "/diagnostics",
    response_model=None,
    dependencies=[Depends(_guard_diagnostics_access)],
)
async def get_diagnostics(request: Request) -> dict[str, Any]:
    """Get comprehensive diagnostic information for browser mode (#13001).

    Returns real measured state for probes rather than hard-coded constants,
    and returns a relative repo_root to avoid pre-auth filesystem layout
    disclosure.
    """
    local_server_path = Path(__file__).resolve().parent.parent / "local_server.py"
    port = (
        request.url.port
        if request.url.port and request.url.port not in (80, 443)
        else get_env_int("API_PORT", default=DEFAULT_SERVER_PORT)
    )

    return {
        "backend": {
            "running": True,
            "pid": os.getpid(),
            "port": port,
            "error": None,
        },
        "python_found": bool(sys.executable and Path(sys.executable).exists()),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "repo_root": ".",
        "local_server_found": local_server_path.is_file(),
    }
