"""Agent API capability routes — REST surface for sim_run (#15021).

These endpoints call :mod:`src.sim_run.agent_capabilities`, the same functions
exposed as MCP tools on ``ud-sim-mcp``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ContextManager

from fastapi import APIRouter, Body, Depends, Request
from sqlalchemy.orm import Session

from src.api.auth.dependencies import authenticate_bearer_request
from src.api.database import get_db_factory
from src.api.middleware.error_handler import handle_api_errors
from src.api.rate_limit import get_limit, limiter
from src.shared.python.config.environment import is_auth_disabled
from src.sim_run.agent_capabilities import run_agent_capability
from src.sim_run.store import SimulationRunStore

router = APIRouter(prefix="/tools/agent-sim-run", tags=["agent-sim-run"])

_STUDY_LIMIT = get_limit("API_LIMIT_AGENT_SIM_RUN", "30/minute")

_EMPTY_BODY = Body(default_factory=dict)


def _default_store() -> SimulationRunStore:
    return SimulationRunStore("./sim_results")


@router.post("/counterfactual/ztcf-golden")
@limiter.limit(_STUDY_LIMIT)
@handle_api_errors
def run_ztcf_golden(
    request: Request, payload: dict[str, Any] = _EMPTY_BODY
) -> dict[str, Any]:
    """Run the pinned ZTCF golden case; returns a v1 capability run record."""
    del request
    return run_agent_capability("counterfactual_ztcf_golden", payload, _default_store())


@router.post("/golf-simulator/golden-shot")
@limiter.limit(_STUDY_LIMIT)
@handle_api_errors
def run_golf_golden_shot(
    request: Request, payload: dict[str, Any] = _EMPTY_BODY
) -> dict[str, Any]:
    """Simulate the pinned local golf shot; returns a v1 capability run record."""
    del request
    return run_agent_capability("golf_simulator_golden_shot", payload, _default_store())


async def _require_remote_auth(
    request: Request,
    db_factory: Callable[[], ContextManager[Session]] = Depends(get_db_factory),
) -> None:
    """In remote mode (auth enabled) require a Bearer token, as #13001 does."""
    if not is_auth_disabled():
        with db_factory() as db:
            await authenticate_bearer_request(request, db)


_EXPORT_GUARD = [Depends(_require_remote_auth)]


@router.post("/native-export/describe", dependencies=_EXPORT_GUARD)
@limiter.limit(_STUDY_LIMIT)
@handle_api_errors
def describe_native_export(
    request: Request, payload: dict[str, Any] = _EMPTY_BODY
) -> dict[str, Any]:
    """Native viewer exporter request schema (#15150 G8); v1 capability record."""
    del request
    return run_agent_capability("native_export_describe", payload, _default_store())


@router.post("/native-export/validate", dependencies=_EXPORT_GUARD)
@limiter.limit(_STUDY_LIMIT)
@handle_api_errors
def validate_native_export(
    request: Request, payload: dict[str, Any] = _EMPTY_BODY
) -> dict[str, Any]:
    """Check an export request; the record carries the equivalent command line."""
    del request
    return run_agent_capability("native_export_validate", payload, _default_store())
