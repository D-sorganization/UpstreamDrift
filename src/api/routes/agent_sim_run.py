"""Agent API capability routes — REST surface for sim_run (#15021).

These endpoints call :mod:`src.sim_run.agent_capabilities`, the same functions
exposed as MCP tools on ``ud-sim-mcp``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Request

from src.api.middleware.error_handler import handle_api_errors
from src.api.rate_limit import get_limit, limiter
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
