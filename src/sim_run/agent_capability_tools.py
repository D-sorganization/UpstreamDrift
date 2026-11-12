"""MCP tools for Agent API counterfactual and golf-simulator capabilities (#15021)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .agent_capabilities import run_agent_capability
from .store import SimulationRunStore

ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]
AuthCheck = Callable[[Any], None]


def _envelope(payload: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(payload, sort_keys=True)}]}


def agent_capability_tool_specs() -> list[dict[str, Any]]:
    auth = {"auth_token": {"type": "string"}}
    return [
        {
            "name": "run_counterfactual_ztcf_golden",
            "description": (
                "Evaluate the pinned ZTCF golden case via sim_run (ODE backend); "
                "returns a v1 capability run record."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {**auth, "request": {"type": "object"}},
            },
        },
        {
            "name": "run_golf_simulator_golden_shot",
            "description": (
                "Simulate the pinned local golf shot via sim_run; "
                "returns a v1 capability run record with trajectory."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    **auth,
                    "request": {"type": "object"},
                    "trajectory_padding": {
                        "type": "integer",
                        "description": "Test-only trajectory repeat factor.",
                    },
                },
            },
        },
    ]


def agent_capability_tool_handlers(
    check_auth: AuthCheck,
    store: SimulationRunStore,
) -> dict[str, ToolHandler]:
    def _ztcf(arguments: dict[str, Any]) -> dict[str, Any]:
        check_auth(arguments.get("auth_token"))
        req = dict(arguments.get("request") or {})
        record = run_agent_capability("counterfactual_ztcf_golden", req, store)
        return _envelope(record)

    def _golf(arguments: dict[str, Any]) -> dict[str, Any]:
        check_auth(arguments.get("auth_token"))
        req = dict(arguments.get("request") or {})
        if arguments.get("trajectory_padding") is not None:
            req["trajectory_padding"] = arguments["trajectory_padding"]
        record = run_agent_capability("golf_simulator_golden_shot", req, store)
        return _envelope(record)

    return {
        "run_counterfactual_ztcf_golden": _ztcf,
        "run_golf_simulator_golden_shot": _golf,
    }
