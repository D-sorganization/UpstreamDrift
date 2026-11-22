"""MCP tools for Agent API counterfactual and golf-simulator capabilities (#15021)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .agent_capabilities import CapabilityId, run_agent_capability
from .store import SimulationRunStore

ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]
AuthCheck = Callable[[Any], None]

#: MCP tool name -> capability id of the native viewer exporter (#15150 G8).
NATIVE_EXPORT_TOOLS: dict[str, CapabilityId] = {
    "native_export_describe": "native_export_describe",
    "native_export_validate": "native_export_validate",
}


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
        {
            "name": "native_export_describe",
            "description": (
                "Native viewer exporter request schema derived from its command "
                "line: groups with maturity and sources, options with defaults. "
                "request.min_maturity hides unready groups."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {**auth, "request": {"type": "object"}},
            },
        },
        {
            "name": "native_export_validate",
            "description": (
                "Check a native viewer export request {options: {name: value}} "
                "against the schema and allowed roots; returns the command line."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {**auth, "request": {"type": "object"}},
                "required": ["request"],
            },
        },
    ]


def _capability_handler(
    capability_id: CapabilityId, check_auth: AuthCheck, store: SimulationRunStore
) -> ToolHandler:
    def _handler(arguments: dict[str, Any]) -> dict[str, Any]:
        check_auth(arguments.get("auth_token"))
        req = dict(arguments.get("request") or {})
        return _envelope(run_agent_capability(capability_id, req, store))

    return _handler


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
        **{
            name: _capability_handler(cap, check_auth, store)
            for name, cap in NATIVE_EXPORT_TOOLS.items()
        },
    }
