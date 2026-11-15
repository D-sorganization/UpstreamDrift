"""Fleet Agent API helpers for ``ud-sim-mcp`` (UpstreamDrift #15020).

Capability inventory and environment-variable naming for the stdio MCP adapter.
FastMCP stdio transport is implemented in :mod:`src.sim_run.mcp_fastmcp` (#15022).
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any, Literal

from src.shared.python.engine_core.engine_availability import (
    EngineStatus,
    get_engine_error,
    get_engine_status,
)

SERVER_NAME = "ud-sim-mcp"

SIM_RUN_ENGINES: tuple[tuple[str, str, list[str]], ...] = (
    ("pendulum", "Double pendulum dynamics benchmark", ["rigid_body", "kinematics"]),
    ("mujoco", "MuJoCo physics engine", ["rigid_body", "contacts"]),
    ("pinocchio", "Pinocchio rigid body algorithms", ["rigid_body", "dynamics"]),
    ("drake", "Drake multibody plant simulation", ["rigid_body", "contacts"]),
    ("opensim", "OpenSim musculoskeletal biomechanics", ["muscles", "biomechanics"]),
)

RuntimeStatus = Literal["available", "unavailable"]

_CANONICAL_AUTH_ENV = "UPSTREAMDRIFT_AGENT_API_TOKEN"
_AUTH_ENV_ALIASES = (_CANONICAL_AUTH_ENV, "UD_SIM_AUTH_TOKEN")

_CANONICAL_PRIVATE_ENV = "UPSTREAMDRIFT_PRIVATE_MODE"
_PRIVATE_ENV_ALIASES = (_CANONICAL_PRIVATE_ENV, "UD_PRIVATE_MODE")

_PUTTING_ANALYTICS_TOOLS = (
    "optimize_putt",
    "putting_misread_cost",
    "putting_library_study",
    "putting_variance_attribution",
)

_PUTTING_CORE_TOOLS = (
    "putting_simulate_putt",
    "putting_read_green",
    "putting_stimpmeter",
    "putting_capture_speed",
)

_AGENT_CAPABILITY_TOOLS = (
    "run_counterfactual_ztcf_golden",
    "run_golf_simulator_golden_shot",
)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def read_private_mode() -> bool:
    """Return whether Agent API Private Mode denies simulation execution."""
    return any(_env_truthy(name) for name in _PRIVATE_ENV_ALIASES)


def read_auth_token() -> str | None:
    """Return the configured auth token, preferring the canonical env var name."""
    for name in _AUTH_ENV_ALIASES:
        value = os.environ.get(name)
        if value:
            return value
    return None


def agent_api_env_docs() -> dict[str, str]:
    """Short reference for generated docs and agent guides."""
    return {
        "auth_canonical": _CANONICAL_AUTH_ENV,
        "auth_aliases": "UD_SIM_AUTH_TOKEN",
        "private_mode_canonical": _CANONICAL_PRIVATE_ENV,
        "private_mode_aliases": "UD_PRIVATE_MODE",
    }


@dataclass(frozen=True)
class CapabilityRow:
    """One engine or putting tool with a coarse runtime status."""

    id: str
    kind: Literal["engine", "putting_tool", "agent_capability"]
    description: str
    runtime_status: RuntimeStatus
    unavailable_reason: str | None = None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "description": self.description,
            "runtime_status": self.runtime_status,
        }
        if self.unavailable_reason:
            payload["unavailable_reason"] = self.unavailable_reason
        return payload


def _status_from_engine_probe(engine_type: str) -> tuple[RuntimeStatus, str | None]:
    if engine_type == "pendulum":
        return "available", None
    status = get_engine_status(engine_type)
    if status == EngineStatus.AVAILABLE:
        return "available", None
    err = get_engine_error(engine_type)
    detail = str(err).strip() if err else status.value.replace("_", " ")
    if status == EngineStatus.NOT_INSTALLED:
        reason = f"optional engine dependency not installed ({detail})"
    else:
        reason = f"engine runtime not usable ({detail})"
    return "unavailable", reason


def _probe_import(module_name: str) -> tuple[RuntimeStatus, str | None]:
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        return "unavailable", f"import failed (ImportError: {exc})"
    return "available", None


def _putting_analytics_status() -> tuple[RuntimeStatus, str | None]:
    return _probe_import("src.shared.python.putting_analytics.service")


def _putting_core_status() -> tuple[RuntimeStatus, str | None]:
    return _probe_import("src.shared.python.putting_dynamics")


def _agent_capability_status() -> tuple[RuntimeStatus, str | None]:
    for module in ("src.sim_run.counterfactual_run", "src.sim_run.golf_simulator_run"):
        status, reason = _probe_import(module)
        if status != "available":
            return status, reason
    return "available", None


def build_capability_inventory() -> dict[str, Any]:
    """Machine-readable inventory shared by MCP and generated docs."""
    engines: list[dict[str, Any]] = []
    for engine_type, description, capabilities in SIM_RUN_ENGINES:
        runtime_status, reason = _status_from_engine_probe(engine_type)
        row = CapabilityRow(
            id=engine_type,
            kind="engine",
            description=description,
            runtime_status=runtime_status,
            unavailable_reason=reason,
        )
        entry = row.to_json()
        entry["engine_type"] = engine_type
        entry["capabilities"] = list(capabilities)
        engines.append(entry)

    analytics_status, analytics_reason = _putting_analytics_status()
    core_status, core_reason = _putting_core_status()

    putting_tools: list[dict[str, Any]] = []
    for tool_id in _PUTTING_ANALYTICS_TOOLS:
        putting_tools.append(
            CapabilityRow(
                id=tool_id,
                kind="putting_tool",
                description=f"MCP putting analytics tool `{tool_id}`",
                runtime_status=analytics_status,
                unavailable_reason=analytics_reason,
            ).to_json()
        )
    for tool_id in _PUTTING_CORE_TOOLS:
        putting_tools.append(
            CapabilityRow(
                id=tool_id,
                kind="putting_tool",
                description=f"MCP putting-core tool `{tool_id}`",
                runtime_status=core_status,
                unavailable_reason=core_reason,
            ).to_json()
        )

    agent_status, agent_reason = _agent_capability_status()
    agent_capabilities: list[dict[str, Any]] = []
    for tool_id in _AGENT_CAPABILITY_TOOLS:
        agent_capabilities.append(
            CapabilityRow(
                id=tool_id,
                kind="agent_capability",
                description=f"Agent API capability `{tool_id}` via sim_run",
                runtime_status=agent_status,
                unavailable_reason=agent_reason,
            ).to_json()
        )

    return {
        "schema": "ud-sim-mcp-capability-inventory/v1",
        "server": SERVER_NAME,
        "engines": engines,
        "putting_tools": putting_tools,
        "agent_capabilities": agent_capabilities,
    }


def engines_for_describe() -> list[dict[str, Any]]:
    """Engine metadata for ``describe_engines``, including runtime status."""
    return build_capability_inventory()["engines"]
