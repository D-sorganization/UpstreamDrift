"""Pinned MCP tool list and schemas for ``ud-sim-mcp`` (#15022)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.sim_run.mcp import SimulationMcpServer

pytestmark = pytest.mark.unit

_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "mcp_tool_contract_v1.json"
)


@pytest.mark.unit
def test_mcp_tool_contract_matches_pinned_fixture() -> None:
    expected = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    live = sorted(SimulationMcpServer().tool_specs(), key=lambda row: row["name"])
    assert live == expected


@pytest.mark.unit
def test_mcp_describe_engines_includes_fleet_contract_version() -> None:
    server = SimulationMcpServer()
    envelope = server.tool_describe_engines({})
    payload = json.loads(envelope["content"][0]["text"])
    assert payload["contract_version"] == "fleet-agent-api/v1"
    assert "request_schema" in payload
