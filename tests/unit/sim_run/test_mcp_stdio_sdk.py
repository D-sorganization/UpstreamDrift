"""FastMCP SDK surface parity for ``ud-sim-mcp`` (#15022)."""

from __future__ import annotations

import pytest

pytest.importorskip("mcp")

from src.sim_run.mcp import SimulationMcpServer
from src.sim_run.mcp_fastmcp import build_fastmcp_server

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fastmcp_list_tools_matches_pinned_specs() -> None:
    backend = SimulationMcpServer()
    fastmcp = build_fastmcp_server(backend)
    assert fastmcp is not None
    sdk_tools = await fastmcp.list_tools()
    pinned = {spec["name"] for spec in backend.tool_specs()}
    assert pinned == {tool.name for tool in sdk_tools}
    by_name = {tool.name: tool for tool in sdk_tools}
    for spec in backend.tool_specs():
        listed = by_name[spec["name"]]
        assert listed.description == spec["description"]
        assert listed.inputSchema == spec["inputSchema"]
