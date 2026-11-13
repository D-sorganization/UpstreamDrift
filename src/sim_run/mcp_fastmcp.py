"""FastMCP stdio transport for ``ud-sim-mcp`` (#15022)."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent, Tool as MCPTool

from .agent_api import SERVER_NAME
from .mcp import SimulationMcpServer


class UdSimFastMcp(FastMCP):
    """FastMCP adapter that exposes the pinned hand-written tool surface."""

    def __init__(self, service: SimulationMcpServer) -> None:
        super().__init__(SERVER_NAME)
        self._service = service

    async def list_tools(self) -> list[MCPTool]:
        return [
            MCPTool(
                name=spec["name"],
                description=spec["description"],
                inputSchema=spec["inputSchema"],
            )
            for spec in self._service.tool_specs()
        ]

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> list[TextContent]:
        try:
            payload = self._service.invoke_tool(name, arguments)
        except PermissionError as exc:
            raise ToolError(str(exc)) from exc
        except (ValueError, FileNotFoundError) as exc:
            raise ToolError(str(exc)) from exc

        if payload.get("isError"):
            text = payload.get("content", [{}])[0].get("text", "tool error")
            raise ToolError(text)

        blocks = payload.get("content") or []
        if not blocks:
            return [TextContent(type="text", text="{}")]
        first = blocks[0]
        if first.get("type") != "text":
            raise ToolError("unexpected MCP content block type")
        return [TextContent(type="text", text=str(first.get("text", "")))]


def build_fastmcp_server(
    service: SimulationMcpServer | None = None,
) -> UdSimFastMcp | None:
    """Return a FastMCP server bound to a ``SimulationMcpServer`` service."""
    try:
        import mcp.server.fastmcp  # noqa: F401
    except ImportError:
        return None
    return UdSimFastMcp(service or SimulationMcpServer())
