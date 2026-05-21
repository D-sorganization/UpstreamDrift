"""``McpClientPool`` — owns N ``McpClient``s and aggregates their tools.

The pool is the single entry point external callers use to interact with MCP
servers. Tool names are *always* namespaced (``<server>:<tool>``) so collisions
across servers are impossible by construction.

Failure isolation:
    A server that fails ``connect()`` is logged and skipped — the rest of the
    pool stays usable. This matches Claude Desktop's behavior and avoids one
    misbehaving server taking down the whole AI surface.

DbC:
    ``tools()`` requires ``is_started``. ``add_server`` rejects duplicate names.

LOD:
    Callers must not index into ``pool._clients`` directly. The pool exposes
    ``tools()``, ``call_tool()``, ``refresh_all()``, and lifecycle methods.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from src.shared.python.ai.mcp.contracts import (
    McpResourceDescriptor,
    McpServerConfig,
    McpToolDescriptor,
)

_LOG = logging.getLogger(__name__)


class _ClientLike(Protocol):
    """The subset of ``McpClient`` the pool depends on (for fakes in tests)."""

    @property
    def is_connected(self) -> bool: ...
    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def list_tools(self) -> list[McpToolDescriptor]: ...
    async def list_resources(self) -> list[McpResourceDescriptor]: ...
    async def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]: ...


def _serialize_tool_for_provider(
    tool: McpToolDescriptor, server_name: str, provider: str = "registry"
) -> dict[str, Any]:
    """Single helper that turns an MCP tool into an aggregated, tagged dict.

    DRY: every caller (pool aggregation, tool_registry merge) goes through
    this helper so the namespacing and ``source`` tagging are defined once.
    """
    namespaced = f"{server_name}:{tool.name}"
    base = {
        "name": tool.name,
        "namespaced_name": namespaced,
        "description": tool.description,
        "input_schema": tool.input_schema,
        "source": f"mcp:{server_name}",
        "server": server_name,
    }
    if provider == "openai":
        base["openai"] = {
            "type": "function",
            "function": {
                "name": namespaced,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }
    elif provider == "anthropic":
        base["anthropic"] = {
            "name": namespaced,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
    return base


class McpClientPool:
    """Pool of MCP clients keyed by server name."""

    def __init__(self) -> None:
        self._clients: dict[str, _ClientLike] = {}
        self._configs: dict[str, McpServerConfig] = {}
        self._started = False

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def server_names(self) -> list[str]:
        return list(self._clients.keys())

    def add_server(
        self,
        config: McpServerConfig,
        client: _ClientLike | None = None,
    ) -> None:
        """Register a server. ``client`` is optional and used for tests."""
        if config.name in self._clients:
            raise ValueError(f"server already registered: {config.name}")
        if client is None:
            # Lazy import keeps the heavy transport modules out of the pool's
            # import graph during unit tests that pass fakes.
            from src.shared.python.ai.mcp.client import McpClient

            client = McpClient(config)
        self._clients[config.name] = client
        self._configs[config.name] = config

    async def remove_server(self, name: str) -> None:
        if name not in self._clients:
            return
        client = self._clients.pop(name)
        self._configs.pop(name, None)
        try:
            await client.disconnect()
        except Exception:
            _LOG.exception("error disconnecting MCP server %s", name)

    async def start_all(self) -> None:
        """Connect every registered server; isolate per-server failures."""
        for name, client in list(self._clients.items()):
            try:
                await client.connect()
            except Exception:
                _LOG.exception("failed to connect MCP server %s; skipping", name)
        self._started = True

    async def stop_all(self) -> None:
        for name, client in list(self._clients.items()):
            try:
                await client.disconnect()
            except Exception:
                _LOG.exception("error stopping MCP server %s", name)
        self._started = False

    async def tools(self, provider: str = "registry") -> list[dict[str, Any]]:
        """Aggregated, namespaced tool descriptors across all connected servers.

        Each entry carries ``source="mcp:<server>"`` so downstream consumers
        (e.g. ``ToolRegistry.refresh``) can distinguish MCP-sourced tools.
        """
        if not self._started:
            raise RuntimeError("pool not started")
        aggregated: list[dict[str, Any]] = []
        for name, client in self._clients.items():
            if not client.is_connected:
                continue
            try:
                tools = await client.list_tools()
            except Exception:
                _LOG.exception("list_tools failed for %s", name)
                continue
            for tool in tools:
                aggregated.append(_serialize_tool_for_provider(tool, name, provider))
        return aggregated

    async def resources(self) -> list[dict[str, Any]]:
        if not self._started:
            raise RuntimeError("pool not started")
        aggregated: list[dict[str, Any]] = []
        for name, client in self._clients.items():
            if not client.is_connected:
                continue
            try:
                resources = await client.list_resources()
            except Exception:
                _LOG.exception("list_resources failed for %s", name)
                continue
            for resource in resources:
                aggregated.append(
                    {
                        "uri": resource.uri,
                        "name": resource.name,
                        "description": resource.description,
                        "mime_type": resource.mime_type,
                        "source": f"mcp:{name}",
                        "server": name,
                    }
                )
        return aggregated

    async def call_tool(
        self, namespaced_name: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Call ``server:tool``. Raises if the server is unknown/disconnected."""
        if ":" not in namespaced_name:
            raise ValueError(
                "tool name must be namespaced as 'server:tool', got "
                f"{namespaced_name!r}"
            )
        server_name, tool_name = namespaced_name.split(":", 1)
        client = self._clients.get(server_name)
        if client is None:
            raise KeyError(f"unknown MCP server: {server_name}")
        if not client.is_connected:
            raise RuntimeError(f"MCP server {server_name} is not connected")
        return await client.call_tool(tool_name, args)

    async def refresh_all(self) -> list[dict[str, Any]]:
        """Re-query every connected server's tools and return the aggregated set."""
        return await self.tools()
