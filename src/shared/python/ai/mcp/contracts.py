"""Data contracts for the MCP client layer.

Defines lightweight, stdlib-only dataclasses for the objects that flow
between the MCP pool and the rest of the application. Deliberately avoids
importing the optional ``mcp`` transport library so that this module is
always importable.

Design notes
------------
- :class:`McpToolDescriptor` carries the information the chat layer needs
  to expose a tool to an LLM provider (name, description, input schema)
  plus the routing metadata (``server_name``, ``namespaced_name``) needed
  for :meth:`McpClientPool.call_tool` to forward the call correctly.
- :class:`McpServerConfig` is the validated in-memory form of a single
  entry from ``~/.upstreamdrift/mcp_servers.json``. It mirrors the schema
  used by :mod:`src.launchers.mcp_config_writer` so the two can round-trip
  without translation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class McpServerConfig:
    """Validated configuration for a single MCP server.

    Attributes:
        name: Unique identifier; used as the namespace prefix for tool
            names (e.g. ``"filesystem"`` → ``"filesystem__read_file"``).
        command: Executable to spawn (stdio transport) or base URL
            (http transport).
        args: Extra command-line arguments passed after *command*.
        env: Environment overrides. Values may contain ``${VAR}``
            placeholders expanded at spawn time.
        enabled: When ``False`` the server is skipped on startup.
    """

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("McpServerConfig.name must be non-empty")
        if not self.command:
            raise ValueError("McpServerConfig.command must be non-empty")


@dataclass(frozen=True)
class McpToolDescriptor:
    """Description of a single tool exposed by an MCP server.

    This is the stable data object the chat layer (and tests) work with.
    It intentionally does *not* hold a reference back to the server
    connection — call routing goes through the pool, not through this
    descriptor (LoD: the chat layer need not know which pool hosts the
    tool).

    Attributes:
        name: Tool name as returned by the MCP server (no namespace).
        namespaced_name: Globally unique name — ``"{server_name}__{name}"``.
        server_name: Name of the server that owns this tool.
        description: Human-readable description forwarded to the LLM.
        input_schema: JSON Schema dict for the tool's parameters.
    """

    name: str
    namespaced_name: str
    server_name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("McpToolDescriptor.name must be non-empty")
        if not self.namespaced_name:
            raise ValueError("McpToolDescriptor.namespaced_name must be non-empty")
        if not self.server_name:
            raise ValueError("McpToolDescriptor.server_name must be non-empty")

    def to_openai_format(self) -> dict[str, Any]:
        """Return the descriptor in OpenAI function-calling format.

        Returns:
            Dict with ``type`` and ``function`` keys.
        """
        return {
            "type": "function",
            "function": {
                "name": self.namespaced_name,
                "description": self.description,
                "parameters": self.input_schema or {"type": "object", "properties": {}},
            },
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        """Return the descriptor in Anthropic tool-use format.

        Returns:
            Dict with ``name``, ``description``, and ``input_schema`` keys.
        """
        return {
            "name": self.namespaced_name,
            "description": self.description,
            "input_schema": self.input_schema or {"type": "object", "properties": {}},
        }
