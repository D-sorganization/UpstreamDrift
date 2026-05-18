"""MCP (Model Context Protocol) client infrastructure for UpstreamDrift.

This package provides:

- :class:`~src.shared.python.ai.mcp.contracts.McpToolDescriptor` — a
  lightweight description of a single tool exposed by an MCP server.
- :class:`~src.shared.python.ai.mcp.mcp_chat_integration.McpChatIntegration`
  — the high-level facade that loads config, manages the pool lifecycle,
  and exposes a call_tool() + tools() API to the chat panel.

The package degrades gracefully when the optional ``mcp`` transport library
is not installed: all public APIs remain importable and return empty
collections / raise :class:`RuntimeError` only when a tool call is attempted.
"""

from __future__ import annotations

from src.shared.python.ai.mcp.contracts import McpToolDescriptor
from src.shared.python.ai.mcp.mcp_chat_integration import McpChatIntegration

__all__ = [
    "McpChatIntegration",
    "McpToolDescriptor",
]
