"""GitHub MCP server integration for Sidekick.

Wires the official `@modelcontextprotocol/server-github` MCP server into
the shared :class:`McpClientPool`, so its tools (list_repos, list_issues,
list_prs, get_issue, get_pr_diff, create_issue, add_comment, search_code,
search_issues, merge_pr) become discoverable via the standard
``<server>:<tool>`` namespacing.

This package is one of two GitHub integration paths in the fleet:

* This module (Tools #2897) — MCP-based, broad surface, delegates auth
  to the MCP server subprocess.
* The GitHub CLI agent provider (Tools #2899) — ``gh``-backed, narrow
  surface, used when ``gh`` is already the auth source of truth.

Public surface:

* :func:`build_github_mcp_config` — construct a validated
  :class:`McpServerConfig` pre-wired for the official server.
* :data:`GITHUB_MCP_TOOL_DESCRIPTORS` — declarative list of the tools the
  server is expected to expose, including which are write operations.
* :func:`is_github_mcp_available` — pre-flight check (token + ``npx`` +
  package resolvable).
* :func:`register_github_mcp` — convenience helper to add the server to
  an :class:`McpClientPool`.

External callers should use :func:`register_github_mcp` rather than
poking the pool's internals (LOD).
"""

from __future__ import annotations

from src.shared.python.ai.integrations.github_mcp.discovery import (
    is_github_mcp_available,
)
from src.shared.python.ai.integrations.github_mcp.integration import (
    register_github_mcp,
)
from src.shared.python.ai.integrations.github_mcp.server_config import (
    GITHUB_MCP_PACKAGE,
    GITHUB_MCP_SERVER_NAME,
    GITHUB_MCP_TOKEN_ENV_VAR,
    build_github_mcp_config,
)
from src.shared.python.ai.integrations.github_mcp.tool_descriptors import (
    GITHUB_MCP_TOOL_DESCRIPTORS,
    GitHubMcpToolDescriptor,
    get_tool_descriptor,
    write_tool_names,
)

__all__ = [
    "GITHUB_MCP_PACKAGE",
    "GITHUB_MCP_SERVER_NAME",
    "GITHUB_MCP_TOKEN_ENV_VAR",
    "GITHUB_MCP_TOOL_DESCRIPTORS",
    "GitHubMcpToolDescriptor",
    "build_github_mcp_config",
    "get_tool_descriptor",
    "is_github_mcp_available",
    "register_github_mcp",
    "write_tool_names",
]
