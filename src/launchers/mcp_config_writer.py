"""Launcher compatibility facade for shared MCP config I/O."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.shared.python.ai.mcp import config_io as _config_io
from src.shared.python.ai.mcp.config_io import (
    DEFAULT_CONFIG_DIR,
    ENV_VAR_PATTERN,
    McpServerConfig,
    McpServersFile,
    validate_env_placeholders,
)

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "ENV_VAR_PATTERN",
    "McpServerConfig",
    "McpServersFile",
    "load",
    "read",
    "validate_env_placeholders",
    "write",
]

DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "mcp_servers.json"
expand_env = _config_io.expand_env


def write(
    servers: Iterable[McpServerConfig | dict[str, Any]],
    *,
    path: Path | None = None,
) -> Path:
    """Write MCP server configs through the shared data-layer implementation."""
    return _config_io.write(
        servers,
        path=path if path is not None else DEFAULT_CONFIG_PATH,
    )


def read(*, path: Path | None = None) -> McpServersFile:
    """Read MCP server configs through the shared data-layer implementation."""
    return _config_io.read(path=path if path is not None else DEFAULT_CONFIG_PATH)


load = read
