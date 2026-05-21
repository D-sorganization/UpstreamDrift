"""Pure-data round-trip for the MCP servers JSON config file.

This module is the canonical writer for ``~/.upstreamdrift/mcp_servers.json``
(and any future per-app override path passed by callers). It is intentionally
Qt-free and lives in shared Tools so both UpstreamDrift and Gasification_Model
can write the same file format.

Round-trips with :mod:`src.shared.python.ai.mcp.config_loader`, which already
handles reading. The writer's documented failure mode is :class:`ValueError`
(DbC: callers handle a single exception type).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from src.shared.python.ai.mcp.contracts import McpServerConfig

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "ENV_VAR_PATTERN",
    "McpServersFile",
    "load",
    "read",
    "validate_env_placeholders",
    "write",
]


DEFAULT_CONFIG_DIR = Path.home() / ".upstreamdrift"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "mcp_servers.json"

ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_BAD_PLACEHOLDER = re.compile(r"\$\{[^}]*$|\$\{\}|\$\{[^A-Za-z_}][^}]*\}")


def validate_env_placeholders(value: str) -> str:
    """Return *value* unchanged after rejecting malformed ``${VAR}`` syntax.

    Raises:
        ValueError: If *value* contains an unterminated/empty placeholder.
    """
    if value is None:
        raise ValueError("value must be a string, not None")
    if _BAD_PLACEHOLDER.search(value):
        raise ValueError(
            f"Malformed environment-variable placeholder in MCP server env "
            f"value: {value!r}. Use the form ${{VAR_NAME}}."
        )
    return value


class McpServersFile(BaseModel):
    """Top-level JSON structure for ``mcp_servers.json``.

    Uses the Claude-Desktop-compatible layout: a top-level ``mcpServers``
    object keyed by server name. We model it as a flat list internally
    for ergonomic UI work and convert at the JSON boundary.
    """

    version: int = 1
    servers: list[McpServerConfig] = Field(default_factory=list)

    @field_validator("servers")
    @classmethod
    def _unique_names(cls, value: list[McpServerConfig]) -> list[McpServerConfig]:
        seen: set[str] = set()
        for server in value:
            if server.name in seen:
                raise ValueError(
                    f"Duplicate MCP server name {server.name!r} — names "
                    "must be unique within the file."
                )
            seen.add(server.name)
        return value


def _coerce_to_canonical_servers(
    servers: Iterable[McpServerConfig | dict[str, Any]],
) -> list[McpServerConfig]:
    result: list[McpServerConfig] = []
    for raw in servers:
        if isinstance(raw, McpServerConfig):
            result.append(raw)
        elif isinstance(raw, dict):
            result.append(McpServerConfig(**raw))
        else:
            raise TypeError(
                f"servers entries must be McpServerConfig or dict, "
                f"got {type(raw).__name__}"
            )
    return result


def _to_claude_desktop_dict(file_model: McpServersFile) -> dict[str, Any]:
    """Convert a :class:`McpServersFile` to Claude-Desktop JSON layout."""
    mcp_servers: dict[str, Any] = {}
    for srv in file_model.servers:
        entry: dict[str, Any] = {"transport": srv.transport.value}
        if srv.command is not None:
            entry["command"] = srv.command
        if srv.args:
            entry["args"] = list(srv.args)
        if srv.env:
            entry["env"] = dict(srv.env)
        if srv.url is not None:
            entry["url"] = srv.url
        if srv.timeout_seconds != 30.0:
            entry["timeoutSeconds"] = srv.timeout_seconds
        mcp_servers[srv.name] = entry
    return {"version": file_model.version, "mcpServers": mcp_servers}


def write(
    servers: Iterable[McpServerConfig | dict[str, Any]],
    *,
    path: Path | None = None,
) -> Path:
    """Write *servers* to the MCP servers JSON file (creates parent dir).

    Args:
        servers: Iterable of server entries (models or dicts).
        path: Override destination path (defaults to :data:`DEFAULT_CONFIG_PATH`).

    Returns:
        The resolved destination path.

    Raises:
        ValueError: If validation fails (duplicate names, malformed env
            placeholders, missing required fields).
        TypeError: If an entry is neither :class:`McpServerConfig` nor dict.
    """
    target = path if path is not None else DEFAULT_CONFIG_PATH
    try:
        validated = _coerce_to_canonical_servers(servers)
        file_model = McpServersFile(servers=validated)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc

    # Pre-flight env placeholder check so the writer fails loudly on bad input.
    for srv in file_model.servers:
        for raw in srv.env.values():
            validate_env_placeholders(raw)

    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_claude_desktop_dict(file_model)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return target


def read(*, path: Path | None = None) -> McpServersFile:
    """Read the MCP servers file. Returns an empty file model if missing.

    Args:
        path: Override source path.

    Returns:
        Parsed :class:`McpServersFile`. Empty when the file does not exist.

    Raises:
        ValueError: If the file is malformed JSON.
    """
    source = path if path is not None else DEFAULT_CONFIG_PATH
    if not source.exists():
        return McpServersFile()

    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Failed to read MCP servers file at {source}: {exc}") from exc

    if not isinstance(raw, dict):
        return McpServersFile()

    # Two accepted shapes: flat ``{"servers": [...]}`` and Claude-Desktop
    # ``{"mcpServers": {name: {...}}}``. Normalise to the flat form.
    if isinstance(raw.get("mcpServers"), dict):
        flat: list[dict[str, Any]] = []
        for name, entry in raw["mcpServers"].items():
            if not isinstance(entry, dict):
                continue
            flat.append({**entry, "name": name})
        raw = {"version": raw.get("version", 1), "servers": flat}

    servers_raw = raw.get("servers", [])
    if not isinstance(servers_raw, list):
        return McpServersFile()

    good: list[dict[str, Any]] = []
    for entry in servers_raw:
        if not isinstance(entry, dict):
            continue
        try:
            McpServerConfig(**entry)
            good.append(entry)
        except (ValidationError, TypeError):
            continue
    raw["servers"] = good

    try:
        return McpServersFile(**raw)
    except ValidationError:
        return McpServersFile()


load = read
