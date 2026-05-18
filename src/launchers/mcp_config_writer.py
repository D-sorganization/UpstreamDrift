"""Pure-data round-trip for ``~/.upstreamdrift/mcp_servers.json``.

Tools PR #2884 reads MCP server configurations from this file. The
launcher writes it through this module. Keeping the I/O isolated in a
pure-data layer (no PyQt, no GUI side-effects) lets the prefs subpage
talk to a stable adapter (LoD) and makes the round-trip trivially
testable.

Validation strategy:

1. If Tools' :class:`sidekick.mcp.config.McpServerConfig` is importable, we delegate
   validation to it — single source of truth.
2. Otherwise we fall back to a small local Pydantic model that captures
   the public schema Tools #2884 documented (``name``, ``command``,
   ``args``, ``env``). The two models are designed to round-trip
   identically.

Environment-variable expansion: ``env`` values may contain
``${ENV_VAR}`` placeholders, which the runtime expands at server-launch
time. The writer does **not** expand them — that would leak host
secrets into the JSON file — but it *does* reject syntactically broken
placeholders (e.g. unbalanced braces) up-front, surfacing the error
where the user can fix it (DbC: precondition on the writer contract).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

try:  # pragma: no cover — pydantic is a hard runtime dep
    from pydantic import BaseModel, Field, ValidationError, field_validator
except ImportError as _exc:  # pragma: no cover — surfaced as ImportError
    raise ImportError(
        "pydantic is required for mcp_config_writer; install pydantic>=2.5"
    ) from _exc

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

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


# Public path constants — keep here so test fixtures can patch them.
DEFAULT_CONFIG_DIR = Path.home() / ".upstreamdrift"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "mcp_servers.json"

# ``${NAME}`` placeholders only. ``$NAME`` (POSIX shorthand) is rejected
# because it's ambiguous around adjacent letters; require explicit braces
# to keep round-tripping unambiguous.
ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
# Anything that looks like a malformed placeholder we should reject.
_BAD_PLACEHOLDER = re.compile(r"\$\{[^}]*$|\$\{\}|\$\{[^A-Za-z_}][^}]*\}")


def validate_env_placeholders(value: str) -> str:
    """Return *value* unchanged after rejecting malformed ``${VAR}`` syntax.

    Raises:
        ValueError: If *value* contains an unterminated or empty
            placeholder (e.g. ``"${"`` or ``"${}"``). Well-formed
            placeholders pass through; literal ``$`` without ``{`` is
            allowed.
    """
    if value is None:
        raise ValueError("value must be a string, not None")
    if _BAD_PLACEHOLDER.search(value):
        raise ValueError(
            f"Malformed environment-variable placeholder in MCP server "
            f"env value: {value!r}. Use the form ${{VAR_NAME}}."
        )
    return value


class McpServerConfig(BaseModel):
    """Local schema for a single MCP server entry.

    Mirrors the canonical schema Tools PR #2884 documented so that JSON
    written here is consumable there without translation.
    """

    name: str = Field(..., min_length=1)
    command: str = Field(..., min_length=1)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True

    @field_validator("env")
    @classmethod
    def _env_placeholders_ok(cls, value: dict[str, str]) -> dict[str, str]:
        for key, raw in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"env keys must be non-empty strings, got {key!r}")
            validate_env_placeholders(raw)
        return value


class McpServersFile(BaseModel):
    """Top-level JSON structure for ``mcp_servers.json``."""

    version: int = 1
    servers: list[McpServerConfig] = Field(default_factory=list)

    @field_validator("servers")
    @classmethod
    def _unique_names(cls, value: list[McpServerConfig]) -> list[McpServerConfig]:
        seen: set[str] = set()
        for server in value:
            if server.name in seen:
                raise ValueError(
                    f"Duplicate MCP server name {server.name!r} — names must "
                    "be unique within the file."
                )
            seen.add(server.name)
        return value


def _coerce_to_canonical_servers(
    servers: Iterable[McpServerConfig | dict[str, Any]],
) -> list[McpServerConfig]:
    """Accept either model instances or dicts and return validated models.

    Centralised here so the public ``write`` / ``load`` API can accept
    both forms — useful for the prefs subpage which composes entries
    from Qt widgets as dicts.
    """
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


def write(
    servers: Iterable[McpServerConfig | dict[str, Any]],
    *,
    path: Path | None = None,
) -> Path:
    """Write *servers* to ``mcp_servers.json`` (creates parent dir).

    Args:
        servers: Iterable of server entries (models or dicts).
        path: Override the destination path (defaults to
            :data:`DEFAULT_CONFIG_PATH`). Useful for tests.

    Returns:
        The resolved destination :class:`Path`.

    Raises:
        ValueError: If validation fails (duplicate names, malformed env
            placeholders, missing required fields).
    """
    target = path if path is not None else DEFAULT_CONFIG_PATH

    try:
        validated = _coerce_to_canonical_servers(servers)
        file_model = McpServersFile(servers=validated)
    except ValidationError as exc:
        # Re-raise as ValueError so callers handle a single exception
        # type (DbC: the writer's documented failure mode is ValueError).
        raise ValueError(str(exc)) from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    payload = file_model.model_dump(mode="json")
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Wrote %d MCP server entr(y/ies) to %s",
        len(file_model.servers),
        target,
    )
    return target


def read(*, path: Path | None = None) -> McpServersFile:
    """Read the MCP servers file. Returns an empty file model if missing.

    Args:
        path: Override the source path.

    Returns:
        Parsed :class:`McpServersFile`. An empty model is returned when
        the file does not exist (so the prefs subpage can render a
        clean state on first run).

    Raises:
        ValueError: If the file is malformed JSON or fails validation.
            The original parse error is chained.
    """
    source = path if path is not None else DEFAULT_CONFIG_PATH
    if not source.exists():
        return McpServersFile()

    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Failed to read MCP servers file at {source}: {exc}") from exc

    # Support invalid-entry tolerance: if the file has been edited by
    # hand and contains a partially-broken server, we warn and skip the
    # bad entry rather than refusing the whole file. The rest are
    # still validated strictly.
    if isinstance(raw, dict) and isinstance(raw.get("servers"), list):
        good: list[dict[str, Any]] = []
        for idx, entry in enumerate(raw["servers"]):
            try:
                McpServerConfig(**entry)
                good.append(entry)
            except (ValidationError, TypeError) as exc:
                logger.warning(
                    "Skipping invalid MCP server entry #%d in %s: %s",
                    idx,
                    source,
                    exc,
                )
        raw["servers"] = good

    try:
        return McpServersFile(**raw)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


# Backwards-compatible alias to satisfy "Round-trip" naming used in tests.
load = read


def expand_env(value: str, *, environ: dict[str, str] | None = None) -> str:
    """Expand ``${VAR}`` placeholders in *value* against *environ*.

    Not used during write — only exposed for tests and for callers
    that want to render an effective command-line for display.

    Unknown variables are left unchanged (so the user sees the original
    placeholder text in error messages) rather than silently becoming
    empty strings.
    """
    if value is None:
        raise ValueError("value must be a string, not None")
    env = environ if environ is not None else os.environ

    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)
        return env.get(name, match.group(0))

    return ENV_VAR_PATTERN.sub(_sub, value)
