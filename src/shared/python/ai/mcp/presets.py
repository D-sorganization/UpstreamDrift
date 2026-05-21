# ruff: noqa: E501
"""MCP server preset catalogue.

A curated registry of well-known MCP servers (GitHub, Linear, Notion, etc.)
that the UD launcher's preferences UI can present as a pick-list instead of
forcing users to hand-author ``~/.upstreamdrift/mcp_servers.json`` entries.

Each preset bundles:

- A stable, snake_case ``id``.
- User-facing ``display_name`` + 1-line ``description``.
- A ready-to-go ``McpServerConfig`` with ``${ENV_VAR}`` placeholders for
  any secrets the user still needs to supply.
- ``required_env``: env vars the UI must prompt for before launch.
- A ``category`` (``code``, ``docs``, ``productivity``, etc.) for grouping.
- Optional ``docs_url`` linking to the upstream server's documentation.

Design rules:

- **DRY**: callers consume presets via :func:`apply_preset`, never by indexing
  ``MCP_SERVER_PRESETS`` directly. This way preset cloning, env override
  validation, and command normalization stay in one place.
- **DbC**: :func:`apply_preset` validates that the preset ID exists, and
  optionally that every required env var has a concrete value (no placeholder
  left behind).
- **Reversibility**: the returned ``McpServerConfig`` is a fresh model
  instance; mutating it never reaches back into the catalogue.

The dict is exposed for read-only introspection (tests, UI catalogue
rendering); for instantiation always use :func:`apply_preset`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Final, Literal

from src.shared.python.ai.mcp.contracts import McpServerConfig, McpTransport

PresetCategory = Literal[
    "code",
    "docs",
    "productivity",
    "search",
    "memory",
    "filesystem",
    "thinking",
    "time",
]

_ENV_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True)
class McpServerPreset:
    """A pre-baked MCP server configuration ready for one-click install.

    Attributes:
        id: Stable, snake_case identifier (also used as dict key in the catalogue).
        display_name: Title-case label shown in the UI.
        description: One-line description shown in the picker.
        config: Ready-to-drop ``McpServerConfig`` with ``${VAR}`` placeholders.
        required_env: Env vars the user must supply before the server can start.
        category: Grouping label for the picker UI.
        docs_url: Link to upstream documentation (optional).
        optional_env: Env vars the user *may* supply (informational; unused values
            stay as ``${VAR}`` placeholders).
        homepage_url: Project homepage (optional).
    """

    id: str
    display_name: str
    description: str
    config: McpServerConfig
    required_env: tuple[str, ...] = field(default_factory=tuple)
    category: PresetCategory = "productivity"
    docs_url: str | None = None
    optional_env: tuple[str, ...] = field(default_factory=tuple)
    homepage_url: str | None = None


def _normalize_command_args(command: str, args: list[str]) -> tuple[str, list[str]]:
    """Normalize command+args for cross-platform stdio launches.

    npx/uvx commands are kept verbatim — the OS-level shim handles dispatch.
    Returns a (command, args) tuple with a defensive copy of ``args``.
    """
    if not command:
        raise ValueError("preset command must be non-empty")
    return command, list(args)


def _build_stdio_preset(
    *,
    id_: str,
    display_name: str,
    description: str,
    command: str,
    args: list[str],
    env: dict[str, str] | None = None,
    required_env: tuple[str, ...] = (),
    category: PresetCategory = "productivity",
    docs_url: str | None = None,
    optional_env: tuple[str, ...] = (),
    homepage_url: str | None = None,
) -> McpServerPreset:
    """DRY helper for stdio-transport presets (the common case)."""
    cmd, normalized_args = _normalize_command_args(command, args)
    config = McpServerConfig(
        name=id_,
        transport=McpTransport.STDIO,
        command=cmd,
        args=normalized_args,
        env=dict(env or {}),
    )
    return McpServerPreset(
        id=id_,
        display_name=display_name,
        description=description,
        config=config,
        required_env=required_env,
        category=category,
        docs_url=docs_url,
        optional_env=optional_env,
        homepage_url=homepage_url,
    )


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------

_GITHUB = _build_stdio_preset(
    id_="github",
    display_name="GitHub",
    description="Access GitHub repos, issues, and PRs via the official MCP server.",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-github"],
    env={"GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_PERSONAL_ACCESS_TOKEN}"},
    required_env=("GITHUB_PERSONAL_ACCESS_TOKEN",),
    category="code",
    docs_url="https://github.com/modelcontextprotocol/servers/tree/main/src/github",
    homepage_url="https://github.com",
)

_LINEAR = _build_stdio_preset(
    id_="linear",
    display_name="Linear",
    description="Query and manage Linear issues, projects, and cycles.",
    command="npx",
    args=["-y", "mcp-linear"],
    env={"LINEAR_API_KEY": "${LINEAR_API_KEY}"},
    required_env=("LINEAR_API_KEY",),
    category="productivity",
    docs_url="https://linear.app/developers/mcp",
    homepage_url="https://linear.app",
)

_NOTION = _build_stdio_preset(
    id_="notion",
    display_name="Notion",
    description="Read and write Notion pages, databases, and comments.",
    command="npx",
    args=["-y", "@notionhq/notion-mcp-server"],
    env={"NOTION_API_KEY": "${NOTION_API_KEY}"},
    required_env=("NOTION_API_KEY",),
    category="docs",
    docs_url="https://developers.notion.com/docs/mcp",
    homepage_url="https://notion.so",
)

_SLACK = _build_stdio_preset(
    id_="slack",
    display_name="Slack",
    description="Read messages, post to channels, and search Slack history.",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-slack"],
    env={
        "SLACK_BOT_TOKEN": "${SLACK_BOT_TOKEN}",
        "SLACK_TEAM_ID": "${SLACK_TEAM_ID}",
    },
    required_env=("SLACK_BOT_TOKEN", "SLACK_TEAM_ID"),
    category="productivity",
    docs_url="https://github.com/modelcontextprotocol/servers/tree/main/src/slack",
    homepage_url="https://slack.com",
)

_OBSIDIAN = _build_stdio_preset(
    id_="obsidian",
    display_name="Obsidian",
    description="Query your local Obsidian vault for notes and backlinks.",
    command="npx",
    args=["-y", "mcp-obsidian"],
    env={"OBSIDIAN_VAULT_PATH": "${OBSIDIAN_VAULT_PATH}"},
    required_env=("OBSIDIAN_VAULT_PATH",),
    category="docs",
    docs_url="https://github.com/MarkusPfundstein/mcp-obsidian",
    homepage_url="https://obsidian.md",
)

_FILESYSTEM = _build_stdio_preset(
    id_="filesystem",
    display_name="Filesystem",
    description="Sandboxed local-filesystem access to a whitelisted directory.",
    command="npx",
    args=[
        "-y",
        "@modelcontextprotocol/server-filesystem",
        str(os.path.expanduser("~/Documents")),
    ],
    category="filesystem",
    docs_url="https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem",
)

_MEMORY = _build_stdio_preset(
    id_="memory",
    display_name="Memory",
    description="Persistent in-process knowledge graph the agent can write to.",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-memory"],
    category="memory",
    docs_url="https://github.com/modelcontextprotocol/servers/tree/main/src/memory",
)

_SEQUENTIAL_THINKING = _build_stdio_preset(
    id_="sequential_thinking",
    display_name="Sequential Thinking",
    description="Structured step-by-step reasoning scaffold for the agent.",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-sequential-thinking"],
    category="thinking",
    docs_url=(
        "https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking"
    ),
)

_BRAVE_SEARCH = _build_stdio_preset(
    id_="brave_search",
    display_name="Brave Search",
    description="Web search via the Brave Search API (privacy-preserving).",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-brave-search"],
    env={"BRAVE_API_KEY": "${BRAVE_API_KEY}"},
    required_env=("BRAVE_API_KEY",),
    category="search",
    docs_url="https://github.com/modelcontextprotocol/servers/tree/main/src/brave-search",
    homepage_url="https://brave.com/search/api/",
)

_TIME = _build_stdio_preset(
    id_="time",
    display_name="Time",
    description="Timezone-aware current time + arithmetic helpers.",
    command="uvx",
    args=["mcp-server-time"],
    category="time",
    docs_url="https://github.com/modelcontextprotocol/servers/tree/main/src/time",
)

_NOTEBOOKLM = _build_stdio_preset(
    id_="notebooklm",
    display_name="NotebookLM (local shim)",
    description="In-tree Python NotebookLM shim — no install, no auth.",
    command="python",
    args=["-m", "src.shared.python.ai.mcp.notebooklm_server"],
    category="docs",
    docs_url=None,
)


MCP_SERVER_PRESETS: Final[dict[str, McpServerPreset]] = {
    preset.id: preset
    for preset in (
        _GITHUB,
        _LINEAR,
        _NOTION,
        _SLACK,
        _OBSIDIAN,
        _FILESYSTEM,
        _MEMORY,
        _SEQUENTIAL_THINKING,
        _BRAVE_SEARCH,
        _TIME,
        _NOTEBOOKLM,
    )
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_preset_ids() -> list[str]:
    """Return the sorted list of available preset IDs."""
    return sorted(MCP_SERVER_PRESETS)


def get_preset(preset_id: str) -> McpServerPreset:
    """Return the preset descriptor by ID.

    Raises:
        KeyError: if ``preset_id`` is not in the catalogue.
    """
    if preset_id not in MCP_SERVER_PRESETS:
        raise KeyError(
            f"Unknown MCP preset {preset_id!r}; "
            f"known IDs: {', '.join(list_preset_ids())}"
        )
    return MCP_SERVER_PRESETS[preset_id]


def apply_preset(
    preset_id: str,
    *,
    name: str | None = None,
    env: dict[str, str] | None = None,
    command: str | None = None,
    args: list[str] | None = None,
    validate_env: bool = False,
) -> McpServerConfig:
    """Instantiate a preset, optionally with user overrides.

    Args:
        preset_id: Key into :data:`MCP_SERVER_PRESETS`.
        name: Override the server name (defaults to ``preset_id``).
        env: User-supplied env values. Keys present here replace the
            ``${VAR}`` placeholders in the preset's config.env.
            Any keys not in the preset's env template are added verbatim.
        command: Override the command (rare; for advanced users).
        args: Override the args list (rare; for advanced users).
        validate_env: If True, raise :class:`ValueError` when any
            ``required_env`` value is still a ``${VAR}`` placeholder.

    Returns:
        A fresh :class:`McpServerConfig`. Mutations on the returned model
        do not propagate back to the catalogue.

    Raises:
        KeyError: ``preset_id`` is not in the catalogue.
        ValueError: ``validate_env=True`` and a required env var is missing.
    """
    preset = get_preset(preset_id)

    # Start from a deep-copy-equivalent: re-build the config from a dict.
    base = preset.config
    merged_env = dict(base.env)
    if env:
        for key, value in env.items():
            merged_env[key] = value  # noqa: PERF403

    final_command = command if command is not None else base.command
    final_args = list(args) if args is not None else list(base.args)

    if validate_env:
        for var in preset.required_env:
            value = merged_env.get(var, "")
            if _ENV_PLACEHOLDER.fullmatch(value or ""):
                raise ValueError(
                    f"preset {preset_id!r} requires {var!r} to be set "
                    f"(got placeholder {value!r})"
                )

    return McpServerConfig(
        name=name or preset_id,
        transport=base.transport,
        command=final_command,
        args=final_args,
        env=merged_env,
        url=base.url,
        timeout_seconds=base.timeout_seconds,
    )


__all__ = [
    "MCP_SERVER_PRESETS",
    "McpServerPreset",
    "PresetCategory",
    "apply_preset",
    "get_preset",
    "list_preset_ids",
]
