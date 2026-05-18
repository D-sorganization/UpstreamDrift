"""Pure-data aggregation for the Integrations Health dashboard.

This module has **no Qt dependency** and is safe to import in headless contexts
(tests, CLI scripts, etc.).  The companion Qt widget lives in
:mod:`src.launchers.integrations_health_panel`.

Issue #5643: feat(launcher): integrations health dashboard — one pane of glass
for clients, MCP, CLI, API.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Known integration catalogues
# ---------------------------------------------------------------------------

#: CLI tool names to probe with :func:`shutil.which`.
_KNOWN_CLI_TOOLS: tuple[str, ...] = ("claude", "codex", "cline", "gh")

#: (provider_name, env_var_name) pairs for API adapters.
_KNOWN_API_PROVIDERS: tuple[tuple[str, str], ...] = (
    ("anthropic", "ANTHROPIC_API_KEY"),
    ("openai", "OPENAI_API_KEY"),
    ("gemini", "GEMINI_API_KEY"),
    ("ollama", "OLLAMA_HOST"),
)

#: Default path for MCP server configuration.
_MCP_CONFIG_PATH = Path.home() / ".upstreamdrift" / "mcp_servers.json"

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

_VALID_STATUSES = frozenset(
    {"healthy", "warning", "error", "unconfigured", "unknown", "configured"}
)


@dataclass
class IntegrationRecord:
    """Describes the live health state of a single integration endpoint.

    Attributes:
        kind: Category — one of ``"mcp"``, ``"cli"``, ``"api"``, ``"client"``.
        name: Human-readable identifier (e.g. ``"claude"``, ``"openai"``).
        status: One of ``"healthy"``, ``"warning"``, ``"error"``,
            ``"unconfigured"``, ``"configured"``, ``"unknown"``.
        last_checked: UTC timestamp of the most recent probe, or ``None`` if
            the record has never been probed.
        last_error: Short error description from the most recent failed probe,
            or ``None`` when no error is recorded.
        detail: Optional supplementary detail string (must never contain
            secrets — callers are responsible for scrubbing before setting).
    """

    kind: str
    name: str
    status: str
    last_checked: datetime | None = field(default=None)
    last_error: str | None = field(default=None)
    detail: str | None = field(default=None)


# ---------------------------------------------------------------------------
# Sub-aggregators
# ---------------------------------------------------------------------------


def aggregate_cli_agents() -> list[IntegrationRecord]:
    """Probe each known CLI agent and return one :class:`IntegrationRecord` per tool.

    Uses :func:`shutil.which` to detect executables on the current ``PATH``.

    Returns:
        List of records, one per tool in :data:`_KNOWN_CLI_TOOLS`.
    """
    records: list[IntegrationRecord] = []
    now = datetime.now(tz=timezone.utc)

    for tool in _KNOWN_CLI_TOOLS:
        found = shutil.which(tool)
        if found:
            status = "healthy"
            detail = found
            last_error = None
        else:
            status = "unconfigured"
            detail = None
            last_error = f"'{tool}' not found on PATH"

        records.append(
            IntegrationRecord(
                kind="cli",
                name=tool,
                status=status,
                last_checked=now,
                last_error=last_error,
                detail=detail,
            )
        )

    return records


def aggregate_api_adapters() -> list[IntegrationRecord]:
    """Check known API adapter env vars and return one record per provider.

    A provider is considered ``"configured"`` when its primary env var is
    non-empty.  No network requests are made; only environment variable
    presence is checked.

    Returns:
        List of records, one per provider in :data:`_KNOWN_API_PROVIDERS`.
    """
    records: list[IntegrationRecord] = []
    now = datetime.now(tz=timezone.utc)

    for provider_name, env_var in _KNOWN_API_PROVIDERS:
        value = os.environ.get(env_var, "")
        if value:
            status = "configured"
            # Never store the actual key value in detail or last_error
            detail = f"{env_var} is set"
            last_error = None
        else:
            status = "unconfigured"
            detail = f"{env_var} is not set"
            last_error = f"Environment variable {env_var} is not configured"

        records.append(
            IntegrationRecord(
                kind="api",
                name=provider_name,
                status=status,
                last_checked=now,
                last_error=last_error,
                detail=detail,
            )
        )

    return records


def aggregate_mcp_servers() -> list[IntegrationRecord]:
    """Read MCP server configuration and return one record per defined server.

    Reads from :data:`_MCP_CONFIG_PATH` (``~/.upstreamdrift/mcp_servers.json``).
    If the file does not exist or cannot be parsed, returns a single record
    with ``status="unknown"`` and a description in ``last_error``.

    Returns:
        List of :class:`IntegrationRecord` instances with ``kind="mcp"``.
    """
    now = datetime.now(tz=timezone.utc)

    if not _MCP_CONFIG_PATH.exists():
        return [
            IntegrationRecord(
                kind="mcp",
                name="mcp_servers.json",
                status="unknown",
                last_checked=now,
                last_error=f"Config not found: {_MCP_CONFIG_PATH}",
            )
        ]

    try:
        raw = _MCP_CONFIG_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        return [
            IntegrationRecord(
                kind="mcp",
                name="mcp_servers.json",
                status="error",
                last_checked=now,
                last_error=str(exc),
            )
        ]

    servers: list[dict] = data if isinstance(data, list) else data.get("servers", [])
    if not servers:
        return [
            IntegrationRecord(
                kind="mcp",
                name="mcp_servers.json",
                status="warning",
                last_checked=now,
                last_error="No servers defined in config",
            )
        ]

    records: list[IntegrationRecord] = []
    for server in servers:
        name = server.get("name", "unnamed")
        # Best-effort: mark as "healthy" if we can parse the entry; a live
        # connectivity check would require network access.
        records.append(
            IntegrationRecord(
                kind="mcp",
                name=name,
                status="unknown",
                last_checked=now,
                last_error=None,
                detail="Connectivity not probed (config-only check)",
            )
        )

    return records


def collect_all() -> list[IntegrationRecord]:
    """Collect integration records from all sub-aggregators.

    Calls :func:`aggregate_cli_agents`, :func:`aggregate_api_adapters`, and
    :func:`aggregate_mcp_servers`, and returns their combined output.

    Returns:
        Combined list of :class:`IntegrationRecord` instances.
    """
    records: list[IntegrationRecord] = []
    for aggregator in (
        aggregate_cli_agents,
        aggregate_api_adapters,
        aggregate_mcp_servers,
    ):
        try:
            records.extend(aggregator())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Aggregator %s failed: %s", aggregator.__name__, exc)
    return records


# ---------------------------------------------------------------------------
# Diagnostics export
# ---------------------------------------------------------------------------

# Pattern to strip Bearer tokens and any raw key-like values from detail strings.
_SECRET_PATTERN = re.compile(r"Bearer\s+\S+", re.IGNORECASE)


def _scrub(text: str | None) -> str:
    """Remove Bearer tokens from *text*; return empty string for ``None``."""
    if text is None:
        return ""
    return _SECRET_PATTERN.sub("[REDACTED]", text)


def copy_diagnostics(records: list[IntegrationRecord]) -> str:
    """Render integration records as a Markdown report suitable for sharing.

    The output intentionally omits raw API key values and Bearer tokens so
    it can be shared in bug reports without leaking secrets.

    Args:
        records: List of :class:`IntegrationRecord` instances to summarise.

    Returns:
        Markdown string beginning with ``# Integration Health``.

    Raises:
        TypeError: If *records* is not a :class:`list`.
    """
    if not isinstance(records, list):
        raise TypeError(
            f"records must be a list of IntegrationRecord, got {type(records).__name__}"
        )

    lines: list[str] = [
        "# Integration Health",
        f"_Generated: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}_",
        "",
        "| Kind | Name | Status | Last Checked | Notes |",
        "|------|------|--------|--------------|-------|",
    ]

    for rec in records:
        checked = (
            rec.last_checked.strftime("%Y-%m-%d %H:%M:%S") if rec.last_checked else "—"
        )
        # Scrub detail; never include last_error values that might contain key hints
        notes = _scrub(rec.detail) if rec.detail else ""
        lines.append(
            f"| {rec.kind} | {rec.name} | {rec.status} | {checked} | {notes} |"
        )

    if records:
        total = len(records)
        healthy = sum(1 for r in records if r.status in ("healthy", "configured"))
        lines += [
            "",
            f"**Summary:** {healthy}/{total} integrations healthy or configured.",
        ]

    return "\n".join(lines)


__all__ = [
    "IntegrationRecord",
    "aggregate_cli_agents",
    "aggregate_api_adapters",
    "aggregate_mcp_servers",
    "collect_all",
    "copy_diagnostics",
]
