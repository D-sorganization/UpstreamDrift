"""McpChatIntegration — wires McpClientPool into the chat panel.

This module is the top-level facade that the chat panel (and tests) use
to interact with MCP servers. It:

1. Reads ``~/.upstreamdrift/mcp_servers.json`` via
   :mod:`src.launchers.mcp_config_writer`.
2. Builds an :class:`~src.shared.python.ai.mcp.pool.McpClientPool` from
   the enabled server configs (via the module-level helper
   :func:`_build_pool`).
3. Exposes :meth:`McpChatIntegration.tools` and
   :meth:`McpChatIntegration.call_tool` so the chat panel does not need
   to know about pool internals (LoD).
4. Returns a human-readable :meth:`McpChatIntegration.status` string
   ("MCP: N/M connected") for the chat panel status bar.

Usage in the chat panel::

    from src.shared.python.ai.mcp.mcp_chat_integration import McpChatIntegration

    _mcp = McpChatIntegration()
    tools = _mcp.tools()           # list[McpToolDescriptor]
    status = _mcp.status()         # "MCP: 2/3 connected"
    result = _mcp.call_tool("srv__my_tool", {"arg": "val"})

Design notes
------------
- If the config file is absent or the server list is empty, no pool is
  created and :meth:`tools` returns ``[]``.
- Per-server failures are handled inside :class:`McpClientPool` — this
  class only catches configuration-level errors.
- ``_build_pool`` is a module-level function (not a method) so tests can
  patch it with ``unittest.mock.patch``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.shared.python.ai.mcp.contracts import McpServerConfig, McpToolDescriptor
from src.shared.python.ai.mcp.pool import McpClientPool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default config path — matches mcp_config_writer.DEFAULT_CONFIG_PATH
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG_PATH = Path.home() / ".upstreamdrift" / "mcp_servers.json"


# ---------------------------------------------------------------------------
# Module-level helpers (patchable by tests)
# ---------------------------------------------------------------------------


def _load_server_configs(config_path: Path) -> list[McpServerConfig]:
    """Read enabled server configs from *config_path*.

    Falls back to the launcher's :mod:`~src.launchers.mcp_config_writer`
    for parsing. If the file is absent, returns ``[]`` silently.

    Args:
        config_path: Path to ``mcp_servers.json``.

    Returns:
        List of :class:`McpServerConfig` objects for enabled servers.
    """
    if not config_path.exists():
        logger.debug(
            "MCP config file not found at %s — no MCP servers configured",
            config_path,
        )
        return []

    try:
        from src.launchers.mcp_config_writer import read as _read_file

        file_model = _read_file(path=config_path)
        configs: list[McpServerConfig] = []
        for srv in file_model.servers:
            if not srv.enabled:
                logger.debug("Skipping disabled MCP server %r", srv.name)
                continue
            configs.append(
                McpServerConfig(
                    name=srv.name,
                    command=srv.command,
                    args=list(srv.args),
                    env=dict(srv.env),
                    enabled=srv.enabled,
                )
            )
        logger.info(
            "Loaded %d enabled MCP server config(s) from %s",
            len(configs),
            config_path,
        )
        return configs
    except Exception as exc:  # noqa: BLE001 — config errors must not crash the UI
        logger.warning(
            "Failed to load MCP server configs from %s: %s — "
            "no MCP servers will be available",
            config_path,
            exc,
        )
        return []


def _build_pool(configs: list[McpServerConfig]) -> McpClientPool:
    """Construct an :class:`McpClientPool` from *configs*.

    This is a module-level function so tests can patch it::

        with patch(
            "src.shared.python.ai.mcp.mcp_chat_integration._build_pool",
            return_value=mock_pool,
        ):
            ...

    Args:
        configs: Validated server configurations.

    Returns:
        A new :class:`McpClientPool` (not yet started).
    """
    return McpClientPool(configs)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class McpChatIntegration:
    """Facade wiring MCP servers into the UpstreamDrift chat panel.

    Attributes:
        config_path: Path to the MCP servers JSON config file.

    Preconditions (DbC):
        - *config_path* must be a :class:`pathlib.Path` or ``None``.

    Postconditions (DbC):
        - :meth:`tools` always returns a list (never ``None``).
        - :meth:`status` always returns a non-empty string.
        - :meth:`call_tool` raises :class:`RuntimeError` when no pool is
          active (config missing or no servers configured).
    """

    def __init__(
        self,
        config_path: Path | None = None,
    ) -> None:
        """Initialise and eagerly load the MCP pool.

        Args:
            config_path: Path to ``mcp_servers.json``.  Defaults to
                ``~/.upstreamdrift/mcp_servers.json``.

        Raises:
            TypeError: If *config_path* is not a :class:`Path` or ``None``.
        """
        if config_path is not None and not isinstance(config_path, Path):
            raise TypeError(
                f"config_path must be a pathlib.Path or None, "
                f"got {type(config_path).__name__}"
            )

        self._config_path: Path = (
            config_path if config_path is not None else _DEFAULT_CONFIG_PATH
        )
        self._pool: McpClientPool | None = None

        self._initialise()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialise(self) -> None:
        """Load config and build the pool (internal, called from __init__)."""
        configs = _load_server_configs(self._config_path)
        if not configs:
            logger.info("No enabled MCP servers found — pool not created")
            return

        self._pool = _build_pool(configs)
        # Eagerly start so tools() is fast on first call
        try:
            self._pool.start()
        except Exception as exc:  # noqa: BLE001 — pool startup errors must not crash UI
            logger.warning("McpClientPool.start() raised an error: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tools(self) -> list[McpToolDescriptor]:
        """Return all tools from all connected MCP servers.

        Returns:
            List of :class:`McpToolDescriptor` objects. Empty when no
            servers are configured or all have failed.

        Postcondition: return value is always a list.
        """
        if self._pool is None:
            return []
        try:
            return self._pool.tools()
        except Exception as exc:  # noqa: BLE001
            logger.warning("McpClientPool.tools() raised: %s", exc)
            return []

    def call_tool(
        self,
        namespaced_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Route a tool call to the correct MCP server.

        Args:
            namespaced_name: Globally-unique tool name
                (``"{server}__{tool}"``).
            arguments: Tool arguments dict.

        Returns:
            The result returned by the MCP server.

        Raises:
            RuntimeError: If no pool is active (no servers configured or
                config file missing).
            ValueError: If no server owns the given tool name.
        """
        if not namespaced_name:
            raise ValueError("namespaced_name must be non-empty")
        if self._pool is None:
            raise RuntimeError(
                "No MCP pool is active — either no servers are configured "
                "or the config file is missing."
            )
        return self._pool.call_tool(namespaced_name, arguments)

    def status(self) -> str:
        """Return a human-readable status string for the chat panel.

        Format: ``"MCP: {connected}/{total} connected"`` or
        ``"MCP: disabled"`` when no pool is active.

        Returns:
            Non-empty status string. Postcondition: always non-empty.
        """
        if self._pool is None:
            return "MCP: disabled"
        connected = self._pool.connected_count
        total = self._pool.server_count
        return f"MCP: {connected}/{total} connected"

    def reload(self) -> None:
        """Tear down the existing pool and reload from config.

        Call this after the user saves changes in the MCP Servers
        preferences panel.
        """
        if self._pool is not None:
            try:
                self._pool.stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning("McpClientPool.stop() raised during reload: %s", exc)
            self._pool = None

        self._initialise()
        logger.info("McpChatIntegration reloaded")
