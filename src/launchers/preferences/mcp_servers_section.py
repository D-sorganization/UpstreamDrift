"""Launcher preferences subpage: MCP Servers — thin embed of shared widget.

Architectural rule (CLAUDE.md, AGENTS.md): if two consumers render the
same UI for shared infrastructure, the widget lives in Tools shared.
MCP is shared infrastructure (contracts, client, pool, config loader
all live under ``src/shared/python/ai/mcp/`` in Tools), so the
preferences widget lives there too — see Tools PR #2914.

This module is just a thin wrapper that:
    1. Imports the shared :class:`McpServersPrefsWidget`.
    2. Wraps it in the launcher's section convention (a small adapter
       around ``build_prefs_section``).
    3. Exposes the same external API previous local copies advertised
       so existing consumers don't need to change.

Direct consumers wanting just the widget should import the shared one
themselves — this module is for the *launcher preferences dialog* path.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWidget

__all__ = ["McpServersSection"]


class McpServersSection:
    """Launcher preferences section wrapping the shared MCP prefs widget.

    Args:
        config_path: Optional override for the JSON config path. Defaults
            to ``~/.upstreamdrift/mcp_servers.json``.
    """

    SECTION_ID = "mcp_servers"
    SECTION_LABEL = "MCP Servers"

    def __init__(self, *, config_path: Path | None = None) -> None:
        self._config_path = config_path
        self._widget: Any | None = None
        self._wrapped_widget: Any | None = None

    @property
    def widget(self) -> Any | None:
        """Return the underlying :class:`McpServersPrefsWidget`, if built."""
        return self._widget

    @property
    def servers(self) -> list[Any]:
        """Return the currently configured server list (defensive copy)."""
        if self._widget is None:
            return []
        return self._widget.servers

    def build_widget(self) -> QWidget:
        """Construct the embedded widget. Idempotent within an instance.

        Returns:
            A :class:`QWidget` wrapping the shared
            :class:`McpServersPrefsWidget`, suitable for ``QTabWidget.addTab``
            in a launcher preferences dialog.
        """
        # Imported lazily so headless callers (and importers that just
        # want to read ``servers``) don't pay the PyQt6 cost.
        from src.shared.python.ai.mcp.widgets import McpServersPrefsWidget
        from . import build_prefs_section

        if self._wrapped_widget is None:
            if self._widget is None:
                self._widget = McpServersPrefsWidget(config_path=self._config_path)
            self._wrapped_widget = build_prefs_section(
                self.SECTION_ID, self.SECTION_LABEL, [self._widget]
            )
        return self._wrapped_widget

    def add_server(self, server: Any) -> None:
        """Add a server to the configuration."""
        if self._widget is None:
            raise RuntimeError("build_widget() must be called before add_server()")

        from src.shared.python.ai.mcp.contracts import McpServerConfig as SharedConfig
        from src.launchers.mcp_config_writer import McpServerConfig as LauncherConfig

        if isinstance(server, LauncherConfig):
            server = SharedConfig(
                name=server.name,
                command=server.command,
                args=server.args,
                env=server.env,
            )
        elif not isinstance(server, SharedConfig):
            raise TypeError("server must be an McpServerConfig instance")
        self._widget.add_server(server)

    def remove_server(self, name: str) -> bool:
        """Remove a server by name from the configuration."""
        if self._widget is None:
            raise RuntimeError("build_widget() must be called before remove_server()")
        return self._widget.remove_server(name)

    def persist(self) -> Path:
        """Persist any in-memory edits. Building the widget first is required."""
        if self._widget is None:
            raise RuntimeError("build_widget() must be called before persist()")
        return self._widget.persist()
