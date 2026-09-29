"""MCP GUI widgets — status indicator and settings tab.

Provides two lightweight Qt widgets that surface MCP connection state in the
chat panel and settings dialog:

- ``McpStatusIndicator`` — a compact label showing how many MCP servers are
  connected. Updated via ``update_status(connected_count, total_count)``.
- ``McpServersTab`` — a settings tab (QWidget) where users can add/remove
  ``McpServerConfig`` entries. Backed by an in-memory list; callers are
  responsible for persisting changes (e.g. via ``config_loader``).

Design constraints:
    * No direct import of ``McpClientPool`` here — the widgets receive data
      through their public methods (LOD/DI). This keeps them testable without
      starting any real MCP servers.
    * Both classes work in headless test environments that have PyQt6 but no
      display (they just won't be shown).
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.ai.mcp.contracts import McpServerConfig
from src.shared.python.logging_pkg.logging_config import get_logger

_LOG = get_logger(__name__)


class McpStatusIndicator(QWidget):
    """Compact status label for MCP server connection health.

    Shows the number of connected servers out of the total configured. Callers
    call ``update_status`` after querying the pool; the widget does not contact
    the pool directly.

    Attributes:
        server_count: Number of currently connected MCP servers.
        status_text: Human-readable status string shown in the label.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._connected = 0
        self._total = 0

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        self._label = QLabel()
        self._label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        layout.addWidget(self._label)

        self._refresh_label()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @property
    def server_count(self) -> int:
        """Number of currently connected MCP servers."""
        return self._connected

    @property
    def status_text(self) -> str:
        """Human-readable label text."""
        return self._label.text()

    def update_status(self, *, connected_count: int, total_count: int) -> None:
        """Refresh the indicator from fresh pool data.

        Args:
            connected_count: Number of servers that are currently connected.
            total_count: Total number of configured servers.
        """
        self._connected = connected_count
        self._total = total_count
        self._refresh_label()

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _refresh_label(self) -> None:
        if self._total == 0:
            text = "MCP: disconnected"
        elif self._connected == self._total:
            text = f"MCP: {self._connected}/{self._total} connected"
        else:
            text = f"MCP: {self._connected}/{self._total} connected"
        self._label.setText(text)
        _LOG.debug("McpStatusIndicator: %s", text)


class McpServersTab(QWidget):
    """Settings tab for managing MCP server configurations.

    Users can view, add (pre-built ``McpServerConfig`` objects), and remove
    server entries. The widget maintains an in-memory list; persistence is the
    caller's responsibility.

    Attributes:
        server_count: Number of configured servers in this tab.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._configs: dict[str, McpServerConfig] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._list = QListWidget()
        layout.addWidget(self._list)

        btn_row = QHBoxLayout()
        self._remove_btn = QPushButton("Remove selected")
        self._remove_btn.clicked.connect(self._on_remove_clicked)
        btn_row.addStretch()
        btn_row.addWidget(self._remove_btn)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @property
    def server_count(self) -> int:
        """Number of configured MCP server entries."""
        return len(self._configs)

    def add_server(self, config: McpServerConfig) -> None:
        """Add a server configuration entry.

        Args:
            config: Validated ``McpServerConfig`` to add.

        Raises:
            ValueError: If a server with the same name is already configured.
        """
        if config.name in self._configs:
            raise ValueError(f"server already configured: {config.name}")
        self._configs[config.name] = config
        label = f"{config.name}  [{config.transport.value}]"
        if config.transport.value == "stdio" and config.command:
            label += f"  command={config.command}"
        elif config.url:
            label += f"  url={config.url}"
        item = QListWidgetItem(label)
        item.setData(Qt.ItemDataRole.UserRole, config.name)
        self._list.addItem(item)
        _LOG.debug("McpServersTab: added server %s", config.name)

    def remove_server(self, name: str) -> None:
        """Remove a server configuration entry by name.

        Args:
            name: Server name to remove. No-op if not found.
        """
        if name not in self._configs:
            return
        del self._configs[name]
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == name:
                self._list.takeItem(i)
                break
        _LOG.debug("McpServersTab: removed server %s", name)

    def get_configs(self) -> list[McpServerConfig]:
        """Return all configured server entries.

        Returns:
            List of ``McpServerConfig`` in insertion order.
        """
        return list(self._configs.values())

    # ------------------------------------------------------------------ #
    # Internal slots                                                        #
    # ------------------------------------------------------------------ #

    def _on_remove_clicked(self) -> None:
        selected = self._list.currentItem()
        if selected is None:
            return
        name = selected.data(Qt.ItemDataRole.UserRole)
        if name:
            self.remove_server(name)
