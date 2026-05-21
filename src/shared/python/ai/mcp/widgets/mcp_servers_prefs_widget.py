"""Shared Qt widget for MCP server preferences.

Canonical home for the MCP servers preferences UI. Both UpstreamDrift
and Gasification_Model embed this widget — neither owns a copy.

Design:
    * The widget is a proper :class:`QWidget` subclass so it can be
      embedded directly in any host preferences dialog.
    * Persistence is delegated to
      :mod:`src.shared.python.ai.mcp.config_writer` (LoD: this widget
      does not own the on-disk format).
    * Presets and Claude-Desktop import are wired through small adapter
      hooks so consumers can override them without subclassing.
    * A ``servers_changed`` signal lets the host dialog know when to
      enable a Save button.

LoD: the widget consumes :class:`McpServerConfig` and
:func:`config_writer.read` / :func:`config_writer.write` only — nothing
else.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.ai.mcp.config_writer import (
    DEFAULT_CONFIG_PATH,
    read,
    write,
)
from src.shared.python.ai.mcp.contracts import McpServerConfig, McpTransport

__all__ = [
    "McpServerEditDialog",
    "McpServersPrefsWidget",
]

_LOG = logging.getLogger(__name__)


def _discover_claude_desktop_servers() -> list[McpServerConfig]:
    """Hook: return any MCP servers configured in Claude Desktop.

    Replaced by a richer implementation in Tools issue #2901 (preset +
    import wave). Until that lands, this returns an empty list. Tests
    monkeypatch this symbol to supply fake imports.
    """
    return []


class McpServerEditDialog:
    """Modal dialog for adding or editing a single MCP server entry.

    Façade over a :class:`QDialog` so tests can drive validation logic
    without instantiating the full Qt event loop.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        initial: McpServerConfig | None = None,
    ) -> None:
        self._parent = parent
        self._initial = initial
        self._inputs: dict[str, Any] = {}

    def show(self) -> McpServerConfig | None:
        """Display the dialog modally; return the saved entry or None."""
        dialog = QDialog(self._parent)
        dialog.setWindowTitle("MCP Server")
        form = QFormLayout(dialog)

        name_edit = QLineEdit()
        command_edit = QLineEdit()
        args_edit = QLineEdit()
        env_edit = QLineEdit()
        enabled_box = QCheckBox("Enabled")
        enabled_box.setChecked(True)

        if self._initial is not None:
            name_edit.setText(self._initial.name)
            command_edit.setText(self._initial.command or "")
            args_edit.setText(" ".join(self._initial.args))
            env_edit.setText(",".join(f"{k}={v}" for k, v in self._initial.env.items()))

        form.addRow("Name:", name_edit)
        form.addRow("Command:", command_edit)
        form.addRow("Args (space-separated):", args_edit)
        form.addRow("Env (KEY=VALUE,KEY=VALUE):", env_edit)
        form.addRow(enabled_box)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        form.addRow(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        self._inputs = {
            "name": name_edit,
            "command": command_edit,
            "args": args_edit,
            "env": env_edit,
            "enabled": enabled_box,
        }

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        try:
            return self._build_entry()
        except ValueError as exc:
            QMessageBox.warning(self._parent, "Invalid MCP server entry", str(exc))
            return None

    def _build_entry(self) -> McpServerConfig:
        name = self._inputs["name"].text().strip()
        command = self._inputs["command"].text().strip()
        args = [a for a in self._inputs["args"].text().split() if a]
        env: dict[str, str] = {}
        for pair in self._inputs["env"].text().split(","):
            pair_s = pair.strip()
            if not pair_s:
                continue
            if "=" not in pair_s:
                raise ValueError(f"Env entry {pair_s!r} is missing '=' — use KEY=VALUE")
            key, _, value = pair_s.partition("=")
            env[key.strip()] = value.strip()
        return McpServerConfig(
            name=name,
            transport=McpTransport.STDIO,
            command=command,
            args=args,
            env=env,
        )


class McpServersPrefsWidget(QWidget):
    """Qt widget for managing MCP servers in a preferences pane.

    Signals:
        servers_changed: Emitted whenever the in-memory server list
            mutates (add/edit/remove/import). Hosts wire this to enable
            a "Save" button.

    Args:
        config_path: Override the on-disk JSON path (defaults to
            ``~/.upstreamdrift/mcp_servers.json``). Useful for tests.
        parent: Standard Qt parent.
    """

    servers_changed = pyqtSignal()

    def __init__(
        self,
        *,
        config_path: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._config_path = config_path or DEFAULT_CONFIG_PATH
        self._servers: list[McpServerConfig] = []

        outer = QVBoxLayout(self)
        self._table = QTableWidget(0, 4, self)
        self._table.setHorizontalHeaderLabels(["Name", "Command", "Args", "Enabled"])
        outer.addWidget(self._table)

        btn_row = QHBoxLayout()
        self._add_btn = QPushButton("Add…", self)
        self._edit_btn = QPushButton("Edit…", self)
        self._remove_btn = QPushButton("Remove", self)
        self._import_btn = QPushButton("Import from Claude Desktop", self)
        self._save_btn = QPushButton("Save", self)
        for btn in (
            self._add_btn,
            self._edit_btn,
            self._remove_btn,
            self._import_btn,
            self._save_btn,
        ):
            btn_row.addWidget(btn)
        btn_row.addStretch(1)
        outer.addLayout(btn_row)

        self._add_btn.clicked.connect(self._on_add)
        self._edit_btn.clicked.connect(self._on_edit)
        self._remove_btn.clicked.connect(self._on_remove)
        self._import_btn.clicked.connect(self.import_from_claude_desktop)
        self._save_btn.clicked.connect(self.persist)

        self._load_initial()

    # ------------------------------------------------------------------ #
    # Public read API
    # ------------------------------------------------------------------ #

    @property
    def server_count(self) -> int:
        """Number of currently-configured MCP servers."""
        return len(self._servers)

    @property
    def servers(self) -> list[McpServerConfig]:
        """A defensive copy of the configured server list."""
        return list(self._servers)

    # ------------------------------------------------------------------ #
    # Public mutation API (programmatic — used by tests + presets)
    # ------------------------------------------------------------------ #

    def add_server(self, server: McpServerConfig) -> None:
        """Append *server* to the configured list.

        Raises:
            TypeError: If *server* is not an :class:`McpServerConfig`.
            ValueError: If a server with the same name already exists.
        """
        if not isinstance(server, McpServerConfig):
            raise TypeError(
                f"server must be McpServerConfig, got {type(server).__name__}"
            )
        if any(s.name == server.name for s in self._servers):
            raise ValueError(f"server already configured: {server.name}")
        self._servers.append(server)
        self._refresh_table()
        self.servers_changed.emit()

    def remove_server(self, name: str) -> bool:
        """Remove the server named *name*. Returns True if anything changed."""
        if not isinstance(name, str):
            raise TypeError(f"name must be str, got {type(name).__name__}")
        before = len(self._servers)
        self._servers = [s for s in self._servers if s.name != name]
        changed = len(self._servers) < before
        if changed:
            self._refresh_table()
            self.servers_changed.emit()
        return changed

    def apply_preset(self, preset: McpServerConfig) -> bool:
        """Add *preset* if no server with that name exists.

        Returns:
            True if added, False if a server with that name was already
            present (preset apply is idempotent).
        """
        if not isinstance(preset, McpServerConfig):
            raise TypeError(
                f"preset must be McpServerConfig, got {type(preset).__name__}"
            )
        if any(s.name == preset.name for s in self._servers):
            return False
        self._servers.append(preset)
        self._refresh_table()
        self.servers_changed.emit()
        return True

    def import_from_claude_desktop(self) -> int:
        """Import servers configured in Claude Desktop. Returns count added."""
        discovered = _discover_claude_desktop_servers()
        added = 0
        for srv in discovered:
            if any(existing.name == srv.name for existing in self._servers):
                continue
            self._servers.append(srv)
            added += 1
        if added:
            self._refresh_table()
            self.servers_changed.emit()
        return added

    def persist(self) -> Path:
        """Write the current server list to :data:`config_path`."""
        return write(self._servers, path=self._config_path)

    def replace_all(self, servers: Iterable[McpServerConfig]) -> None:
        """Replace the entire server list (used after external edits)."""
        new_list = list(servers)
        for srv in new_list:
            if not isinstance(srv, McpServerConfig):
                raise TypeError("all entries must be McpServerConfig")
        self._servers = new_list
        self._refresh_table()
        self.servers_changed.emit()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _load_initial(self) -> None:
        try:
            loaded = read(path=self._config_path)
            self._servers = list(loaded.servers)
        except ValueError as exc:
            _LOG.warning("Failed to load %s: %s", self._config_path, exc)
            self._servers = []
        self._refresh_table()

    def _refresh_table(self) -> None:
        self._table.setRowCount(0)
        for srv in self._servers:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(srv.name))
            self._table.setItem(row, 1, QTableWidgetItem(srv.command or ""))
            self._table.setItem(row, 2, QTableWidgetItem(" ".join(srv.args)))
            self._table.setItem(row, 3, QTableWidgetItem("yes"))

    def _on_add(self) -> None:
        dialog = McpServerEditDialog(self)
        entry = dialog.show()
        if entry is None:
            return
        try:
            self.add_server(entry)
        except ValueError as exc:
            QMessageBox.warning(self, "Duplicate server", str(exc))

    def _on_edit(self) -> None:
        row = self._table.currentRow()
        if row < 0 or row >= len(self._servers):
            return
        dialog = McpServerEditDialog(self, initial=self._servers[row])
        entry = dialog.show()
        if entry is None:
            return
        self._servers[row] = entry
        self._refresh_table()
        self.servers_changed.emit()

    def _on_remove(self) -> None:
        row = self._table.currentRow()
        if row < 0 or row >= len(self._servers):
            return
        name = self._servers[row].name
        self.remove_server(name)
