"""MCP server management preferences for Sidekick.

Provides a pure-Python data model (McpServersConfig, McpServerEntry) and a
Qt widget (McpServersSection) that hooks into the Settings dialog as a new
Preferences tab.

Config file: ``~/.upstreamdrift/mcp_servers.json``

Env-var values are stored as ``${VAR_NAME}`` placeholders — never as raw
secrets.

Issue: #5642
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

# Regex that matches a valid env-var placeholder like ${MY_VAR_123}
_PLACEHOLDER_RE = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*\}$")

_STDIO_TEMPLATE: dict[str, Any] = {
    "name": "",
    "transport": "stdio",
    "command": "",
    "args": [],
    "env": {},
    "enabled": True,
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class McpServersConfig:
    """Pure-Python config object for MCP server definitions.

    Servers are stored as plain dicts matching the JSON schema:

    .. code-block:: json

        {
            "name": "my-server",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem"],
            "env": {"HOME_DIR": "${HOME_DIR}"},
            "enabled": true
        }

    DbC postconditions:
        - ``servers`` is always a list of dicts.
        - Every env-var value must be a ``${VAR_NAME}`` placeholder.
        - Server names are unique within the list.
    """

    def __init__(self) -> None:
        self.servers: list[dict[str, Any]] = []

    # ── Class-level helpers ─────────────────────────────────────────────────

    @classmethod
    def default_path(cls) -> Path:
        """Return the canonical config file path.

        Returns:
            ``~/.upstreamdrift/mcp_servers.json``
        """
        return Path.home() / ".upstreamdrift" / "mcp_servers.json"

    @classmethod
    def load(cls, path: Path | None = None) -> McpServersConfig:
        """Load config from *path*, returning an empty config on missing file.

        Args:
            path: JSON file to read.  Defaults to :meth:`default_path`.

        Returns:
            A populated :class:`McpServersConfig` instance.
        """
        resolved = path if path is not None else cls.default_path()
        cfg = cls()
        if not resolved.exists():
            return cfg
        try:
            data = json.loads(resolved.read_text(encoding="utf-8"))
            cfg.servers = list(data.get("servers", []))
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.warning("Could not load MCP config from %s: %s", resolved, exc)
        return cfg

    # ── Instance methods ────────────────────────────────────────────────────

    def save(self, path: Path | None = None) -> None:
        """Persist config to *path* as JSON.

        Creates parent directories as needed.

        Args:
            path: Destination file.  Defaults to :meth:`default_path`.
        """
        resolved = path if path is not None else self.default_path()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        data = {"servers": self.servers}
        resolved.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def add_server(self, entry: dict[str, Any]) -> None:
        """Append a server entry after validation.

        Args:
            entry: Server definition dict.

        Raises:
            ValueError: If ``name`` is missing, a duplicate exists, or any
                env-var value is not a ``${VAR_NAME}`` placeholder.
            TypeError: If *entry* is not a dict.
        """
        if not isinstance(entry, dict):
            raise TypeError(f"entry must be a dict, got {type(entry).__name__}")
        name = entry.get("name", "").strip()
        if not name:
            raise ValueError("Server entry must have a non-empty 'name' field")
        if self._find_by_name(name) is not None:
            raise ValueError(f"Server '{name}' already exists in config")
        self._validate_env_placeholders(entry.get("env", {}))
        self.servers.append(entry)

    def disable_server(self, name: str) -> None:
        """Set ``enabled=False`` on the named server.

        Args:
            name: Server name.

        Raises:
            ValueError: If no server with that name is found.
        """
        entry = self._require_by_name(name)
        entry["enabled"] = False

    def remove_server(self, name: str) -> None:
        """Remove the named server from the list.

        Args:
            name: Server name.

        Raises:
            ValueError: If no server with that name is found.
        """
        entry = self._require_by_name(name)
        self.servers.remove(entry)

    # ── Private helpers ─────────────────────────────────────────────────────

    def _find_by_name(self, name: str) -> dict[str, Any] | None:
        for srv in self.servers:
            if srv.get("name") == name:
                return srv
        return None

    def _require_by_name(self, name: str) -> dict[str, Any]:
        entry = self._find_by_name(name)
        if entry is None:
            raise ValueError(f"Server '{name}' not found in config")
        return entry

    @staticmethod
    def _validate_env_placeholders(env: dict[str, str]) -> None:
        """Raise ValueError if any env value is not a ${VAR} placeholder."""
        for key, val in env.items():
            if val and not _PLACEHOLDER_RE.match(val):
                raise ValueError(
                    f"Env var '{key}' value must be a placeholder like "
                    f"${{VAR_NAME}}, got {val!r}. "
                    "Raw secrets must not be stored in the config file."
                )


# ---------------------------------------------------------------------------
# Qt widget — only imported when PyQt6 is available
# ---------------------------------------------------------------------------


def _build_mcp_servers_section():  # type: ignore[return]
    """Factory that returns McpServersSection (defers PyQt6 import)."""
    from PyQt6.QtCore import pyqtSignal
    from PyQt6.QtWidgets import (
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    class McpServersSection(QWidget):
        """Preferences widget for managing MCP servers.

        Shows the list of configured servers (from
        ``~/.upstreamdrift/mcp_servers.json``) with per-row enable/disable
        and remove actions.  Add-server support is provided by the
        ``add_server_requested`` signal (caller opens the appropriate dialog).

        Signals:
            config_changed: Emitted after any mutation; carries the updated
                :class:`McpServersConfig`.
            restart_required: Emitted when a change that requires restarting
                Sidekick chat is made.
        """

        config_changed = pyqtSignal(object)
        restart_required = pyqtSignal()

        def __init__(
            self,
            config: McpServersConfig | None = None,
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self._config = config if config is not None else McpServersConfig.load()
            self._setup_ui()
            self._refresh_list()

        # ── UI construction ─────────────────────────────────────────────

        def _setup_ui(self) -> None:
            layout = QVBoxLayout(self)

            group = QGroupBox("MCP Servers")
            inner = QVBoxLayout(group)

            # List of configured servers
            self._list = QListWidget()
            self._list.setMinimumHeight(160)
            inner.addWidget(self._list)

            # Action buttons
            btn_row = QHBoxLayout()

            self._btn_add = QPushButton("Add Server…")
            self._btn_add.setToolTip("Add a new MCP server configuration")
            self._btn_add.clicked.connect(self._on_add)
            btn_row.addWidget(self._btn_add)

            self._btn_disable = QPushButton("Disable")
            self._btn_disable.setToolTip(
                "Disable the selected server (skipped on startup)"
            )
            self._btn_disable.setEnabled(False)
            self._btn_disable.clicked.connect(self._on_disable)
            btn_row.addWidget(self._btn_disable)

            self._btn_remove = QPushButton("Remove")
            self._btn_remove.setToolTip("Permanently remove the selected server")
            self._btn_remove.setEnabled(False)
            self._btn_remove.clicked.connect(self._on_remove)
            btn_row.addWidget(self._btn_remove)

            btn_row.addStretch()
            inner.addLayout(btn_row)

            # Status label
            self._status = QLabel("")
            inner.addWidget(self._status)

            layout.addWidget(group)

            # Import from Claude Desktop
            import_group = QGroupBox("Import")
            import_inner = QHBoxLayout(import_group)
            self._btn_import_claude = QPushButton("Import from Claude Desktop…")
            self._btn_import_claude.setToolTip(
                "One-click import from Claude Desktop's mcp_servers config"
            )
            self._btn_import_claude.clicked.connect(self._on_import_claude_desktop)
            import_inner.addWidget(self._btn_import_claude)
            import_inner.addStretch()
            layout.addWidget(import_group)

            self._list.currentItemChanged.connect(self._on_selection_changed)

        # ── List management ─────────────────────────────────────────────

        def _refresh_list(self) -> None:
            self._list.clear()
            for srv in self._config.servers:
                label = self._format_server_label(srv)
                item = QListWidgetItem(label)
                item.setData(256, srv["name"])  # Qt.UserRole == 256
                self._list.addItem(item)
            self._update_status()

        @staticmethod
        def _format_server_label(srv: dict) -> str:
            status = "" if srv.get("enabled", True) else " [disabled]"
            transport = srv.get("transport", "stdio")
            name = srv.get("name", "<unnamed>")
            return f"{name}  ({transport}){status}"

        def _update_status(self) -> None:
            total = len(self._config.servers)
            enabled = sum(1 for s in self._config.servers if s.get("enabled", True))
            self._status.setText(f"{total} server(s), {enabled} enabled")

        # ── Slot handlers ───────────────────────────────────────────────

        def _on_selection_changed(self) -> None:
            has_selection = self._list.currentItem() is not None
            self._btn_disable.setEnabled(has_selection)
            self._btn_remove.setEnabled(has_selection)

        def _on_add(self) -> None:
            """Open add-server dialog (stdio template pre-filled)."""
            from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QTextEdit

            dlg = QDialog(self)
            dlg.setWindowTitle("Add MCP Server (JSON)")
            dlg.resize(480, 320)
            dlg_layout = QVBoxLayout(dlg)

            dlg_layout.addWidget(QLabel("Paste or edit the server JSON definition:"))
            editor = QTextEdit()
            editor.setPlainText(json.dumps(_STDIO_TEMPLATE, indent=2))
            dlg_layout.addWidget(editor)

            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok
                | QDialogButtonBox.StandardButton.Cancel
            )
            buttons.accepted.connect(dlg.accept)
            buttons.rejected.connect(dlg.reject)
            dlg_layout.addWidget(buttons)

            if dlg.exec() != QDialog.DialogCode.Accepted:
                return

            try:
                entry = json.loads(editor.toPlainText())
                self._config.add_server(entry)
                self._config.save()
                self._refresh_list()
                self.config_changed.emit(self._config)
                self.restart_required.emit()
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                from PyQt6.QtWidgets import QMessageBox

                QMessageBox.warning(self, "Invalid Server Config", str(exc))

        def _on_disable(self) -> None:
            item = self._list.currentItem()
            if item is None:
                return
            name = item.data(256)
            try:
                self._config.disable_server(name)
                self._config.save()
                self._refresh_list()
                self.config_changed.emit(self._config)
                self.restart_required.emit()
            except ValueError as exc:
                logger.warning("disable_server failed: %s", exc)

        def _on_remove(self) -> None:
            item = self._list.currentItem()
            if item is None:
                return
            name = item.data(256)
            try:
                self._config.remove_server(name)
                self._config.save()
                self._refresh_list()
                self.config_changed.emit(self._config)
                self.restart_required.emit()
            except ValueError as exc:
                logger.warning("remove_server failed: %s", exc)

        def _on_import_claude_desktop(self) -> None:
            """Import servers from Claude Desktop's config file."""
            import platform

            if platform.system() == "Windows":
                base = Path.home() / "AppData" / "Roaming" / "Claude"
            elif platform.system() == "Darwin":
                base = Path.home() / "Library" / "Application Support" / "Claude"
            else:
                base = Path.home() / ".config" / "Claude"

            claude_cfg = base / "claude_desktop_config.json"
            if not claude_cfg.exists():
                from PyQt6.QtWidgets import QMessageBox

                QMessageBox.information(
                    self,
                    "Claude Desktop Config Not Found",
                    f"Expected config at:\n{claude_cfg}\n\n"
                    "Install Claude Desktop or specify another path.",
                )
                return

            try:
                data = json.loads(claude_cfg.read_text(encoding="utf-8"))
                mcp_servers = data.get("mcpServers", {})
                imported = 0
                skipped = 0
                for name, srv_def in mcp_servers.items():
                    entry = {
                        "name": name,
                        "transport": "stdio",
                        "command": srv_def.get("command", ""),
                        "args": srv_def.get("args", []),
                        "env": {k: f"${{{k}}}" for k in srv_def.get("env", {})},
                        "enabled": True,
                    }
                    try:
                        self._config.add_server(entry)
                        imported += 1
                    except ValueError:
                        skipped += 1

                if imported:
                    self._config.save()
                    self._refresh_list()
                    self.config_changed.emit(self._config)
                    self.restart_required.emit()

                from PyQt6.QtWidgets import QMessageBox

                QMessageBox.information(
                    self,
                    "Import Complete",
                    f"Imported {imported} server(s). "
                    f"Skipped {skipped} (already present).",
                )
            except (json.JSONDecodeError, OSError) as exc:
                from PyQt6.QtWidgets import QMessageBox

                QMessageBox.warning(self, "Import Error", str(exc))

    return McpServersSection


# Lazily expose McpServersSection at module level without a hard Qt import
def __getattr__(name: str):  # noqa: N807
    if name == "McpServersSection":
        return _build_mcp_servers_section()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
