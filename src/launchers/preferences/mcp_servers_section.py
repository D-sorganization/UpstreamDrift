"""Preferences subpage: MCP Servers — table editor (Tools #2884).

Renders a :class:`QTableWidget` of configured MCP servers with
``Add`` / ``Edit`` / ``Remove`` buttons. Persistence is delegated to
:mod:`src.launchers.mcp_config_writer` (LoD: this subpage does not
own the JSON file format; the writer does).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.launchers.mcp_config_writer import (
    DEFAULT_CONFIG_PATH,
    McpServerConfig,
    read,
    write,
)
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

__all__ = ["McpServersSection", "McpServerEditDialog"]


class McpServerEditDialog:
    """Modal dialog for adding/editing a single MCP server entry.

    Designed as a thin façade over a QDialog so tests can drive it
    without instantiating the full dialog when only the validation
    logic matters.
    """

    def __init__(
        self,
        parent: Any | None = None,
        initial: McpServerConfig | None = None,
    ) -> None:
        self._parent = parent
        self._initial = initial
        self._dialog: Any | None = None
        self._inputs: dict[str, Any] = {}

    def show(self) -> McpServerConfig | None:
        """Display the dialog modally; return the saved entry or None."""
        from PyQt6.QtWidgets import (
            QCheckBox,
            QDialog,
            QDialogButtonBox,
            QFormLayout,
            QLineEdit,
            QMessageBox,
        )

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
            command_edit.setText(self._initial.command)
            args_edit.setText(" ".join(self._initial.args))
            env_edit.setText(",".join(f"{k}={v}" for k, v in self._initial.env.items()))
            enabled_box.setChecked(self._initial.enabled)

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

        self._dialog = dialog
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
        """Parse widgets into a validated :class:`McpServerConfig`."""
        name = self._inputs["name"].text().strip()
        command = self._inputs["command"].text().strip()
        args = [a for a in self._inputs["args"].text().split() if a]
        env: dict[str, str] = {}
        for pair in self._inputs["env"].text().split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(f"Env entry {pair!r} is missing '=' — use KEY=VALUE")
            key, _, value = pair.partition("=")
            env[key.strip()] = value.strip()
        enabled = bool(self._inputs["enabled"].isChecked())
        return McpServerConfig(
            name=name,
            command=command,
            args=args,
            env=env,
            enabled=enabled,
        )


class McpServersSection:
    """Top-level MCP servers settings section."""

    SECTION_ID = "mcp_servers"
    SECTION_LABEL = "MCP Servers"

    def __init__(self, *, config_path: Path | None = None) -> None:
        self._config_path = config_path or DEFAULT_CONFIG_PATH
        self._table: Any | None = None
        self._servers: list[McpServerConfig] = []

    @property
    def servers(self) -> list[McpServerConfig]:
        """Return a copy of the currently-loaded server list."""
        return list(self._servers)

    def build_widget(self) -> Any:
        from PyQt6.QtWidgets import (
            QHBoxLayout,
            QPushButton,
            QTableWidget,
            QTableWidgetItem,
            QVBoxLayout,
            QWidget,
        )

        from . import build_prefs_section

        widget = QWidget()
        outer = QVBoxLayout(widget)

        table = QTableWidget(0, 4)
        table.setHorizontalHeaderLabels(["Name", "Command", "Args", "Enabled"])
        self._table = table
        outer.addWidget(table)

        # Load existing servers.
        try:
            loaded = read(path=self._config_path)
            self._servers = list(loaded.servers)
        except ValueError as exc:
            logger.warning("Failed to load %s: %s", self._config_path, exc)
            self._servers = []
        self._refresh_table(table, QTableWidgetItem)

        # Buttons row.
        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add…")
        edit_btn = QPushButton("Edit…")
        remove_btn = QPushButton("Remove")
        save_btn = QPushButton("Save")
        for btn in (add_btn, edit_btn, remove_btn, save_btn):
            btn_row.addWidget(btn)
        btn_row.addStretch(1)
        outer.addLayout(btn_row)

        add_btn.clicked.connect(lambda: self._on_add(table, QTableWidgetItem))
        edit_btn.clicked.connect(lambda: self._on_edit(table, QTableWidgetItem))
        remove_btn.clicked.connect(lambda: self._on_remove(table, QTableWidgetItem))
        save_btn.clicked.connect(self._on_save)

        return build_prefs_section(self.SECTION_ID, self.SECTION_LABEL, [widget])

    # ---- handlers (kept small, each ≤ 5 lines for DRY/orthogonality) ----

    def _refresh_table(self, table: Any, item_cls: Any) -> None:
        table.setRowCount(0)
        for srv in self._servers:
            row = table.rowCount()
            table.insertRow(row)
            table.setItem(row, 0, item_cls(srv.name))
            table.setItem(row, 1, item_cls(srv.command))
            table.setItem(row, 2, item_cls(" ".join(srv.args)))
            table.setItem(row, 3, item_cls("yes" if srv.enabled else "no"))

    def _on_add(self, table: Any, item_cls: Any) -> None:
        dialog = McpServerEditDialog(table.parent())
        entry = dialog.show()
        if entry is None:
            return
        self._servers.append(entry)
        self._refresh_table(table, item_cls)

    def _on_edit(self, table: Any, item_cls: Any) -> None:
        row = table.currentRow()
        if row < 0 or row >= len(self._servers):
            return
        dialog = McpServerEditDialog(table.parent(), initial=self._servers[row])
        entry = dialog.show()
        if entry is None:
            return
        self._servers[row] = entry
        self._refresh_table(table, item_cls)

    def _on_remove(self, table: Any, item_cls: Any) -> None:
        row = table.currentRow()
        if row < 0 or row >= len(self._servers):
            return
        del self._servers[row]
        self._refresh_table(table, item_cls)

    def _on_save(self) -> None:
        write(self._servers, path=self._config_path)

    # ---- programmatic API used by tests ----

    def add_server(self, server: McpServerConfig) -> None:
        if not isinstance(server, McpServerConfig):
            raise TypeError("server must be an McpServerConfig instance")
        self._servers.append(server)

    def remove_server(self, name: str) -> bool:
        before = len(self._servers)
        self._servers = [s for s in self._servers if s.name != name]
        return len(self._servers) < before

    def persist(self) -> Path:
        return write(self._servers, path=self._config_path)
