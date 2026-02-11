"""Theme Manager Dialog.

Comprehensive interface for managing themes: view, apply, create, edit,
delete, export, and import themes. Ported from Gasification Model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .custom_theme_editor import CustomThemeEditor

if TYPE_CHECKING:
    from ..theme_manager import ThemeManager

logger = logging.getLogger(__name__)


class ThemeListItem(QListWidgetItem):
    """Custom list item for themes with metadata."""

    def __init__(
        self, theme_name: str, is_builtin: bool = False, is_current: bool = False
    ) -> None:
        super().__init__()
        self.theme_name = theme_name
        self.is_builtin = is_builtin
        self.is_current = is_current
        self._update_display()

    def _update_display(self) -> None:
        display = self.theme_name
        if self.is_current:
            display += " (Current)"
        if self.is_builtin:
            display += " [Built-in]"
        self.setText(display)
        tooltip = f"Theme: {self.theme_name}\n"
        tooltip += f"Type: {'Built-in' if self.is_builtin else 'Custom'}"
        if self.is_current:
            tooltip += "\nStatus: Currently active"
        self.setToolTip(tooltip)

    def set_current(self, is_current: bool) -> None:
        if self.is_current != is_current:
            self.is_current = is_current
            self._update_display()


class ThemeManagerDialog(QDialog):
    """Dialog for comprehensive theme management."""

    theme_changed = pyqtSignal(str)

    def __init__(
        self, theme_manager: ThemeManager, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.theme_manager = theme_manager
        self.theme_items: dict[str, ThemeListItem] = {}
        self.setWindowTitle("Theme Manager")
        self.setModal(True)
        self.resize(600, 500)
        self._setup_ui()
        self._populate_themes()
        self._connect_signals()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        content = QHBoxLayout()

        # Left: theme list
        left = QWidget()
        left_layout = QVBoxLayout(left)
        themes_group = QGroupBox("Available Themes")
        themes_layout = QVBoxLayout(themes_group)
        self.theme_list = QListWidget()
        self.theme_list.setAlternatingRowColors(True)
        themes_layout.addWidget(self.theme_list)
        left_layout.addWidget(themes_group)
        content.addWidget(left, 2)

        # Right: actions
        right = QWidget()
        right_layout = QVBoxLayout(right)
        actions_group = QGroupBox("Theme Actions")
        actions_layout = QVBoxLayout(actions_group)

        self.apply_btn = QPushButton("Apply Theme")
        actions_layout.addWidget(self.apply_btn)
        actions_layout.addWidget(self._separator())

        self.create_btn = QPushButton("Create New Theme...")
        actions_layout.addWidget(self.create_btn)

        self.edit_btn = QPushButton("Edit Theme...")
        actions_layout.addWidget(self.edit_btn)

        self.duplicate_btn = QPushButton("Duplicate Theme...")
        actions_layout.addWidget(self.duplicate_btn)
        actions_layout.addWidget(self._separator())

        self.delete_btn = QPushButton("Delete Theme")
        self.delete_btn.setStyleSheet("QPushButton { color: #d32f2f; }")
        actions_layout.addWidget(self.delete_btn)
        actions_layout.addWidget(self._separator())

        self.export_btn = QPushButton("Export Theme...")
        actions_layout.addWidget(self.export_btn)

        self.import_btn = QPushButton("Import Theme...")
        actions_layout.addWidget(self.import_btn)
        actions_layout.addStretch()

        right_layout.addWidget(actions_group)

        # Current theme info
        info_group = QGroupBox("Current Theme")
        info_layout = QVBoxLayout(info_group)
        self.current_label = QLabel()
        self.current_label.setWordWrap(True)
        info_layout.addWidget(self.current_label)
        right_layout.addWidget(info_group)

        content.addWidget(right, 1)
        layout.addLayout(content)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.accept)
        layout.addWidget(button_box)

        self._update_info()

    def _separator(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(1)
        w.setStyleSheet("background-color: #555; margin: 5px 0;")
        return w

    def _connect_signals(self) -> None:
        self.theme_list.itemSelectionChanged.connect(self._on_selection_changed)
        self.theme_list.itemDoubleClicked.connect(self._on_double_click)
        self.apply_btn.clicked.connect(self._apply_selected)
        self.create_btn.clicked.connect(self._create_new)
        self.edit_btn.clicked.connect(self._edit_selected)
        self.duplicate_btn.clicked.connect(self._duplicate_selected)
        self.delete_btn.clicked.connect(self._delete_selected)
        self.export_btn.clicked.connect(self._export_selected)
        self.import_btn.clicked.connect(self._import_theme)

    def _populate_themes(self) -> None:
        self.theme_list.clear()
        self.theme_items.clear()
        current = self.theme_manager.get_current_theme_name()

        for name in self.theme_manager.get_builtin_themes():
            item = ThemeListItem(name, is_builtin=True, is_current=(name == current))
            self.theme_items[name] = item
            self.theme_list.addItem(item)

        for name in self.theme_manager.get_custom_theme_names():
            item = ThemeListItem(name, is_builtin=False, is_current=(name == current))
            self.theme_items[name] = item
            self.theme_list.addItem(item)

        if current in self.theme_items:
            self.theme_list.setCurrentItem(self.theme_items[current])
        self._on_selection_changed()

    def _update_info(self) -> None:
        current = self.theme_manager.get_current_theme_name()
        builtin = current in self.theme_manager.get_builtin_themes()
        self.current_label.setText(
            f"<b>{current}</b><br>Type: {'Built-in' if builtin else 'Custom'}"
        )

    def _on_selection_changed(self) -> None:
        items = self.theme_list.selectedItems()
        has = len(items) > 0
        is_builtin = True
        is_current = False
        if has and isinstance(items[0], ThemeListItem):
            is_builtin = items[0].is_builtin
            is_current = items[0].is_current
        self.apply_btn.setEnabled(has and not is_current)
        self.edit_btn.setEnabled(has and not is_builtin)
        self.duplicate_btn.setEnabled(has)
        self.delete_btn.setEnabled(has and not is_builtin)
        self.export_btn.setEnabled(has)

    def _on_double_click(self, item: QListWidgetItem) -> None:
        if isinstance(item, ThemeListItem) and not item.is_current:
            self._apply_theme(item.theme_name)

    def _apply_selected(self) -> None:
        items = self.theme_list.selectedItems()
        if items and isinstance(items[0], ThemeListItem):
            self._apply_theme(items[0].theme_name)

    def _apply_theme(self, name: str) -> None:
        try:
            self.theme_manager.change_theme(name)
            self.theme_changed.emit(name)
            for n, item in self.theme_items.items():
                item.set_current(n == name)
            self._update_info()
            self._on_selection_changed()
        except (RuntimeError, ValueError, OSError) as e:
            QMessageBox.critical(self, "Error", f"Failed to apply theme: {e}")

    def _create_new(self) -> None:
        editor = CustomThemeEditor(self.theme_manager, self)
        editor.theme_created.connect(self._on_theme_created)
        editor.exec()

    def _edit_selected(self) -> None:
        items = self.theme_list.selectedItems()
        if items and isinstance(items[0], ThemeListItem) and not items[0].is_builtin:
            editor = CustomThemeEditor(self.theme_manager, self, items[0].theme_name)
            editor.theme_created.connect(self._on_theme_created)
            editor.exec()

    def _duplicate_selected(self) -> None:
        items = self.theme_list.selectedItems()
        if not items or not isinstance(items[0], ThemeListItem):
            return
        theme_def = self.theme_manager.get_theme_definition(items[0].theme_name)
        if not theme_def:
            return
        editor = CustomThemeEditor(self.theme_manager, self)
        base = f"{items[0].theme_name} Copy"
        name, counter = base, 1
        while name in self.theme_manager.get_available_themes():
            name = f"{base} {counter}"
            counter += 1
        editor.name_edit.setText(name)
        editor.theme_colors = dict(theme_def)
        for key, btn in editor.color_buttons.items():
            if key in theme_def:
                btn.set_color(theme_def[key])
        editor._update_preview()
        editor.theme_created.connect(self._on_theme_created)
        editor.exec()

    def _delete_selected(self) -> None:
        items = self.theme_list.selectedItems()
        if not items or not isinstance(items[0], ThemeListItem) or items[0].is_builtin:
            return
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Delete theme '{items[0].theme_name}'?\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            if self.theme_manager.delete_custom_theme(items[0].theme_name):
                self._populate_themes()

    def _export_selected(self) -> None:
        items = self.theme_list.selectedItems()
        if not items or not isinstance(items[0], ThemeListItem):
            return
        theme_def = self.theme_manager.get_theme_definition(items[0].theme_name)
        if not theme_def:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Theme",
            f"{items[0].theme_name}.json",
            "JSON Files (*.json);;All Files (*)",
        )
        if filename:
            try:
                data = {
                    "name": items[0].theme_name,
                    "type": "builtin" if items[0].is_builtin else "custom",
                    "colors": theme_def,
                    "version": "1.0",
                }
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                QMessageBox.information(
                    self, "Exported", f"Theme exported to:\n{filename}"
                )
            except (FileNotFoundError, PermissionError, OSError) as e:
                QMessageBox.critical(self, "Error", f"Export failed: {e}")

    def _import_theme(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "Import Theme", "", "JSON Files (*.json);;All Files (*)"
        )
        if not filename:
            return
        try:
            with open(filename, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Invalid theme file format")
            name = data.get("name", Path(filename).stem)
            colors = data.get("colors", {})
            if not colors:
                raise ValueError("No color data found")

            if name in self.theme_manager.get_available_themes():
                reply = QMessageBox.question(
                    self,
                    "Name Conflict",
                    f"'{name}' already exists. Import with a different name?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    base = f"{name} Imported"
                    counter = 1
                    new_name = base
                    while new_name in self.theme_manager.get_available_themes():
                        new_name = f"{base} {counter}"
                        counter += 1
                    name = new_name
                else:
                    return

            self.theme_manager.save_custom_theme(name, colors)
            self._populate_themes()
            if name in self.theme_items:
                self.theme_list.setCurrentItem(self.theme_items[name])
            QMessageBox.information(self, "Imported", f"Theme '{name}' imported!")
        except (FileNotFoundError, PermissionError, OSError) as e:
            QMessageBox.critical(self, "Error", f"Import failed: {e}")

    def _on_theme_created(self, theme_name: str) -> None:
        self._populate_themes()
        if theme_name in self.theme_items:
            self.theme_list.setCurrentItem(self.theme_items[theme_name])
