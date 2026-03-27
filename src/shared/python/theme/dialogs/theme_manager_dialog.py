"""Theme Manager Dialog

This dialog provides a comprehensive interface for managing themes:
- View all available themes (built-in and custom)
- Apply themes
- Create new custom themes
- Edit existing custom themes
- Delete custom themes
- Export/import themes
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
    """Custom list item for themes with additional metadata."""

    def __init__(
        self, theme_name: str, is_builtin: bool = False, is_current: bool = False
    ):
        if not (theme_name is not None):
            raise ValueError("theme_name must be provided")
        if not (theme_name is not None):
            raise ValueError("theme_name must be provided")
        super().__init__()
        self.theme_name = theme_name
        self.is_builtin = is_builtin
        self.is_current = is_current

        self._update_display()

    def _update_display(self) -> None:
        """Update the display text and styling."""
        display_text = self.theme_name

        if self.is_current:
            display_text += " (Current)"

        if self.is_builtin:
            display_text += " [Built-in]"

        self.setText(display_text)

        tooltip = f"Theme: {self.theme_name}"
        if self.is_builtin:
            tooltip += "\nType: Built-in theme"
        else:
            tooltip += "\nType: Custom theme"
        if self.is_current:
            tooltip += "\nStatus: Currently active"

        self.setToolTip(tooltip)

    def set_current(self, is_current: bool) -> None:
        """Update current status."""
        if self.is_current != is_current:
            self.is_current = is_current
            self._update_display()


class ThemeManagerDialog(QDialog):
    """Dialog for comprehensive theme management."""

    theme_changed = pyqtSignal(str)  # Emits when theme is changed

    def __init__(self, theme_manager: ThemeManager, parent: QWidget | None = None):
        if not (theme_manager is not None):
            raise ValueError("theme_manager must be provided")
        if not (theme_manager is not None):
            raise ValueError("theme_manager must be provided")
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
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        content_layout = QHBoxLayout()

        # Left side - Theme list
        left_widget = self._create_left_panel()
        content_layout.addWidget(left_widget, 2)

        # Right side - Actions
        right_widget = self._create_right_panel()
        content_layout.addWidget(right_widget, 1)

        layout.addLayout(content_layout)

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.accept)
        layout.addWidget(button_box)

        self._update_current_theme_info()

    def _create_left_panel(self) -> QWidget:
        """Create the left panel containing available themes."""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        themes_group = QGroupBox("Available Themes")
        themes_layout = QVBoxLayout(themes_group)

        self.theme_list = QListWidget()
        self.theme_list.setAlternatingRowColors(True)
        themes_layout.addWidget(self.theme_list)

        left_layout.addWidget(themes_group)
        return left_widget

    def _create_right_panel(self) -> QWidget:
        """Create the right panel containing theme actions and info."""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        actions_group = QGroupBox("Theme Actions")
        actions_layout = QVBoxLayout(actions_group)

        self.apply_btn = QPushButton("Apply Theme")
        self.apply_btn.setToolTip("Apply the selected theme")
        actions_layout.addWidget(self.apply_btn)

        actions_layout.addWidget(self._create_separator())

        self.create_btn = QPushButton("Create New Theme...")
        self.create_btn.setToolTip("Create a new custom theme")
        actions_layout.addWidget(self.create_btn)

        self.edit_btn = QPushButton("Edit Theme...")
        self.edit_btn.setToolTip("Edit the selected custom theme")
        actions_layout.addWidget(self.edit_btn)

        self.duplicate_btn = QPushButton("Duplicate Theme...")
        self.duplicate_btn.setToolTip("Create a copy of the selected theme")
        actions_layout.addWidget(self.duplicate_btn)

        actions_layout.addWidget(self._create_separator())

        self.delete_btn = QPushButton("Delete Theme")
        self.delete_btn.setToolTip("Delete the selected custom theme")
        self.delete_btn.setStyleSheet("QPushButton { color: #d32f2f; }")
        actions_layout.addWidget(self.delete_btn)

        actions_layout.addWidget(self._create_separator())

        self.export_btn = QPushButton("Export Theme...")
        self.export_btn.setToolTip("Export theme to file")
        actions_layout.addWidget(self.export_btn)

        self.import_btn = QPushButton("Import Theme...")
        self.import_btn.setToolTip("Import theme from file")
        actions_layout.addWidget(self.import_btn)

        actions_layout.addStretch()
        right_layout.addWidget(actions_group)

        # Current theme info
        info_group = QGroupBox("Current Theme")
        info_layout = QVBoxLayout(info_group)

        self.current_theme_label = QLabel()
        self.current_theme_label.setWordWrap(True)
        info_layout.addWidget(self.current_theme_label)

        right_layout.addWidget(info_group)
        return right_widget

    def _create_separator(self) -> QWidget:
        """Create a visual separator."""
        separator = QWidget()
        separator.setFixedHeight(1)
        separator.setStyleSheet("background-color: #cccccc; margin: 5px 0;")
        return separator

    def _setup_ui_connections(self) -> None:
        """Set up UI signal connections."""
        self.theme_list.itemSelectionChanged.connect(self._on_selection_changed)
        self.theme_list.itemDoubleClicked.connect(self._on_item_double_clicked)

        self.apply_btn.clicked.connect(self._apply_selected_theme)
        self.create_btn.clicked.connect(self._create_new_theme)
        self.edit_btn.clicked.connect(self._edit_selected_theme)
        self.duplicate_btn.clicked.connect(self._duplicate_selected_theme)
        self.delete_btn.clicked.connect(self._delete_selected_theme)
        self.export_btn.clicked.connect(self._export_selected_theme)
        self.import_btn.clicked.connect(self._import_theme)

    def _connect_signals(self) -> None:
        """Connect all signals."""
        self._setup_ui_connections()

    def _populate_themes(self) -> None:
        """Populate the theme list."""
        self.theme_list.clear()
        self.theme_items.clear()

        current_theme = self.theme_manager.get_current_theme_name()

        for theme_name in self.theme_manager.get_builtin_themes():
            is_current = theme_name == current_theme
            item = ThemeListItem(theme_name, is_builtin=True, is_current=is_current)
            self.theme_items[theme_name] = item
            self.theme_list.addItem(item)

        for theme_name in self.theme_manager.get_custom_theme_names():
            is_current = theme_name == current_theme
            item = ThemeListItem(theme_name, is_builtin=False, is_current=is_current)
            self.theme_items[theme_name] = item
            self.theme_list.addItem(item)

        if current_theme in self.theme_items:
            self.theme_list.setCurrentItem(self.theme_items[current_theme])

        self._on_selection_changed()

    def _update_current_theme_info(self) -> None:
        """Update the current theme information display."""
        current_theme = self.theme_manager.get_current_theme_name()
        is_builtin = current_theme in self.theme_manager.get_builtin_themes()

        info_text = f"<b>{current_theme}</b><br>"
        info_text += f"Type: {'Built-in' if is_builtin else 'Custom'} theme"

        self.current_theme_label.setText(info_text)

    def _on_selection_changed(self) -> None:
        """Handle theme selection change."""
        selected_items = self.theme_list.selectedItems()
        has_selection = len(selected_items) > 0

        if has_selection:
            item = selected_items[0]
            if isinstance(item, ThemeListItem):
                is_builtin = item.is_builtin
                is_current = item.is_current
            else:
                is_builtin = True
                is_current = False
        else:
            is_builtin = True
            is_current = False

        self.apply_btn.setEnabled(has_selection and not is_current)
        self.edit_btn.setEnabled(has_selection and not is_builtin)
        self.duplicate_btn.setEnabled(has_selection)
        self.delete_btn.setEnabled(has_selection and not is_builtin)
        self.export_btn.setEnabled(has_selection)

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        """Handle double-click on theme item."""
        if isinstance(item, ThemeListItem) and not item.is_current:
            self._apply_theme(item.theme_name)

    def _apply_selected_theme(self) -> None:
        """Apply the currently selected theme."""
        selected_items = self.theme_list.selectedItems()
        if selected_items:
            item = selected_items[0]
            if isinstance(item, ThemeListItem):
                self._apply_theme(item.theme_name)

    def _apply_theme(self, theme_name: str) -> None:
        """Apply the specified theme."""
        try:
            self.theme_manager.change_theme(theme_name)
            self.theme_changed.emit(theme_name)

            for name, item in self.theme_items.items():
                item.set_current(name == theme_name)

            self._update_current_theme_info()
            self._on_selection_changed()

            logger.info(f"Applied theme: {theme_name}")

        except (OSError, ValueError, KeyError) as e:
            QMessageBox.critical(self, "Error", f"Failed to apply theme: {e}")
            logger.exception("Failed to apply theme")

    def _create_new_theme(self) -> None:
        """Open dialog to create a new theme."""
        editor = CustomThemeEditor(self.theme_manager, self)
        editor.theme_created.connect(self._on_theme_created)
        editor.exec()

    def _edit_selected_theme(self) -> None:
        """Edit the selected custom theme."""
        selected_items = self.theme_list.selectedItems()
        if not selected_items:
            return

        item = selected_items[0]
        if isinstance(item, ThemeListItem) and not item.is_builtin:
            editor = CustomThemeEditor(self.theme_manager, self, item.theme_name)
            editor.theme_created.connect(self._on_theme_updated)
            editor.exec()

    def _duplicate_selected_theme(self) -> None:
        """Create a duplicate of the selected theme."""
        selected_items = self.theme_list.selectedItems()
        if not selected_items:
            return

        item = selected_items[0]
        if isinstance(item, ThemeListItem):
            theme_def = self.theme_manager.get_theme_definition(item.theme_name)
            if not theme_def:
                QMessageBox.warning(self, "Error", "Could not load theme definition.")
                return

            editor = CustomThemeEditor(self.theme_manager, self)

            base_name = f"{item.theme_name} Copy"
            counter = 1
            new_name = base_name
            while new_name in self.theme_manager.get_available_themes():
                new_name = f"{base_name} {counter}"
                counter += 1

            editor.name_edit.setText(new_name)
            editor.theme_colors = dict(theme_def)

            for key, button in editor.color_buttons.items():
                if key in theme_def:
                    button.set_color(theme_def[key])

            editor._update_preview()
            editor.theme_created.connect(self._on_theme_created)
            editor.exec()

    def _delete_selected_theme(self) -> None:
        """Delete the selected custom theme."""
        selected_items = self.theme_list.selectedItems()
        if not selected_items:
            return

        item = selected_items[0]
        if isinstance(item, ThemeListItem) and not item.is_builtin:
            reply = QMessageBox.question(
                self,
                "Confirm Delete",
                f"Are you sure you want to delete the theme '{item.theme_name}'?\n\n"
                "This action cannot be undone.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                try:
                    success = self.theme_manager.delete_custom_theme(item.theme_name)
                    if success:
                        self._populate_themes()
                        logger.info(f"Deleted theme: {item.theme_name}")
                    else:
                        QMessageBox.warning(self, "Error", "Failed to delete theme.")
                except (OSError, ValueError, KeyError) as e:
                    QMessageBox.critical(self, "Error", f"Failed to delete theme: {e}")
                    logger.exception("Failed to delete theme")

    def _export_selected_theme(self) -> None:
        """Export the selected theme to a file."""
        selected_items = self.theme_list.selectedItems()
        if not selected_items:
            return

        item = selected_items[0]
        if isinstance(item, ThemeListItem):
            theme_def = self.theme_manager.get_theme_definition(item.theme_name)
            if not theme_def:
                QMessageBox.warning(self, "Error", "Could not load theme definition.")
                return

            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Export Theme",
                f"{item.theme_name}.json",
                "JSON Files (*.json);;All Files (*)",
            )

            if filename:
                try:
                    export_data = {
                        "name": item.theme_name,
                        "type": "builtin" if item.is_builtin else "custom",
                        "colors": theme_def,
                        "exported_by": "D-sorganization Theme Manager",
                        "version": "1.0",
                    }

                    with open(filename, "w", encoding="utf-8") as f:
                        json.dump(export_data, f, indent=2)

                    QMessageBox.information(
                        self,
                        "Export Successful",
                        f"Theme '{item.theme_name}' exported to:\n{filename}",
                    )
                    logger.info(f"Exported theme {item.theme_name} to {filename}")

                except (PermissionError, OSError) as e:
                    QMessageBox.critical(
                        self, "Export Error", f"Failed to export theme: {e}"
                    )
                    logger.exception("Failed to export theme")

    def _import_theme(self) -> None:
        """Import a theme from a file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Import Theme", "", "JSON Files (*.json);;All Files (*)"
        )

        if filename:
            try:
                with open(filename, encoding="utf-8") as f:
                    import_data = json.load(f)

                if not isinstance(import_data, dict):
                    raise ValueError("Invalid theme file format")

                theme_name = import_data.get("name", Path(filename).stem)
                colors = import_data.get("colors", {})

                if not colors:
                    raise ValueError("No color data found in theme file")

                if theme_name in self.theme_manager.get_available_themes():
                    reply = QMessageBox.question(
                        self,
                        "Name Conflict",
                        f"A theme named '{theme_name}' already exists.\n\n"
                        "Do you want to import with a different name?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    )

                    if reply == QMessageBox.StandardButton.Yes:
                        base_name = f"{theme_name} Imported"
                        counter = 1
                        new_name = base_name
                        while new_name in self.theme_manager.get_available_themes():
                            new_name = f"{base_name} {counter}"
                            counter += 1
                        theme_name = new_name
                    else:
                        return

                self.theme_manager.save_custom_theme(theme_name, colors)
                self._populate_themes()

                if theme_name in self.theme_items:
                    self.theme_list.setCurrentItem(self.theme_items[theme_name])

                QMessageBox.information(
                    self,
                    "Import Successful",
                    f"Theme '{theme_name}' imported successfully!",
                )
                logger.info(f"Imported theme {theme_name} from {filename}")

            except (PermissionError, OSError) as e:
                QMessageBox.critical(
                    self, "Import Error", f"Failed to import theme: {e}"
                )
                logger.exception("Failed to import theme")

    def _on_theme_created(self, theme_name: str) -> None:
        """Handle new theme creation."""
        if not (theme_name is not None):
            raise ValueError("theme_name must be provided")
        if not (theme_name is not None):
            raise ValueError("theme_name must be provided")
        self._populate_themes()

        if theme_name in self.theme_items:
            self.theme_list.setCurrentItem(self.theme_items[theme_name])

    def _on_theme_updated(self, theme_name: str) -> None:
        """Handle theme update."""
        if not (theme_name is not None):
            raise ValueError("theme_name must be provided")
        if not (theme_name is not None):
            raise ValueError("theme_name must be provided")
        self._populate_themes()
        self._update_current_theme_info()

        if theme_name in self.theme_items:
            self.theme_list.setCurrentItem(self.theme_items[theme_name])
