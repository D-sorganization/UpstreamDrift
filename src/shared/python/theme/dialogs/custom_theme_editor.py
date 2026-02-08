"""Custom Theme Editor Dialog.

Allows users to create and edit custom themes for the application.
Users can modify colors, preview the theme in real-time, and save their
custom themes. Ported from the Gasification Model shared utilities.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from ..theme_manager import ThemeManager

logger = logging.getLogger(__name__)


class ColorPickerButton(QPushButton):
    """A button that displays a color and opens a color picker when clicked."""

    color_changed = pyqtSignal(str)  # Emits hex color string

    def __init__(self, initial_color: str = "#ffffff", parent: QWidget | None = None):
        super().__init__(parent)
        self._color = initial_color
        self.setFixedSize(60, 30)
        self.clicked.connect(self._open_color_picker)
        self._update_button_style()

    def get_color(self) -> str:
        """Get the current color as hex string."""
        return self._color

    def set_color(self, color: str) -> None:
        """Set the color from hex string."""
        if color != self._color:
            self._color = color
            self._update_button_style()
            self.color_changed.emit(color)

    def _update_button_style(self) -> None:
        """Update button appearance to show the current color."""
        qcolor = QColor(self._color)
        text_color = "#ffffff" if qcolor.lightness() < 128 else "#000000"
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {self._color};
                color: {text_color};
                border: 2px solid #666666;
                border-radius: 4px;
                font-weight: bold;
                font-size: 10px;
            }}
            QPushButton:hover {{
                border: 2px solid #333333;
            }}
        """)
        self.setText(self._color.upper())

    def _open_color_picker(self) -> None:
        """Open color picker dialog."""
        initial_color = QColor(self._color)
        color = QColorDialog.getColor(initial_color, self, "Select Color")
        if color.isValid():
            self.set_color(color.name())


class ThemePreviewWidget(QWidget):
    """Widget that shows a preview of how the theme will look."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        title = QLabel("Theme Preview")
        title.setStyleSheet("font-size: 14px; font-weight: bold; margin: 5px;")
        layout.addWidget(title)

        group = QGroupBox("Sample Group")
        group_layout = QFormLayout(group)
        group_layout.addRow("Input:", QLineEdit("Sample text input"))
        group_layout.addRow("Action:", QPushButton("Sample Button"))
        group_layout.addRow("Info:", QLabel("Sample label text"))
        layout.addWidget(group)
        layout.addStretch()

    def apply_theme_colors(self, colors: dict[str, str]) -> None:
        """Apply theme colors to preview elements."""
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {colors.get("bg", "#1A1A1A")};
                color: {colors.get("text", "#FFFFFF")};
            }}
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {colors.get("border", "#404040")};
                border-radius: 5px; margin: 5px; padding-top: 10px;
                background-color: {colors.get("group_bg", "#242424")};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin; left: 10px; padding: 0 5px;
            }}
            QLineEdit {{
                padding: 5px 8px;
                border: 1px solid {colors.get("border", "#404040")};
                border-radius: 3px;
                background-color: {colors.get("input_bg", "#242424")};
                color: {colors.get("text", "#FFFFFF")};
            }}
            QPushButton {{
                padding: 6px 14px;
                border: 1px solid {colors.get("border", "#404040")};
                border-radius: 4px;
                background-color: {colors.get("group_bg", "#242424")};
                color: {colors.get("text", "#FFFFFF")};
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {colors.get("accent", "#0A84FF")};
                color: white;
            }}
            QLabel {{
                color: {colors.get("text_secondary", "#E0E0E0")};
                background: transparent;
            }}
        """)


class CustomThemeEditor(QDialog):
    """Dialog for creating and editing custom themes."""

    theme_created = pyqtSignal(str)  # Emits theme name when created

    def __init__(
        self,
        theme_manager: ThemeManager,
        parent: QWidget | None = None,
        edit_theme: str | None = None,
    ):
        super().__init__(parent)
        self.theme_manager = theme_manager
        self.edit_theme = edit_theme
        self.color_buttons: dict[str, ColorPickerButton] = {}
        self.theme_colors: dict[str, str] = {}
        self.save_btn: QPushButton | None = None
        self.save_apply_btn: QPushButton | None = None

        self.setWindowTitle(
            "Custom Theme Editor" if not edit_theme else f"Edit Theme: {edit_theme}"
        )
        self.setModal(True)
        self.resize(800, 600)
        self._setup_ui()
        self._load_initial_colors()
        self._connect_signals()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout()

        # Left side - Color editor
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Theme name
        name_group = QGroupBox("Theme Name")
        name_layout = QFormLayout(name_group)
        self.name_edit = QLineEdit()
        if self.edit_theme:
            self.name_edit.setText(self.edit_theme)
            self.name_edit.setEnabled(False)
        else:
            self.name_edit.setPlaceholderText("Enter theme name...")
        name_layout.addRow("Name:", self.name_edit)
        left_layout.addWidget(name_group)

        # Color editor
        colors_group = QGroupBox("Theme Colors")
        colors_scroll = QScrollArea()
        colors_widget = QWidget()
        colors_layout = QFormLayout(colors_widget)

        color_definitions = [
            ("bg", "Background", "Main window background color"),
            ("group_bg", "Group Background", "Background for group boxes and panels"),
            ("border", "Border", "Border color for controls"),
            ("text", "Primary Text", "Main text color"),
            ("text_secondary", "Secondary Text", "Secondary text and labels"),
            ("label", "Label Text", "Form labels and hints"),
            ("focus", "Focus Highlight", "Color when controls are focused"),
            ("input_bg", "Input Background", "Background for text inputs"),
            ("accent", "Accent Color", "Primary accent color for highlights"),
            ("title_bg", "Title Background", "Background for titles and headers"),
            ("title_border", "Title Border", "Border for title areas"),
            ("table_header", "Table Header", "Table header background"),
            ("table_alt", "Table Alternate", "Alternate table row color"),
            ("button_hover", "Button Hover", "Button color when hovered"),
        ]

        for key, name, tooltip in color_definitions:
            btn = ColorPickerButton()
            btn.setToolTip(tooltip)
            btn.color_changed.connect(partial(self._on_color_changed, key))
            self.color_buttons[key] = btn
            colors_layout.addRow(f"{name}:", btn)

        colors_scroll.setWidget(colors_widget)
        colors_scroll.setWidgetResizable(True)
        colors_scroll.setMaximumHeight(400)
        colors_group_layout = QVBoxLayout(colors_group)
        colors_group_layout.addWidget(colors_scroll)
        left_layout.addWidget(colors_group)

        # Quick start presets
        preset_group = QGroupBox("Quick Start")
        preset_layout = QVBoxLayout(preset_group)
        preset_layout.addWidget(QLabel("Start from an existing theme and modify it:"))
        preset_btn_layout = QHBoxLayout()
        for theme_name in self.theme_manager.get_builtin_themes()[:6]:
            btn = QPushButton(theme_name)
            btn.clicked.connect(
                lambda checked, n=theme_name: self._load_preset_theme(n)
            )
            preset_btn_layout.addWidget(btn)
        preset_layout.addLayout(preset_btn_layout)
        left_layout.addWidget(preset_group)

        left_layout.addStretch()
        layout.addWidget(left_widget, 1)

        # Right side - Preview
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(
            QLabel("Live Preview", styleSheet="font-size: 14px; font-weight: bold;")
        )
        self.preview_widget = ThemePreviewWidget()
        right_layout.addWidget(self.preview_widget, 1)
        layout.addWidget(right_widget, 1)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        self.save_btn = button_box.button(QDialogButtonBox.StandardButton.Save)
        self.save_apply_btn = button_box.addButton(
            "Save && Apply", QDialogButtonBox.ButtonRole.AcceptRole
        )
        button_box.accepted.connect(self._save_theme)
        button_box.rejected.connect(self.reject)
        if self.save_apply_btn:
            self.save_apply_btn.clicked.connect(self._save_and_apply_theme)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(layout)
        main_layout.addWidget(button_box)

    def _load_initial_colors(self) -> None:
        if self.edit_theme:
            theme_def = self.theme_manager.get_theme_definition(self.edit_theme)
            if theme_def:
                self.theme_colors = dict(theme_def)
        else:
            current = self.theme_manager.get_current_theme_name()
            theme_def = self.theme_manager.get_theme_definition(current)
            if theme_def:
                self.theme_colors = dict(theme_def)

        for key, button in self.color_buttons.items():
            if key in self.theme_colors:
                button.set_color(self.theme_colors[key])
        self._update_preview()

    def _load_preset_theme(self, theme_name: str) -> None:
        theme_def = self.theme_manager.get_theme_definition(theme_name)
        if theme_def:
            self.theme_colors = dict(theme_def)
            for key, button in self.color_buttons.items():
                if key in self.theme_colors:
                    button.set_color(self.theme_colors[key])
            self._update_preview()

    def _connect_signals(self) -> None:
        self.name_edit.textChanged.connect(self._validate_input)

    def _on_color_changed(self, key: str, color: str) -> None:
        self.theme_colors[key] = color
        self._update_preview()

    def _update_preview(self) -> None:
        if self.theme_colors:
            self.preview_widget.apply_theme_colors(self.theme_colors)

    def _validate_input(self) -> None:
        name = self.name_edit.text().strip()
        valid = bool(name)
        if valid and not self.edit_theme:
            if name in self.theme_manager.get_builtin_themes():
                valid = False
        if self.save_btn:
            self.save_btn.setEnabled(valid)
        if self.save_apply_btn:
            self.save_apply_btn.setEnabled(valid)

    def _save_theme(self) -> None:
        if self._perform_save():
            self.accept()

    def _save_and_apply_theme(self) -> None:
        if self._perform_save(apply_immediately=True):
            self.accept()

    def _perform_save(self, apply_immediately: bool = False) -> bool:
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Invalid Name", "Please enter a theme name.")
            return False
        if not self.edit_theme and name in self.theme_manager.get_builtin_themes():
            QMessageBox.warning(
                self,
                "Name Conflict",
                f"'{name}' conflicts with a built-in theme.",
            )
            return False
        if not self.theme_colors:
            QMessageBox.warning(self, "No Colors", "Please set theme colors.")
            return False

        try:
            saved = self.theme_manager.save_custom_theme(
                name, self.theme_colors, apply_immediately
            )
            self.theme_created.emit(saved)
            msg = f"Theme '{saved}' saved"
            if apply_immediately:
                msg += " and applied"
            QMessageBox.information(self, "Theme Saved", f"{msg}!")
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save theme: {e}")
            return False
