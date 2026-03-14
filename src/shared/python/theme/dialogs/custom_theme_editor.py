"""Custom Theme Editor Dialog

This dialog allows users to create and edit custom themes for the application.
Users can modify colors, preview the theme in real-time, and save their custom themes.
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

    def __init__(
        self, initial_color: str = "#ffffff", parent: QWidget | None = None
    ) -> None:
        assert initial_color is not None, "initial_color must be provided"
        assert initial_color is not None, "initial_color must be provided"
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
            hex_color = color.name()
            self.set_color(hex_color)


class ThemePreviewWidget(QWidget):
    """Widget that shows a preview of how the theme will look."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the preview UI elements."""
        layout = QVBoxLayout(self)

        title = QLabel("Theme Preview")
        title.setStyleSheet("font-size: 14px; font-weight: bold; margin: 5px;")
        layout.addWidget(title)

        group = QGroupBox("Sample Group")
        group_layout = QFormLayout(group)

        line_edit = QLineEdit("Sample text input")
        button = QPushButton("Sample Button")
        label = QLabel("Sample label text")

        group_layout.addRow("Input:", line_edit)
        group_layout.addRow("Action:", button)
        group_layout.addRow("Info:", label)

        layout.addWidget(group)
        layout.addStretch()

    def apply_theme_colors(self, colors: dict[str, str]) -> None:
        """Apply theme colors to preview elements."""
        assert colors is not None, "colors must be provided"
        assert colors is not None, "colors must be provided"
        stylesheet = f"""
            QWidget {{
                background-color: {colors.get("bg", "#ffffff")};
                color: {colors.get("text", "#000000")};
            }}
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {colors.get("border", "#cccccc")};
                border-radius: 5px;
                margin: 5px;
                padding-top: 10px;
                background-color: {colors.get("group_bg", "#f8f9fa")};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }}
            QLineEdit {{
                padding: 5px 8px;
                border: 1px solid {colors.get("border", "#cccccc")};
                border-radius: 3px;
                background-color: {colors.get("input_bg", "#ffffff")};
                color: {colors.get("text", "#000000")};
            }}
            QPushButton {{
                padding: 6px 14px;
                border: 1px solid {colors.get("border", "#cccccc")};
                border-radius: 4px;
                background-color: {colors.get("group_bg", "#f8f9fa")};
                color: {colors.get("text", "#000000")};
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {colors.get("accent", "#5a8fc4")};
                color: white;
            }}
            QLabel {{
                color: {colors.get("text_secondary", "#666666")};
                background: transparent;
            }}
        """
        self.setStyleSheet(stylesheet)


class CustomThemeEditor(QDialog):
    """Dialog for creating and editing custom themes."""

    theme_created = pyqtSignal(str)  # Emits theme name when created

    def __init__(
        self,
        theme_manager: ThemeManager,
        parent: QWidget | None = None,
        edit_theme: str | None = None,
    ) -> None:
        assert theme_manager is not None, "theme_manager must be provided"
        assert theme_manager is not None, "theme_manager must be provided"
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
        """Set up the dialog UI."""
        layout = QHBoxLayout(self)

        left_widget = self._build_left_panel()
        layout.addWidget(left_widget, 1)

        right_widget = self._build_right_panel()
        layout.addWidget(right_widget, 1)

        button_box = self._build_dialog_buttons()
        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addWidget(button_box)

        container = QWidget()
        container.setLayout(main_layout)
        dialog_layout = QVBoxLayout(self)
        dialog_layout.addWidget(container)

    def _build_left_panel(self) -> QWidget:
        """Build the left panel with name input, color editor, and presets."""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Theme name input
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

        left_layout.addWidget(self._build_color_editor())
        left_layout.addWidget(self._build_preset_buttons())
        left_layout.addStretch()
        return left_widget

    def _build_color_editor(self) -> QGroupBox:
        """Build the scrollable color picker section."""
        colors_group = QGroupBox("Theme Colors")
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
            color_button = ColorPickerButton()
            color_button.setToolTip(tooltip)
            color_button.color_changed.connect(partial(self._on_color_changed, key))
            self.color_buttons[key] = color_button
            colors_layout.addRow(f"{name}:", color_button)

        colors_scroll = QScrollArea()
        colors_scroll.setWidget(colors_widget)
        colors_scroll.setWidgetResizable(True)
        colors_scroll.setMaximumHeight(400)

        colors_group_layout = QVBoxLayout(colors_group)
        colors_group_layout.addWidget(colors_scroll)
        return colors_group

    def _build_preset_buttons(self) -> QGroupBox:
        """Build the quick-start preset theme buttons."""
        preset_group = QGroupBox("Quick Start")
        preset_layout = QVBoxLayout(preset_group)

        preset_info = QLabel("Start with an existing theme and modify it:")
        preset_info.setWordWrap(True)
        preset_layout.addWidget(preset_info)

        preset_buttons_layout = QHBoxLayout()
        for theme_name in self.theme_manager.get_builtin_themes():
            btn = QPushButton(theme_name)
            btn.clicked.connect(
                lambda checked, name=theme_name: self._load_preset_theme(name)
            )
            preset_buttons_layout.addWidget(btn)

        preset_layout.addLayout(preset_buttons_layout)
        return preset_group

    def _build_right_panel(self) -> QWidget:
        """Build the right panel with live preview."""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        preview_label = QLabel("Live Preview")
        preview_label.setStyleSheet("font-size: 14px; font-weight: bold; margin: 5px;")
        right_layout.addWidget(preview_label)

        self.preview_widget = ThemePreviewWidget()
        right_layout.addWidget(self.preview_widget, 1)
        return right_widget

    def _build_dialog_buttons(self) -> QDialogButtonBox:
        """Build Save/Cancel/Save&Apply button box and connect signals."""
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
        if self.save_apply_btn is not None:
            self.save_apply_btn.clicked.connect(self._save_and_apply_theme)
        return button_box

    def _load_initial_colors(self) -> None:
        """Load initial colors from existing theme or defaults."""
        if self.edit_theme:
            theme_def = self.theme_manager.get_theme_definition(self.edit_theme)
            if theme_def:
                self.theme_colors = dict(theme_def)
        else:
            current_theme = self.theme_manager.get_current_theme_name()
            theme_def = self.theme_manager.get_theme_definition(current_theme)
            if theme_def:
                self.theme_colors = dict(theme_def)

        for key, button in self.color_buttons.items():
            if key in self.theme_colors:
                button.set_color(self.theme_colors[key])

        self._update_preview()

    def _load_preset_theme(self, theme_name: str) -> None:
        """Load colors from a preset theme."""
        assert theme_name is not None, "theme_name must be provided"
        assert theme_name is not None, "theme_name must be provided"
        theme_def = self.theme_manager.get_theme_definition(theme_name)
        if theme_def:
            self.theme_colors = dict(theme_def)

            for key, button in self.color_buttons.items():
                if key in self.theme_colors:
                    button.set_color(self.theme_colors[key])

            self._update_preview()

            logger.info(f"Loaded preset theme: {theme_name}")

    def _connect_signals(self) -> None:
        """Connect UI signals."""
        self.name_edit.textChanged.connect(self._validate_input)

    def _on_color_changed(self, key: str, color: str) -> None:
        """Handle color change from color picker."""
        assert key is not None, "key must be provided"
        assert key is not None, "key must be provided"
        self.theme_colors[key] = color
        self._update_preview()
        logger.debug(f"Color changed: {key} = {color}")

    def _update_preview(self) -> None:
        """Update the theme preview."""
        if self.theme_colors:
            self.preview_widget.apply_theme_colors(self.theme_colors)

    def _validate_input(self) -> None:
        """Validate user input and enable/disable save button."""
        name = self.name_edit.text().strip()

        valid = bool(name)

        if (
            valid
            and not self.edit_theme
            and name in self.theme_manager.get_builtin_themes()
        ):
            valid = False

        if self.save_btn:
            self.save_btn.setEnabled(valid)
        if self.save_apply_btn:
            self.save_apply_btn.setEnabled(valid)

    def _save_theme(self) -> None:
        """Save the theme and close dialog."""
        if self._perform_save():
            self.accept()

    def _save_and_apply_theme(self) -> None:
        """Save the theme, apply it, and close dialog."""
        if self._perform_save(apply_immediately=True):
            self.accept()

    def _perform_save(self, apply_immediately: bool = False) -> bool:
        """Perform the actual save operation."""
        assert apply_immediately is not None, "apply_immediately must be provided"
        assert apply_immediately is not None, "apply_immediately must be provided"
        name = self.name_edit.text().strip()

        if not name:
            QMessageBox.warning(self, "Invalid Name", "Please enter a theme name.")
            return False

        if not self.edit_theme and name in self.theme_manager.get_builtin_themes():
            QMessageBox.warning(
                self,
                "Name Conflict",
                f"'{name}' conflicts with a built-in theme. Please choose a different name.",
            )
            return False

        if not self.theme_colors:
            QMessageBox.warning(self, "No Colors", "Please set theme colors.")
            return False

        try:
            saved_name = self.theme_manager.save_custom_theme(
                name, self.theme_colors, apply_immediately
            )

            self.theme_created.emit(saved_name)

            if apply_immediately:
                QMessageBox.information(
                    self,
                    "Theme Saved",
                    f"Theme '{saved_name}' has been saved and applied!",
                )
            else:
                QMessageBox.information(
                    self, "Theme Saved", f"Theme '{saved_name}' has been saved!"
                )

            logger.info(f"Successfully saved theme: {saved_name}")
            return True

        except ValueError as e:
            QMessageBox.warning(self, "Save Error", str(e))
            return False
        except (OSError, KeyError, TypeError) as e:
            QMessageBox.critical(self, "Unexpected Error", f"Failed to save theme: {e}")
            logger.exception("Failed to save custom theme")
            return False
