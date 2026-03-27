"""Dialog for designing and saving custom colour themes."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Final

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..colors import THEME_COLOR_KEYS

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from ..theme_manager import ThemeManager

logger = logging.getLogger(__name__)

# ITU-R BT.601-7 luma coefficients for converting RGB to perceived brightness.
LUMA_RED: Final[float] = 0.299  # [unitless] ITU-R BT.601-7 Recommendation
LUMA_GREEN: Final[float] = 0.587  # [unitless] ITU-R BT.601-7 Recommendation
LUMA_BLUE: Final[float] = 0.114  # [unitless] ITU-R BT.601-7 Recommendation
# Threshold chosen to keep text contrast within WCAG 2.1 readability guidance.
BRIGHTNESS_THRESHOLD: Final[int] = 160  # [0-255] W3C WCAG 2.1 contrast heuristic


def _colour_from_text(value: str) -> QColor | None:
    """Return a :class:`~PyQt6.QtGui.QColor` parsed from *value* if possible."""

    if not value:
        return None

    text = value.strip()
    if not text:
        return None

    colour = QColor()

    if re.search(r"[,\s]", text):
        parts = [part for part in re.split(r"[,\s]+", text) if part]
        if len(parts) != 3:
            return None
        try:
            r, g, b = (max(0, min(255, int(float(part)))) for part in parts)
        except ValueError:
            return None
        colour = QColor(r, g, b)
    else:
        candidate = text
        if len(candidate) in {3, 6} and not candidate.startswith("#"):
            candidate = f"#{candidate}"
        colour.setNamedColor(candidate)

    if colour.isValid():
        return colour

    return None


def _hex_from_colour(colour: QColor) -> str:
    """Return the ``#rrggbb`` representation of *colour*."""

    return colour.name().lower()


class ColorFieldEditor(QWidget):
    """Composite widget that allows text or colour-wheel selection."""

    def __init__(self, initial_colour: str, parent: QWidget | None = None) -> None:
        if not (initial_colour is not None):
            raise ValueError("initial_colour must be provided")
        if not (initial_colour is not None):
            raise ValueError("initial_colour must be provided")
        super().__init__(parent)

        self._current_colour = "#000000"

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText("#RRGGBB or R,G,B")
        layout.addWidget(self.line_edit)

        self.pick_button = QPushButton("Pick\u2026")
        self.pick_button.setAutoDefault(False)
        self.pick_button.setDefault(False)
        self.pick_button.setFixedWidth(72)
        layout.addWidget(self.pick_button)

        self.line_edit.textChanged.connect(self._handle_text_changed)
        self.pick_button.clicked.connect(self._open_colour_dialog)

        try:
            self.set_colour(initial_colour)
        except ValueError:
            self.set_colour("#ffffff")

    # ------------------------------------------------------------------
    def set_colour(self, value: str) -> None:
        """Set the displayed colour."""
        colour = _colour_from_text(value)
        if colour is None:
            raise ValueError(f"Invalid colour value: {value}")

        hex_colour = _hex_from_colour(colour)
        self._current_colour = hex_colour
        self.line_edit.blockSignals(True)
        self.line_edit.setText(hex_colour)
        self.line_edit.blockSignals(False)
        self._update_button_style(hex_colour)

    def get_colour(self) -> str:
        """Get the current colour as a hex string."""
        colour = _colour_from_text(self.line_edit.text())
        if colour is None:
            raise ValueError("Enter a valid colour in hex or RGB form.")
        return _hex_from_colour(colour)

    # ------------------------------------------------------------------
    def _handle_text_changed(self, text: str) -> None:
        """Update preview when text changes."""
        if not (text is not None):
            raise ValueError("text must be provided")
        if not (text is not None):
            raise ValueError("text must be provided")
        colour = _colour_from_text(text)
        if colour is None:
            self._update_button_style(self._current_colour)
            return

        self._current_colour = _hex_from_colour(colour)
        self._update_button_style(self._current_colour)

    def _open_colour_dialog(self) -> None:
        """Open the QColorDialog."""
        from PyQt6.QtWidgets import QColorDialog

        initial = QColor(self._current_colour)
        colour = QColorDialog.getColor(initial, self, "Select colour")
        if colour.isValid():
            hex_colour = _hex_from_colour(colour)
            self._current_colour = hex_colour
            self.line_edit.setText(hex_colour)
            self._update_button_style(hex_colour)

    def _update_button_style(self, hex_colour: str) -> None:
        """Update button background and text colour for contrast."""
        if not (hex_colour is not None):
            raise ValueError("hex_colour must be provided")
        if not (hex_colour is not None):
            raise ValueError("hex_colour must be provided")
        try:
            r, g, b = (int(hex_colour[i : i + 2], 16) for i in (1, 3, 5))
        except ValueError:  # pragma: no cover - defensive
            r = g = b = 255

        brightness = (LUMA_RED * r) + (LUMA_GREEN * g) + (LUMA_BLUE * b)
        text_colour = "#000000" if brightness > BRIGHTNESS_THRESHOLD else "#ffffff"
        self.pick_button.setStyleSheet(
            f"background-color: {hex_colour}; color: {text_colour}; border: 1px solid #6c757d;"
        )


class CustomThemeDialog(QDialog):
    """Interactive dialog for creating, editing, and saving themes."""

    COLOR_LABELS: dict[str, str] = {
        "bg": "Background",
        "group_bg": "Panel background",
        "border": "Border",
        "text": "Primary text",
        "text_secondary": "Secondary text",
        "label": "Muted text",
        "focus": "Focus highlight",
        "input_bg": "Input background",
        "accent": "Accent colour",
        "title_bg": "Section title background",
        "title_border": "Section title border",
        "table_header": "Table header",
        "table_alt": "Table alternate row",
        "button_hover": "Button hover",
    }

    def __init__(
        self,
        theme_manager: ThemeManager,
        parent: QWidget | None = None,
    ) -> None:
        if not (theme_manager is not None):
            raise ValueError("theme_manager must be provided")
        if not (theme_manager is not None):
            raise ValueError("theme_manager must be provided")
        super().__init__(parent)

        self.theme_manager = theme_manager
        self.result_data: dict[str, object] | None = None

        self.setWindowTitle("Create Custom Theme")
        self.setModal(True)
        self.resize(520, 640)

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)

        intro_label = QLabel(
            "Design a colour palette using either hex values or RGB numbers.\n"
            "Start from an existing theme, adjust the colours, then save it for reuse."
        )
        intro_label.setWordWrap(True)
        main_layout.addWidget(intro_label)

        form_layout = QFormLayout()
        self.theme_name_edit = QLineEdit()
        self.theme_name_edit.setPlaceholderText("Enter a name for your theme")
        form_layout.addRow("Theme name:", self.theme_name_edit)

        self.base_theme_combo = QComboBox()
        self.base_theme_combo.currentIndexChanged.connect(
            self._handle_base_theme_changed
        )
        form_layout.addRow("Base theme:", self.base_theme_combo)
        main_layout.addLayout(form_layout)

        colours_group = QGroupBox("Colour settings")
        colours_form = QFormLayout()
        colours_group.setLayout(colours_form)
        main_layout.addWidget(colours_group)

        self.color_editors: dict[str, ColorFieldEditor] = {}

        current_theme = self.theme_manager.get_current_theme_name()
        theme_definition = (
            self.theme_manager.get_theme_definition(current_theme)
            or self.theme_manager.get_theme_definition("Light")
            or {}
        )

        for key in THEME_COLOR_KEYS:
            if key not in self.COLOR_LABELS:
                logger.debug("Skipping colour key not mapped for editing: %s", key)
                continue
            editor = ColorFieldEditor(theme_definition.get(key, "#ffffff"), self)
            colours_form.addRow(f"{self.COLOR_LABELS[key]}:", editor)
            self.color_editors[key] = editor

        self.apply_checkbox = QCheckBox("Apply this theme after saving")
        self.apply_checkbox.setChecked(True)
        main_layout.addWidget(self.apply_checkbox)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._handle_accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

        self._populate_base_theme_combo()
        self._initialise_default_name(current_theme)

    # ------------------------------------------------------------------
    def _populate_base_theme_combo(self) -> None:
        """Populate the combo box with available themes."""
        self.base_theme_combo.blockSignals(True)
        self.base_theme_combo.clear()

        for name in self.theme_manager.get_builtin_themes():
            self.base_theme_combo.addItem(f"{name} (built-in)", name)

        custom_names = self.theme_manager.get_custom_theme_names()
        if custom_names:
            self.base_theme_combo.insertSeparator(self.base_theme_combo.count())
            for name in custom_names:
                self.base_theme_combo.addItem(f"{name} (custom)", name)

        current_theme = self.theme_manager.get_current_theme_name()
        index = self.base_theme_combo.findData(current_theme)
        if index == -1:
            index = 0
        self.base_theme_combo.setCurrentIndex(index)
        self.base_theme_combo.blockSignals(False)

    def _initialise_default_name(self, current_theme: str) -> None:
        """Suggest a unique name for the new theme."""
        if not (current_theme is not None):
            raise ValueError("current_theme must be provided")
        if not (current_theme is not None):
            raise ValueError("current_theme must be provided")
        suggestion = f"{current_theme} Custom"
        reserved = set(self.theme_manager.get_builtin_themes())
        reserved.update(self.theme_manager.get_custom_theme_names())

        counter = 1
        candidate = suggestion
        while candidate in reserved:
            counter += 1
            candidate = f"{suggestion} {counter}"

        self.theme_name_edit.setText(candidate)

    # ------------------------------------------------------------------
    def _handle_base_theme_changed(self) -> None:
        """Update colour fields when a base theme is selected."""
        theme_name = self.base_theme_combo.currentData()
        if not theme_name:
            return

        theme = self.theme_manager.get_theme_definition(theme_name)
        if not theme:
            return

        for key, editor in self.color_editors.items():
            try:
                if key in theme:
                    editor.set_colour(theme[key])
                else:
                    editor.set_colour(editor.get_colour())
            except ValueError:
                editor.set_colour("#ffffff")

    def _handle_accept(self) -> None:
        """Validate and accept the dialog."""
        if self._collect_results():
            self.accept()

    def _collect_results(self) -> bool:
        """Gather results into the result_data dictionary."""
        name = self.theme_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Invalid name", "Please provide a theme name.")
            self.theme_name_edit.setFocus()
            return False

        if name in self.theme_manager.get_builtin_themes():
            QMessageBox.warning(
                self,
                "Reserved name",
                "This name is reserved for a built-in theme. Please choose another.",
            )
            self.theme_name_edit.setFocus()
            return False

        existing = name in self.theme_manager.get_custom_theme_names()
        if existing:
            response = QMessageBox.question(
                self,
                "Overwrite theme?",
                f"A custom theme named '{name}' already exists. Overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if response != QMessageBox.StandardButton.Yes:
                return False

        colours: dict[str, str] = {}
        for key, editor in self.color_editors.items():
            try:
                colours[key] = editor.get_colour()
            except ValueError as exc:
                QMessageBox.warning(
                    self, "Invalid colour", f"{self.COLOR_LABELS[key]}: {exc}"
                )
                editor.line_edit.setFocus()
                return False

        self.result_data = {
            "name": name,
            "colors": colours,
            "apply": self.apply_checkbox.isChecked(),
        }
        return True

    # ------------------------------------------------------------------
    def get_result(self) -> dict[str, object] | None:
        """Return the saved result payload."""

        return self.result_data
