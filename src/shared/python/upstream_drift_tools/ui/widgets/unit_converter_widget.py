#!/usr/bin/env python3
"""Unit Converter - Shared Component for UI Applications.

De-coupled from Gasification Model and modernized for the Fleet.
Features:
- Recent and Saved conversions.
- Bidirectional auto-conversion.
- Case-insensitive unit autocomplete.
- Localized state management using shared StateManager.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, cast

from PyQt6.QtCore import (
    QObject,
    QSettings,
    QStringListModel,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QCompleter,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from ...calculators.conversion.service import UnitConversionService, get_service
from .base_calculator_widget import BaseCalculatorWindow

logger = logging.getLogger(__name__)


class ConversionRow:
    """Represents a single conversion row data model."""

    def __init__(
        self,
        row_id: str,
        from_unit: str = "°C",
        to_unit: str = "°F",
        from_value: str = "",
        to_value: str = "",
        is_saved: bool = False,
        last_used: str | None = None,
    ) -> None:
        assert row_id is not None, "row_id must be provided"
        self.row_id = row_id
        self.from_unit = from_unit
        self.to_unit = to_unit
        self.from_value = from_value
        self.to_value = to_value
        self.is_saved = is_saved
        self.last_used = last_used or datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "row_id": self.row_id,
            "from_unit": self.from_unit,
            "to_unit": self.to_unit,
            "from_value": str(self.from_value),
            "to_value": str(self.to_value),
            "is_saved": self.is_saved,
            "last_used": self.last_used,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversionRow:
        """Create from dictionary."""
        return cls(
            row_id=data.get("row_id", ""),
            from_unit=data.get("from_unit", "°C"),
            to_unit=data.get("to_unit", "°F"),
            from_value=data.get("from_value", ""),
            to_value=data.get("to_value", ""),
            is_saved=data.get("is_saved", False),
            last_used=data.get("last_used"),
        )

    def update_last_used(self) -> None:
        """Update last used timestamp."""
        self.last_used = datetime.now().isoformat()


class TypedConverterWidget(QWidget):
    """Container for UI widgets representing a conversion row."""

    from_value: QLineEdit
    from_unit: QComboBox
    to_value: QLineEdit
    to_unit: QComboBox
    arrow: QPushButton
    copy_btn: QPushButton
    save_btn: QPushButton
    delete_btn: QPushButton
    conversion: ConversionRow
    index: int


class CaseInsensitiveCompleter(QCompleter):
    """Case-insensitive completer for unit autocomplete."""

    def __init__(
        self, units: list[str] | None = None, parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        if units is not None:
            model = QStringListModel(units)
            self.setModel(model)
        self.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.setFilterMode(Qt.MatchFlag.MatchContains)
        self.setCompletionRole(Qt.ItemDataRole.DisplayRole)

    def splitPath(self, path: str | None) -> list[str]:
        """Split path for filtering."""
        return [path] if path is not None else []

    def updateModel(self, units: list[str]) -> None:
        """Update the completer model with new units."""
        assert units is not None, "units must be provided"
        model = QStringListModel(units)
        self.setModel(model)


class UnitConverterWidget(BaseCalculatorWindow):
    """Shared Unit Converter widget with saved configurations and recent history."""

    calculation_finished = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            calculator_name="UnitConverter",
            window_title="Unit Converter",
            min_size=(850, 600),
            parent=parent,
        )

        self.settings = QSettings("UpstreamDriftTools", "UnitConverter")
        self.converter: UnitConversionService = get_service()

        # Internal state
        self.rows: list[ConversionRow] = []
        self.recent_widgets: list[TypedConverterWidget] = []
        self.saved_widgets: list[TypedConverterWidget] = []
        self.all_units: list[str] = []
        self.last_edited: dict[int, str] = {}
        self.pending_conversion: tuple[int, str] | None = None

        # Debounce timer for auto-conversion
        self.debounce_timer = QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.setInterval(300)
        self.debounce_timer.timeout.connect(self._perform_debounced_conversion)

        self._load_all_units()
        self._load_conversions()
        self._init_ui()

    @property
    def recent_conversions(self) -> list[ConversionRow]:
        """Get the most recent (non-saved) conversions."""
        return [r for r in self.rows if not r.is_saved][:3]

    @property
    def saved_conversions(self) -> list[ConversionRow]:
        """Get the saved conversions."""
        return [r for r in self.rows if r.is_saved][:3]

    def _load_all_units(self) -> None:
        """Load all available units from converter."""
        all_units_by_category = self.converter.get_supported_units()
        all_units_list: list[str] = []
        for units in all_units_by_category.values():
            all_units_list.extend(units)
        self.all_units = sorted(set(all_units_list))

    def _get_compatible_units(self, from_unit: str) -> list[str]:
        """Get all units compatible with the given unit."""
        assert from_unit is not None, "from_unit must be provided"
        if not from_unit:
            return self.all_units

        try:
            # Type ignore since _get_category and _normalize_unit might be internal
            from_category = self.converter._get_category(
                self.converter._normalize_unit(from_unit)
            )
            if from_category:
                all_units_by_category = self.converter.get_supported_units()
                return all_units_by_category.get(from_category, self.all_units)
        except (RuntimeError, AttributeError, TypeError):
            pass

        return self.all_units

    def _init_ui(self) -> None:
        """Initialize the user interface."""
        # Use main_layout from BaseCalculatorWidget
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Content area with scroll
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(15, 15, 15, 15)
        content_layout.setSpacing(12)

        # Recent Conversions Section
        self.recent_section = QGroupBox("Recent Conversions")
        recent_layout = QVBoxLayout(self.recent_section)
        recent_layout.setContentsMargins(12, 12, 12, 12)
        recent_layout.setSpacing(4)

        recent_conversions = [r for r in self.rows if not r.is_saved][:3]
        for i, conv in enumerate(recent_conversions):
            row_widget = self._create_single_line_conversion(i, conv, is_saved=False)
            self.recent_widgets.append(row_widget)
            recent_layout.addWidget(row_widget)

        content_layout.addWidget(self.recent_section)

        # Saved Conversions Section
        self.saved_section = QGroupBox("Saved Conversions")
        saved_layout = QVBoxLayout(self.saved_section)
        saved_layout.setContentsMargins(12, 12, 12, 12)
        saved_layout.setSpacing(4)

        saved_conversions = [r for r in self.rows if r.is_saved][:3]
        for i, conv in enumerate(saved_conversions):
            row_widget = self._create_single_line_conversion(i + 3, conv, is_saved=True)
            self.saved_widgets.append(row_widget)
            saved_layout.addWidget(row_widget)

        content_layout.addWidget(self.saved_section)
        content_layout.addStretch()

        scroll.setWidget(content)
        self.main_layout.addWidget(scroll)

    def _create_single_line_conversion(
        self, index: int, conv: ConversionRow, is_saved: bool
    ) -> TypedConverterWidget:
        """Create a single-line conversion widget: VALUE UNIT <> VALUE UNIT."""
        assert index is not None, "index must be provided"
        row_widget = cast(TypedConverterWidget, QWidget())
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 2, 0, 2)
        row_layout.setSpacing(8)

        # Left side
        row_widget.from_value = QLineEdit()
        row_widget.from_value.setText(conv.from_value)
        row_widget.from_value.setFixedWidth(110)
        row_widget.from_value.textChanged.connect(
            lambda t: self._on_value_changed(index, "from", t)
        )

        row_widget.from_unit = QComboBox()
        row_widget.from_unit.setEditable(True)
        row_widget.from_unit.addItems(self.all_units)
        row_widget.from_unit.setCurrentText(conv.from_unit)
        row_widget.from_unit.setFixedWidth(130)
        row_widget.from_unit.currentTextChanged.connect(
            lambda u: self._on_unit_changed(index, "from", u)
        )

        # Arrow
        row_widget.arrow = QPushButton("⇄")
        row_widget.arrow.setFixedWidth(35)
        row_widget.arrow.clicked.connect(lambda: self._swap_values(index))

        # Right side
        row_widget.to_value = QLineEdit()
        row_widget.to_value.setText(conv.to_value)
        row_widget.to_value.setFixedWidth(110)
        row_widget.to_value.textChanged.connect(
            lambda t: self._on_value_changed(index, "to", t)
        )

        row_widget.to_unit = QComboBox()
        row_widget.to_unit.setEditable(True)
        row_widget.to_unit.addItems(self._get_compatible_units(conv.from_unit))
        row_widget.to_unit.setCurrentText(conv.to_unit)
        row_widget.to_unit.setFixedWidth(130)
        row_widget.to_unit.currentTextChanged.connect(
            lambda u: self._on_unit_changed(index, "to", u)
        )

        # Buttons
        row_widget.copy_btn = QPushButton("📋")
        row_widget.copy_btn.setFixedWidth(30)
        row_widget.copy_btn.clicked.connect(lambda: self._copy_result(index))

        action_btn = QPushButton()
        action_btn.setFixedWidth(30)
        style = self.style()
        if not is_saved:
            if style:
                action_btn.setIcon(
                    style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton)
                )
            action_btn.clicked.connect(lambda: self._save_conversion(index))
        else:
            if style:
                action_btn.setIcon(
                    style.standardIcon(QStyle.StandardPixmap.SP_TrashIcon)
                )
            action_btn.clicked.connect(lambda: self._delete_saved_conversion(index))

        row_layout.addWidget(row_widget.from_value)
        row_layout.addWidget(row_widget.from_unit)
        row_layout.addWidget(row_widget.arrow)
        row_layout.addWidget(row_widget.to_value)
        row_layout.addWidget(row_widget.to_unit)
        row_layout.addWidget(row_widget.copy_btn)
        row_layout.addWidget(action_btn)
        row_layout.addStretch()

        row_widget.conversion = conv
        row_widget.index = index
        return row_widget

    # Logic Methods (re-integrated from UnitConverterLogicMixin)

    def _on_value_changed(self, index: int, direction: str, text: str) -> None:
        assert index is not None, "index must be provided"
        self.last_edited[index] = direction
        self.pending_conversion = (index, direction)
        self.debounce_timer.stop()
        self.debounce_timer.start()

    def _on_unit_changed(self, index: int, direction: str, unit: str) -> None:
        assert index is not None, "index must be provided"
        conv = self._get_row_by_index(index)
        if not conv:
            return

        if direction == "from":
            conv.from_unit = unit.strip()
            # Update target compatible units if left unit changed
            widget = self._find_widget_by_index(index)
            if widget:
                comp_units = self._get_compatible_units(unit)
                widget.to_unit.blockSignals(True)
                try:
                    widget.to_unit.clear()
                    widget.to_unit.addItems(comp_units)
                    widget.to_unit.setCurrentText(conv.to_unit)
                finally:
                    widget.to_unit.blockSignals(False)
        else:
            conv.to_unit = unit.strip()

        conv.update_last_used()
        self._convert_row(index, "from")
        self._save_conversions()

    def _perform_debounced_conversion(self) -> None:
        if self.pending_conversion:
            row_id, direction = self.pending_conversion
            self._convert_row(row_id, direction)
            self.pending_conversion = None

    def _convert_row(self, index: int, direction: str) -> None:
        assert index is not None, "index must be provided"
        widget = self._find_widget_by_index(index)
        conv = self._get_row_by_index(index)
        if not widget or not conv:
            return
        try:
            from_u = widget.from_unit.currentText().strip()
            to_u = widget.to_unit.currentText().strip()
            if not from_u or not to_u:
                return

            try:
                widget.from_value.blockSignals(True)
                widget.to_value.blockSignals(True)
                if direction == "from":
                    val_text = widget.from_value.text().strip()
                    if val_text:
                        res = self.converter.convert(float(val_text), from_u, to_u)
                        res_str = f"{res.value:.6g}"
                        widget.to_value.setText(res_str)
                        conv.to_value = res_str
                        conv.from_value = val_text
                else:
                    val_text = widget.to_value.text().strip()
                    if val_text:
                        res = self.converter.convert(float(val_text), to_u, from_u)
                        res_str = f"{res.value:.6g}"
                        widget.from_value.setText(res_str)
                        conv.from_value = res_str
                        conv.to_value = val_text
            finally:
                widget.from_value.blockSignals(False)
                widget.to_value.blockSignals(False)

            conv.update_last_used()

        except (ValueError, KeyError, ZeroDivisionError, ArithmeticError) as e:
            logger.debug("Conversion error: %s", e)

    def _swap_values(self, index: int) -> None:
        assert index is not None, "index must be provided"
        widget = self._find_widget_by_index(index)
        if not widget:
            return
        f_v, f_u = widget.from_value.text(), widget.from_unit.currentText()
        t_v, t_u = widget.to_value.text(), widget.to_unit.currentText()

        widget.from_value.setText(t_v)
        widget.from_unit.setCurrentText(t_u)
        widget.to_value.setText(f_v)
        widget.to_unit.setCurrentText(f_u)
        self._convert_row(index, "from")

    def _copy_result(self, index: int) -> None:
        widget = self._find_widget_by_index(index)
        if widget and widget.to_value.text():
            clipboard = QApplication.clipboard()
            if clipboard:
                clipboard.setText(widget.to_value.text())
            orig = widget.copy_btn.text()
            widget.copy_btn.setText("✓")
            QTimer.singleShot(1000, lambda: widget.copy_btn.setText(orig))

    def _find_widget_by_index(self, index: int) -> TypedConverterWidget | None:
        assert index is not None, "index must be provided"
        all_widgets = self.recent_widgets + self.saved_widgets
        for w in all_widgets:
            if w.index == index:
                return w
        return None

    def _get_row_by_index(self, index: int) -> ConversionRow | None:
        """Get the ConversionRow object associated with a widget index (0-2: recent, 3-5: saved)."""
        assert index is not None, "index must be provided"
        recent = self.recent_conversions
        saved = self.saved_conversions
        if index < 3:
            return recent[index] if index < len(recent) else None
        idx = index - 3
        return saved[idx] if idx < len(saved) else None

    def _save_conversion(self, index: int) -> None:
        assert index is not None, "index must be provided"
        if index >= len(self.rows):
            return
        conv = self.rows[index]
        if conv.is_saved:
            return
        saved = [r for r in self.rows if r.is_saved]
        if len(saved) >= 3:
            oldest = sorted(saved, key=lambda x: x.last_used)[0]
            oldest.is_saved = False
        conv.is_saved = True
        self._rebuild_ui_and_save()

    def _delete_saved_conversion(self, index: int) -> None:
        assert index is not None, "index must be provided"
        if index >= len(self.rows):
            return
        self.rows[index].is_saved = False
        self._rebuild_ui_and_save()

    def _rebuild_ui_and_save(self) -> None:
        # Sort and reorganize
        recent = sorted(
            [r for r in self.rows if not r.is_saved],
            key=lambda x: x.last_used,
            reverse=True,
        )[:3]
        saved = sorted(
            [r for r in self.rows if r.is_saved],
            key=lambda x: x.last_used,
            reverse=True,
        )[:3]
        self.rows = recent + saved

        # Clear and rebuild UI
        for w in self.recent_widgets + self.saved_widgets:
            w.deleteLater()
        self.recent_widgets.clear()
        self.saved_widgets.clear()

        # Re-initialize layout
        cw = self.centralWidget()
        if cw:
            cw.deleteLater()
        if hasattr(self, "central_widget"):
            self.central_widget.setFocus()
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self._init_ui()
        self._save_conversions()

    def _load_conversions(self) -> None:
        saved_json = self.settings.value("saved_conversions", "[]")
        recent_json = self.settings.value("recent_conversions", "[]")
        try:
            saved = [ConversionRow.from_dict(d) for d in json.loads(str(saved_json))]
            recent = [ConversionRow.from_dict(d) for d in json.loads(str(recent_json))]
            self.rows = recent[:3] + saved[:3]
        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            self.rows = []

        while len(self.rows) < 6:
            is_saved = (
                len([r for r in self.rows if r.is_saved]) < 3 and len(self.rows) >= 3
            )
            self.rows.append(ConversionRow(f"row_{len(self.rows)}", is_saved=is_saved))

    def _save_conversions(self) -> None:
        recent = [r.to_dict() for r in self.rows if not r.is_saved]
        saved = [r.to_dict() for r in self.rows if r.is_saved]
        self.settings.setValue("recent_conversions", json.dumps(recent))
        self.settings.setValue("saved_conversions", json.dumps(saved))


def create_unit_converter(parent: QWidget | None = None) -> UnitConverterWidget:
    """Factory function for UnitConverterWidget."""
    return UnitConverterWidget(parent=parent)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = UnitConverterWidget()
    window.show()
    sys.exit(app.exec())
