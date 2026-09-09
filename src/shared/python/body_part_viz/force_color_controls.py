"""Reusable optional PyQt6 controls; connect scale_changed to display.configure."""

from __future__ import annotations

from dataclasses import replace

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

from .force_colors import ForceColorScale


class ForceColorControls(QWidget):
    """A shared toggle, editable scale and legend with atomic settings updates.

    Invalid edits leave the previous scale active. Toggling always acts immediately
    on that last validated scale, so a bad range cannot prevent switching off.
    Hosts own persistence through ForceColorScale.to_dict/from_dict.
    """

    scale_changed = pyqtSignal(object)

    def __init__(
        self, parent: QWidget | None = None, *, scale: ForceColorScale | None = None
    ) -> None:
        if scale is not None and not isinstance(scale, ForceColorScale):
            raise TypeError("scale must be ForceColorScale or None")
        super().__init__(parent)
        self._scale = scale or ForceColorScale()
        layout = QFormLayout(self)
        self._enabled = QCheckBox("Color Segments by Axial Force")
        self._enabled.setObjectName("force_color_enabled")
        self._enabled.setChecked(self._scale.enabled)
        layout.addRow(self._enabled)
        self._ranges: dict[str, QDoubleSpinBox] = {}
        for name, label in (
            ("tension_limit_n", "Tension Saturation (N)"),
            ("compression_limit_n", "Compression Saturation (N)"),
            ("deadband_n", "Neutral Band ± (N)"),
        ):
            field = QDoubleSpinBox()
            field.setObjectName(name)
            field.setDecimals(6)
            field.setRange(0, 1e15)
            field.setValue(getattr(self._scale, name))
            self._ranges[name] = field
            layout.addRow(label, field)
        self._colors: dict[str, QLineEdit] = {}
        for name, label in (
            ("tension_color", "Tension Color (#RRGGBB)"),
            ("compression_color", "Compression Color (#RRGGBB)"),
            ("neutral_color", "Neutral Color (#RRGGBB)"),
        ):
            color_field = QLineEdit(getattr(self._scale, name))
            color_field.setObjectName(name)
            self._colors[name] = color_field
            layout.addRow(label, color_field)
        apply_button = QPushButton("Apply Colors and Ranges")
        apply_button.setObjectName("apply_force_colors")
        layout.addRow(apply_button)
        self._error = QLabel()
        self._error.setWordWrap(True)
        layout.addRow(self._error)
        self._legend = QLabel()
        self._legend.setWordWrap(True)
        layout.addRow(self._legend)
        self._refresh_legend()
        self._enabled.toggled.connect(self._toggle)
        apply_button.clicked.connect(self._apply_edits)

    def _toggle(self, enabled: bool) -> None:
        self._scale = replace(self._scale, enabled=enabled)
        self.scale_changed.emit(self._scale)

    def _apply_edits(self) -> None:
        values = {name: field.value() for name, field in self._ranges.items()}
        colors = {name: field.text().strip() for name, field in self._colors.items()}
        try:
            scale = ForceColorScale(
                enabled=self._enabled.isChecked(), **values, **colors
            )
        except (ValueError, TypeError) as error:
            self._error.setText(str(error))
            return
        self._scale = scale
        self._error.clear()
        self._refresh_legend()
        self.scale_changed.emit(scale)

    def _refresh_legend(self) -> None:
        scale = self._scale
        self._legend.setText(
            f'<span style="color:{scale.compression_color}">■</span> '
            f"Compression ≤ −{scale.compression_limit_n:g} N · "
            f'<span style="color:{scale.neutral_color}">■</span> '
            f"Neutral ±{scale.deadband_n:g} N · "
            f'<span style="color:{scale.tension_color}">■</span> '
            f"Tension ≥ +{scale.tension_limit_n:g} N<br>"
            "Positive = tension; negative = compression. Values clip at the limits. "
            "Unavailable samples keep their original color. Axial force is not stress."
        )
