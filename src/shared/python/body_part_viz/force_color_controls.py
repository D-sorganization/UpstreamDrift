"""Reusable optional PyQt6 controls; connect scale_changed to display.configure."""

from __future__ import annotations

from dataclasses import replace
from collections.abc import Callable, Iterable
from typing import Protocol

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
    QDialog,
    QMenu,
    QMainWindow,
    QVBoxLayout,
)

from .force_colors import ForceColorScale


class ForceColorTarget(Protocol):
    """Small host capability for a shared settings action."""

    def set_axial_color_scale(self, scale: ForceColorScale) -> None:
        """Apply a validated scale to the current view."""
        ...


def install_force_color_menu(
    window: QMainWindow, targets: Callable[[], Iterable[ForceColorTarget]]
) -> QAction:
    """Install a validated View menu consistently across native main windows."""
    if not isinstance(window, QMainWindow) or not callable(targets):
        raise TypeError("main window and callable targets are required")
    menu_bar = window.menuBar()
    if menu_bar is None:
        raise RuntimeError("the simulation window requires a menu bar")
    view_menu = QMenu("View", window)
    menu_bar.addMenu(view_menu)
    return install_force_color_action(view_menu, targets)


def install_force_color_action(
    menu: QMenu, targets: Callable[[], Iterable[ForceColorTarget]]
) -> QAction:
    """Install reusable modeless settings; query current targets when applying."""
    if not isinstance(menu, QMenu) or not callable(targets):
        raise TypeError("menu and callable targets are required")
    action = QAction("Segment Force Colors…", menu)
    action.setShortcut("Ctrl+Shift+F")
    dialog: QDialog | None = None

    def apply(scale: ForceColorScale) -> None:
        for target in targets():
            target.set_axial_color_scale(scale)

    def show() -> None:
        nonlocal dialog
        if dialog is None:
            dialog = QDialog(menu)
            dialog.setWindowTitle("Segment Force Colors")
            layout = QVBoxLayout(dialog)
            controls = ForceColorControls(dialog)
            controls.scale_changed.connect(apply)
            layout.addWidget(controls)
            layout.addWidget(
                QLabel(
                    "Only qualified axial load sources are colored. Views without "
                    "section-force data retain their original colors.",
                    dialog,
                )
            )
        dialog.show()
        dialog.raise_()

    action.triggered.connect(show)
    menu.addAction(action)
    return action


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
            scale = ForceColorScale.from_dict(
                {"enabled": self._enabled.isChecked(), **values, **colors}
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
