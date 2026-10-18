"""Reusable optional PyQt6 controls; connect scale_changed to display.configure."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import Protocol

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QPushButton,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.ui.hex_color_field import ColorPicker, HexColorField

from .force_colors import (
    DEFAULT_COMPRESSION_COLOR,
    DEFAULT_NEUTRAL_COLOR,
    DEFAULT_TENSION_COLOR,
    ForceColorScale,
)

#: (blend_mode, label) shown by the mode combo.
BLEND_MODE_LABELS: tuple[tuple[str, str], ...] = (
    ("tint", "Tint over model"),
    ("replace", "Replace colour"),
)
_COLOR_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("tension_color", "Tension Color (#RRGGBB)", DEFAULT_TENSION_COLOR),
    ("compression_color", "Compression Color (#RRGGBB)", DEFAULT_COMPRESSION_COLOR),
    ("neutral_color", "Neutral Color (#RRGGBB)", DEFAULT_NEUTRAL_COLOR),
)


class ForceColorTarget(Protocol):
    """Small host capability for a shared settings action."""

    def set_axial_color_scale(self, scale: ForceColorScale) -> None:
        """Apply a validated scale to the current view."""
        ...


class ForceColorHub(QObject):
    """Single source of truth for one view's scale, shared by every control.

    The settings dialog, inline toggles and persistence all read and write the
    hub; ``scale_changed`` fires only when the validated scale really changes,
    so two-way widget bindings cannot feed back into each other.
    """

    scale_changed = pyqtSignal(object)

    def __init__(
        self, scale: ForceColorScale | None = None, parent: QObject | None = None
    ) -> None:
        if scale is not None and not isinstance(scale, ForceColorScale):
            raise TypeError("scale must be ForceColorScale or None")
        super().__init__(parent)
        self._scale = scale or ForceColorScale()

    @property
    def scale(self) -> ForceColorScale:
        """The current validated scale."""
        return self._scale

    def set_scale(self, scale: ForceColorScale) -> None:
        """Adopt ``scale``; emit ``scale_changed`` only if it differs."""
        if not isinstance(scale, ForceColorScale):
            raise TypeError("scale must be ForceColorScale")
        if scale == self._scale:
            return
        self._scale = scale
        self.scale_changed.emit(scale)

    def update_scale(self, **changes: object) -> None:
        """Apply ``dataclasses.replace`` changes (validated by ForceColorScale)."""
        self.set_scale(replace(self._scale, **changes))  # type: ignore[arg-type]


class InlineForceColorControls(QWidget):
    """Compact segment tension/compression checkbox (+ optional opacity slider).

    Bound two ways to a :class:`ForceColorHub`; the full dialog stays the place
    for limits and colours.
    """

    def __init__(
        self,
        hub: ForceColorHub,
        parent: QWidget | None = None,
        *,
        show_opacity: bool = False,
        label: str = "Show Segment Tension/Compression",
    ) -> None:
        if not isinstance(hub, ForceColorHub):
            raise TypeError("hub must be ForceColorHub")
        super().__init__(parent)
        self._hub = hub
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.enabled_checkbox = QCheckBox(label)
        self.enabled_checkbox.setObjectName("force_color_inline_enabled")
        self.enabled_checkbox.setToolTip(
            "Colour model segments by signed axial load: tension vs compression"
        )
        layout.addWidget(self.enabled_checkbox)
        self.opacity_slider: QSlider | None = None
        self._opacity_label: QLabel | None = None
        if show_opacity:
            row = QHBoxLayout()
            row.addWidget(QLabel("Overlay Opacity"))
            self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
            self.opacity_slider.setObjectName("force_color_inline_opacity")
            self.opacity_slider.setRange(0, 100)
            self.opacity_slider.setAccessibleName("Overlay Opacity")
            row.addWidget(self.opacity_slider)
            self._opacity_label = QLabel()
            row.addWidget(self._opacity_label)
            layout.addLayout(row)
            self.opacity_slider.valueChanged.connect(
                lambda v: hub.update_scale(opacity=v / 100.0)
            )
        self.enabled_checkbox.toggled.connect(lambda on: hub.update_scale(enabled=on))
        hub.scale_changed.connect(self._sync)
        self._sync(hub.scale)

    def _sync(self, scale: ForceColorScale) -> None:
        for widget in (self.enabled_checkbox, self.opacity_slider):
            if widget is not None:
                widget.blockSignals(True)
        self.enabled_checkbox.setChecked(scale.enabled)
        if self.opacity_slider is not None and self._opacity_label is not None:
            percent = round(scale.opacity * 100)
            self.opacity_slider.setValue(percent)
            self._opacity_label.setText(f"{percent}%")
        for widget in (self.enabled_checkbox, self.opacity_slider):
            if widget is not None:
                widget.blockSignals(False)


def install_force_color_menu(
    window: QMainWindow,
    targets: Callable[[], Iterable[ForceColorTarget]],
    hub: ForceColorHub | None = None,
) -> QAction:
    """Install a validated View menu consistently across native main windows."""
    if not isinstance(window, QMainWindow) or not callable(targets):
        raise TypeError("main window and callable targets are required")
    menu_bar = window.menuBar()
    if menu_bar is None:
        raise RuntimeError("the simulation window requires a menu bar")
    view_menu = QMenu("View", window)
    menu_bar.addMenu(view_menu)
    return install_force_color_action(view_menu, targets, hub=hub)


def install_force_color_action(
    menu: QMenu,
    targets: Callable[[], Iterable[ForceColorTarget]],
    hub: ForceColorHub | None = None,
) -> QAction:
    """Install reusable modeless settings; query current targets when applying.

    ``hub`` (created when omitted, exposed as ``action.force_color_hub``) is the
    shared state: changing it, from the dialog or any inline control, applies
    the scale to the current targets.
    """
    if not isinstance(menu, QMenu) or not callable(targets):
        raise TypeError("menu and callable targets are required")
    if hub is not None and not isinstance(hub, ForceColorHub):
        raise TypeError("hub must be ForceColorHub or None")
    hub = hub or ForceColorHub(parent=menu)
    action = QAction("Segment Force Colors…", menu)
    action.setShortcut("Ctrl+Shift+F")
    action.force_color_hub = hub  # type: ignore[attr-defined]
    dialog: QDialog | None = None

    def apply(scale: ForceColorScale) -> None:
        for target in targets():
            target.set_axial_color_scale(scale)

    hub.scale_changed.connect(apply)

    def show() -> None:
        nonlocal dialog
        if dialog is None:
            dialog = QDialog(menu)
            dialog.setObjectName("segment_force_colors_dialog")
            dialog.setWindowTitle("Segment Force Colors")
            layout = QVBoxLayout(dialog)
            controls = ForceColorControls(dialog, scale=hub.scale)
            controls.scale_changed.connect(hub.set_scale)
            hub.scale_changed.connect(controls.set_scale)
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
        self,
        parent: QWidget | None = None,
        *,
        scale: ForceColorScale | None = None,
        color_picker: ColorPicker | None = None,
    ) -> None:
        if scale is not None and not isinstance(scale, ForceColorScale):
            raise TypeError("scale must be ForceColorScale or None")
        super().__init__(parent)
        self._scale = scale or ForceColorScale()
        picker: ColorPicker = color_picker or _pick_color
        self._color_picker: Callable[[str, QWidget], str | None] = picker
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
        self._opacity = QSlider(Qt.Orientation.Horizontal)
        self._opacity.setObjectName("force_color_opacity")
        self._opacity.setRange(0, 100)
        self._opacity.setAccessibleName("Overlay Opacity")
        self._opacity_readout = QLabel()
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(self._opacity)
        opacity_row.addWidget(self._opacity_readout)
        layout.addRow("Overlay Opacity", opacity_row)
        self._blend = QComboBox()
        self._blend.setObjectName("force_color_blend_mode")
        self._blend.setToolTip(
            "Tint keeps each segment's own colour and shifts it toward the load "
            "colour; Replace paints the load colour over it."
        )
        for mode, mode_label in BLEND_MODE_LABELS:
            self._blend.addItem(mode_label, mode)
        layout.addRow("Blend Mode", self._blend)
        self._colors: dict[str, QLineEdit] = {}
        for name, label, _default in _COLOR_FIELDS:
            field_widget = HexColorField(
                getattr(self._scale, name), object_name=name, color_picker=picker
            )
            field_widget.colour_changed.connect(lambda _hex: self._apply_edits())
            self._colors[name] = field_widget.line_edit
            layout.addRow(label, field_widget)
        apply_button = QPushButton("Apply Colors and Ranges")
        apply_button.setObjectName("apply_force_colors")
        layout.addRow(apply_button)
        reset_button = QPushButton("Reset Colours")
        reset_button.setObjectName("reset_force_colors")
        layout.addRow(reset_button)
        self._error = QLabel()
        self._error.setWordWrap(True)
        layout.addRow(self._error)
        self._legend = QLabel()
        self._legend.setWordWrap(True)
        layout.addRow(self._legend)
        self._write_widgets(self._scale)
        self._enabled.toggled.connect(self._toggle)
        self._opacity.valueChanged.connect(self._opacity_edited)
        self._blend.currentIndexChanged.connect(self._blend_edited)
        apply_button.clicked.connect(self._apply_edits)
        reset_button.clicked.connect(self._reset_colors)

    @property
    def scale(self) -> ForceColorScale:
        """The last validated scale."""
        return self._scale

    def set_scale(self, scale: ForceColorScale) -> None:
        """Show ``scale`` without emitting (hosts calling this own the change)."""
        if not isinstance(scale, ForceColorScale):
            raise TypeError("scale must be ForceColorScale")
        if scale == self._scale:
            return
        self._scale = scale
        self._write_widgets(scale)

    def _write_widgets(self, scale: ForceColorScale) -> None:
        widgets: list[QWidget] = [
            self._enabled,
            self._opacity,
            self._blend,
            *self._ranges.values(),
            *self._colors.values(),
        ]
        for widget in widgets:
            widget.blockSignals(True)
        self._enabled.setChecked(scale.enabled)
        self._opacity.setValue(round(scale.opacity * 100))
        self._blend.setCurrentIndex(self._blend.findData(scale.blend_mode))
        for name, field in self._ranges.items():
            field.setValue(getattr(scale, name))
        for name, color_field in self._colors.items():
            color_field.setText(getattr(scale, name))
        for widget in widgets:
            widget.blockSignals(False)
        self._opacity_readout.setText(f"{self._opacity.value()}%")
        self._error.clear()
        self._refresh_swatches()
        self._refresh_legend()

    def _commit(self, scale: ForceColorScale) -> None:
        self._scale = scale
        self.scale_changed.emit(scale)

    def _toggle(self, enabled: bool) -> None:
        self._commit(replace(self._scale, enabled=enabled))

    def _opacity_edited(self, percent: int) -> None:
        self._opacity_readout.setText(f"{percent}%")
        self._commit(replace(self._scale, opacity=percent / 100.0))

    def _blend_edited(self, _index: int) -> None:
        self._commit(replace(self._scale, blend_mode=self._blend.currentData()))

    def _reset_colors(self) -> None:
        reset = replace(
            self._scale,
            tension_color=DEFAULT_TENSION_COLOR,
            compression_color=DEFAULT_COMPRESSION_COLOR,
            neutral_color=DEFAULT_NEUTRAL_COLOR,
        )
        self._write_widgets(reset)
        self._commit(reset)

    def _refresh_swatches(self, *_args: object) -> None:
        return

    def _apply_edits(self) -> None:
        values = {name: field.value() for name, field in self._ranges.items()}
        colors = {name: field.text().strip() for name, field in self._colors.items()}
        try:
            scale = ForceColorScale.from_dict(
                {
                    "enabled": self._enabled.isChecked(),
                    "opacity": self._scale.opacity,
                    "blend_mode": self._scale.blend_mode,
                    **values,
                    **colors,
                }
            )
        except (ValueError, TypeError) as error:
            self._error.setText(str(error))
            return
        self._error.clear()
        self._refresh_legend()
        self._commit(scale)

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


def _pick_color(current: str, parent: QWidget) -> str | None:
    """Open the Qt colour dialog; None when cancelled."""
    initial = QColor(current) if QColor.isValidColorName(current) else QColor()
    chosen = QColorDialog.getColor(initial, parent, "Pick Segment Colour")
    return chosen.name() if chosen.isValid() else None


def install_inline_shading_toggle(
    host: object,
    layout: object,
    *,
    attr: str,
    label: str = "Shade Segments (Tension/Compression)",
    grid_position: tuple[int, ...] = (),
) -> InlineForceColorControls | None:
    """Add the inline shading checkbox bound to ``host.force_color_hub``.

    Shared by the Drake and Pinocchio GUIs. ``host.<attr>`` becomes the checkbox
    (so legacy ``isChecked()`` readers work) and ``host.segment_shading_controls``
    the widget added to ``layout`` (anything with ``addWidget``; a QGridLayout
    needs ``grid_position`` such as ``(row, col, rowspan, colspan)``). Returns None,
    adding nothing, when the host has no :class:`ForceColorHub`.
    """
    if not attr or not hasattr(layout, "addWidget"):
        raise TypeError("a layout and a checkbox attribute name are required")
    hub = getattr(host, "force_color_hub", None)
    if not isinstance(hub, ForceColorHub):
        return None
    controls = InlineForceColorControls(hub, label=label)
    setattr(host, attr, controls.enabled_checkbox)
    host.segment_shading_controls = controls  # type: ignore[attr-defined]
    layout.addWidget(controls, *grid_position)  # type: ignore[attr-defined]
    return controls


def bind_force_color_persistence(
    hub: ForceColorHub,
    engine: str,
    toggles: dict[str, QCheckBox] | None = None,
    settings_factory: Callable[[], object] | None = None,
) -> None:
    """Restore then keep saving ``hub``'s scale and the named checkboxes.

    Stored values restore silently (checkbox signals blocked; renderers read
    the checkbox on the next tick). Invalid stored data falls back to defaults
    with a logged warning. ``settings_factory`` defaults to the shared
    ``QSettings`` (UpstreamDrift/UpstreamDrift).
    """
    from .force_color_settings import (
        load_force_color_state,
        open_force_color_settings,
        save_force_color_state,
    )

    if not isinstance(hub, ForceColorHub):
        raise TypeError("hub must be ForceColorHub")
    boxes = dict(toggles or {})
    factory = settings_factory or open_force_color_settings

    scale, stored = load_force_color_state(factory(), engine)  # type: ignore[arg-type]
    for name, box in boxes.items():
        if stored.get(name) is True:
            box.blockSignals(True)
            box.setChecked(True)
            box.blockSignals(False)
    hub.set_scale(scale)

    def save(*_args: object) -> None:
        save_force_color_state(
            factory(),  # type: ignore[arg-type]
            engine,
            hub.scale,
            {name: box.isChecked() for name, box in boxes.items()},
        )

    hub.scale_changed.connect(save)
    for box in boxes.values():
        box.toggled.connect(save)
