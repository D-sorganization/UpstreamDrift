"""Composite marker-style editor.

Combines a shape combobox (sourced from
:func:`src.shared.python.plot_style.shapes.SHAPE_REGISTRY` when
available, falling back to :class:`MarkerShape`), a size spinbox, an
edge-width spinbox, and an embedded :class:`ColorPicker` for the edge
color. Whenever any sub-control changes, the widget rebuilds an
immutable :class:`MarkerStyle` and emits :pyattr:`styleChanged` with
the new value.

The fill color is intentionally out of scope for this widget — it is
governed by :class:`DataChannelEditor` (data-driven) or by other
follow-up widgets for static / palette colors.

Public API
----------
* ``value() -> MarkerStyle``
* ``set_value(style: MarkerStyle) -> None``
* ``styleChanged(MarkerStyle)`` Qt signal
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QVBoxLayout,
    QWidget,
)

from ..colors import StaticColor
from ..markers import MarkerShape, MarkerStyle
from .color_picker import ColorPicker

__all__ = ["MarkerStylePicker"]

logger = logging.getLogger(__name__)


def _shape_choices() -> list[MarkerShape]:
    """Return the list of selectable shapes.

    Prefers the shape registry when populated; otherwise falls back to
    every :class:`MarkerShape` value except :pyattr:`MarkerShape.CUSTOM_MESH`,
    which requires a paired :class:`CustomMeshSpec` and is therefore not
    user-pickable from a bare combobox.
    """
    try:
        from ..shapes import SHAPE_REGISTRY  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        SHAPE_REGISTRY = None  # type: ignore[assignment]

    if SHAPE_REGISTRY:
        ids: list[MarkerShape] = [
            shape
            for shape in SHAPE_REGISTRY  # type: ignore[union-attr]
            if isinstance(shape, MarkerShape) and shape is not MarkerShape.CUSTOM_MESH
        ]
        if ids:
            return ids

    return [s for s in MarkerShape if s is not MarkerShape.CUSTOM_MESH]


class MarkerStylePicker(QWidget):
    """Editor for the visual fields of :class:`MarkerStyle`.

    Parameters
    ----------
    initial:
        Optional initial style. Defaults to :class:`MarkerStyle()`.
        The fill color is preserved between rebuilds.
    parent:
        Optional Qt parent.
    """

    styleChanged = pyqtSignal(MarkerStyle)

    def __init__(
        self,
        initial: MarkerStyle | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        if initial is not None and not isinstance(initial, MarkerStyle):
            raise TypeError(
                f"initial must be MarkerStyle or None; got {type(initial).__name__}"
            )
        self._style: MarkerStyle = initial or MarkerStyle()
        self._suppress = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("Marker style", self)
        outer.addWidget(group)

        form = QFormLayout(group)

        self._shape_combo = QComboBox(group)
        self._shape_combo.setObjectName("marker_shape_combo")
        self._shape_choices: list[MarkerShape] = _shape_choices()
        for shape in self._shape_choices:
            self._shape_combo.addItem(str(shape), shape)
        form.addRow("Shape", self._shape_combo)

        self._size_spin = QDoubleSpinBox(group)
        self._size_spin.setObjectName("marker_size_spin")
        self._size_spin.setDecimals(2)
        self._size_spin.setRange(0.1, 1000.0)
        self._size_spin.setSingleStep(0.5)
        form.addRow("Size (px)", self._size_spin)

        self._edge_width_spin = QDoubleSpinBox(group)
        self._edge_width_spin.setObjectName("marker_edge_width_spin")
        self._edge_width_spin.setDecimals(2)
        self._edge_width_spin.setRange(0.0, 100.0)
        self._edge_width_spin.setSingleStep(0.1)
        form.addRow("Edge width (px)", self._edge_width_spin)

        self._edge_color = ColorPicker(self._style.edge_color, group)
        self._edge_color.setObjectName("marker_edge_color")
        form.addRow("Edge color", self._edge_color)

        self._sync_widgets_from_style()

        # Direct connections — the ``_suppress`` flag avoids re-entrant
        # emissions during ``set_value``.
        self._shape_combo.currentIndexChanged.connect(self._on_changed)
        self._size_spin.valueChanged.connect(self._on_changed)
        self._edge_width_spin.valueChanged.connect(self._on_changed)
        self._edge_color.colorChanged.connect(self._on_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def value(self) -> MarkerStyle:
        """Return the current :class:`MarkerStyle`."""
        return self._style

    def set_value(self, style: MarkerStyle) -> None:
        """Programmatically update the style.

        Emits :pyattr:`styleChanged` only when the resulting
        :class:`MarkerStyle` differs from the current one.
        """
        if not isinstance(style, MarkerStyle):
            raise TypeError(f"style must be MarkerStyle; got {type(style).__name__}")
        if style == self._style:
            return
        self._style = style
        with self._suppressed():
            self._sync_widgets_from_style()
        self.styleChanged.emit(self._style)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _suppressed(self) -> Iterator[None]:
        prev = self._suppress
        self._suppress = True
        try:
            yield
        finally:
            self._suppress = prev

    def _sync_widgets_from_style(self) -> None:
        # Shape — fall back to first choice if the current shape is not
        # in the user-pickable list (e.g. CUSTOM_MESH).
        try:
            idx = self._shape_choices.index(self._style.shape)
        except ValueError:
            idx = 0
        self._shape_combo.setCurrentIndex(idx)
        self._size_spin.setValue(float(self._style.size_px))
        self._edge_width_spin.setValue(float(self._style.edge_width))
        # ColorPicker.set_value is no-op when value already matches.
        self._edge_color.set_value(self._style.edge_color)

    def _on_changed(self, *_args: object) -> None:
        if self._suppress:
            return
        try:
            new_style = MarkerStyle(
                shape=self._shape_choices[self._shape_combo.currentIndex()],
                size_px=float(self._size_spin.value()),
                edge_color=self._edge_color.value(),
                edge_width=float(self._edge_width_spin.value()),
                fill_color=self._style.fill_color
                if self._style.fill_color is not None
                else StaticColor("#1f77b4"),
                opacity=self._style.opacity,
            )
        except (TypeError, ValueError) as exc:
            logger.warning("Invalid marker-style edit rejected: %s", exc)
            return
        if new_style == self._style:
            return
        self._style = new_style
        self.styleChanged.emit(new_style)
