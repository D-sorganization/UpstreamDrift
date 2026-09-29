"""Combobox-style colormap picker with mini gradient swatches.

Lists every entry from
:func:`src.shared.python.plot_style.registry.list_colormaps` (built-ins,
semantic aliases, and currently registered :class:`CustomColormap`
instances). Each row is decorated with a small horizontal gradient
preview rendered from the matplotlib colormap.

Public API
----------
* ``value() -> ColormapId | str``
* ``set_value(cmap_id: ColormapId | str) -> None``
* ``colormapChanged(object)`` Qt signal — payload is ``ColormapId``
  for built-ins / aliases or ``str`` for custom colormaps.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QWidget

from ..colormaps import ColormapId
from ..registry import get_colormap, list_colormaps

__all__ = ["ColormapPicker"]

logger = logging.getLogger(__name__)

_SWATCH_W = 64
_SWATCH_H = 16
_DEFAULT_CMAP: ColormapId = ColormapId.VIRIDIS


def _id_to_payload(name: str) -> ColormapId | str:
    """Return a :class:`ColormapId` for built-ins / aliases, else ``name``."""
    try:
        return ColormapId(name)
    except ValueError:
        return name


def _build_swatch(name: str) -> QIcon:
    """Render a horizontal-gradient :class:`QIcon` for ``name``."""
    pixmap = QPixmap(_SWATCH_W, _SWATCH_H)
    pixmap.fill(Qt.GlobalColor.transparent)
    try:
        cmap = get_colormap(name)
    except (KeyError, TypeError):
        # Unknown name — return an empty icon so the combo entry still works.
        return QIcon(pixmap)
    painter = QPainter(pixmap)
    samples = np.linspace(0.0, 1.0, _SWATCH_W)
    for x, t in enumerate(samples):
        rgba = cmap(float(t))
        color = QColor(
            int(round(float(rgba[0]) * 255)),
            int(round(float(rgba[1]) * 255)),
            int(round(float(rgba[2]) * 255)),
            int(round(float(rgba[3]) * 255)),
        )
        painter.setPen(color)
        painter.drawLine(x, 0, x, _SWATCH_H - 1)
    painter.end()
    return QIcon(pixmap)


class ColormapPicker(QWidget):
    """Combobox of available colormaps with a per-row gradient swatch.

    Parameters
    ----------
    initial:
        Initial selection. Accepts a :class:`ColormapId` enum or the
        string name of a registered (built-in or custom) colormap.
    parent:
        Optional Qt parent.
    """

    colormapChanged = pyqtSignal(object)

    def __init__(
        self,
        initial: ColormapId | str = _DEFAULT_CMAP,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._combo = QComboBox(self)
        self._combo.setObjectName("colormap_picker_combo")
        self._combo.setIconSize(QSize(_SWATCH_W, _SWATCH_H))
        layout.addWidget(self._combo)

        self._populate()

        # Validate / coerce the initial value before any signal wiring so
        # the constructor does not emit spuriously.
        initial_name = self._coerce_name(initial)
        index = self._combo.findText(initial_name)
        if index < 0:
            raise ValueError(f"colormap {initial!r} is not registered")
        self._combo.setCurrentIndex(index)
        self._current: ColormapId | str = _id_to_payload(initial_name)

        # ``currentIndexChanged`` fires for both user and programmatic
        # changes; ``set_value`` blocks signals while syncing so this is
        # safe with a direct connection.
        self._combo.currentIndexChanged.connect(self._on_index_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def value(self) -> ColormapId | str:
        """Return the current selection (``ColormapId`` or custom-name str)."""
        return self._current

    def set_value(self, cmap_id: ColormapId | str) -> None:
        """Set the current colormap programmatically.

        Emits :pyattr:`colormapChanged` only when the selection changes.
        """
        name = self._coerce_name(cmap_id)
        index = self._combo.findText(name)
        if index < 0:
            raise ValueError(f"colormap {cmap_id!r} is not registered")
        new_payload = _id_to_payload(name)
        if new_payload == self._current:
            return
        self._combo.blockSignals(True)
        self._combo.setCurrentIndex(index)
        self._combo.blockSignals(False)
        self._current = new_payload
        self.colormapChanged.emit(new_payload)

    def refresh(self) -> None:
        """Re-read :func:`list_colormaps` (e.g. after registering a custom)."""
        current_name = self._coerce_name(self._current)
        self._combo.blockSignals(True)
        self._combo.clear()
        self._populate()
        index = self._combo.findText(current_name)
        if index >= 0:
            self._combo.setCurrentIndex(index)
        self._combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _populate(self) -> None:
        for name in list_colormaps():
            self._combo.addItem(_build_swatch(name), name)

    @staticmethod
    def _coerce_name(cmap_id: ColormapId | str) -> str:
        if isinstance(cmap_id, ColormapId):
            return cmap_id.value
        if isinstance(cmap_id, str):
            if not cmap_id:
                raise ValueError("colormap name must be a non-empty string")
            return cmap_id
        raise TypeError(
            f"cmap_id must be ColormapId or str; got {type(cmap_id).__name__}"
        )

    def _on_index_changed(self, _index: int) -> None:
        name = cast(str, self._combo.currentText())
        new_payload = _id_to_payload(name)
        if new_payload == self._current:
            return
        self._current = new_payload
        self.colormapChanged.emit(new_payload)
