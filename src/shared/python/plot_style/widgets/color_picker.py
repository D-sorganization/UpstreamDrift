"""Hex-color entry widget with Qt :class:`QColorDialog` button.

A small composite :class:`QWidget` that lets a user enter or pick a
matplotlib-recognised color string. The widget validates every input
(Design-by-Contract) and emits :pyattr:`colorChanged` whenever the
current color changes.

Public API
----------
* ``value() -> str``                    — current hex color.
* ``set_value(color: str) -> None``     — programmatic state setter.
* ``colorChanged(str)`` Qt signal       — emitted on validated change.
"""

from __future__ import annotations

import logging
from typing import cast

from matplotlib.colors import is_color_like, to_hex
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QColorDialog,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QWidget,
)

__all__ = ["ColorPicker"]

logger = logging.getLogger(__name__)

_DEFAULT_COLOR: str = "#1f77b4"


def _normalise_hex(color: str) -> str:
    """Return ``color`` as a lowercase ``#rrggbb`` string.

    Raises :class:`ValueError` when ``color`` is not a parseable
    matplotlib color.
    """
    if not isinstance(color, str) or not color:
        raise ValueError(f"color must be a non-empty string; got {color!r}")
    if not is_color_like(color):
        raise ValueError(f"color {color!r} is not a parseable matplotlib color")
    # ``to_hex`` always returns lowercase ``#rrggbb`` (no alpha).
    return cast(str, to_hex(color, keep_alpha=False))


class ColorPicker(QWidget):
    """Hex entry + ``QColorDialog`` button.

    Parameters
    ----------
    initial:
        Initial color string. Must be a matplotlib-parseable color.
    parent:
        Optional Qt parent.
    """

    colorChanged = pyqtSignal(str)

    def __init__(
        self,
        initial: str = _DEFAULT_COLOR,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._color: str = _normalise_hex(initial)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._edit = QLineEdit(self._color, self)
        self._edit.setMaxLength(32)
        self._edit.setObjectName("color_picker_edit")
        layout.addWidget(self._edit)

        self._button = QPushButton("…", self)
        self._button.setObjectName("color_picker_button")
        self._button.setToolTip("Open color dialog")
        self._button.setFixedWidth(32)
        layout.addWidget(self._button)

        self._refresh_button_swatch()

        # Direct connections — there is no signal loop here because
        # ``set_value`` blocks signals while updating the line edit.
        self._edit.editingFinished.connect(self._on_edit_finished)
        self._button.clicked.connect(self._open_dialog)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def value(self) -> str:
        """Return the current normalised hex color (e.g. ``"#1f77b4"``)."""
        return self._color

    def set_value(self, color: str) -> None:
        """Programmatically update the color.

        Validates ``color`` and emits :pyattr:`colorChanged` only when
        the new value differs from the current one.
        """
        new_color = _normalise_hex(color)
        if new_color == self._color:
            # Still resync the line edit so external state matches even
            # if the user typed a casing variant.
            self._edit.blockSignals(True)
            self._edit.setText(new_color)
            self._edit.blockSignals(False)
            return
        self._color = new_color
        self._edit.blockSignals(True)
        self._edit.setText(new_color)
        self._edit.blockSignals(False)
        self._refresh_button_swatch()
        self.colorChanged.emit(new_color)

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------

    def _on_edit_finished(self) -> None:
        text = self._edit.text().strip()
        try:
            self.set_value(text)
        except ValueError:
            logger.warning("Rejected invalid color string %r", text)
            # Roll back to the last good value.
            self._edit.blockSignals(True)
            self._edit.setText(self._color)
            self._edit.blockSignals(False)

    def _open_dialog(self) -> None:
        initial_qcolor = QColor(self._color)
        chosen = QColorDialog.getColor(initial_qcolor, self, "Select color")
        if chosen.isValid():
            self.set_value(chosen.name())

    def _refresh_button_swatch(self) -> None:
        # Tiny visual hint of the current color on the button background.
        self._button.setStyleSheet(
            f"QPushButton#color_picker_button {{ background-color: {self._color}; }}"
        )
