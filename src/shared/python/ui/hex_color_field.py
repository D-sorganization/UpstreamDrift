"""A ``#rrggbb`` entry with a colour-picker button and an injectable picker.

Shared by the force-colour controls and the character builder appearance
group (HRG-B2, #13167). The picker is a callable ``(current_hex, parent) ->
new_hex | None`` so offscreen tests inject a fake instead of opening a modal
``QColorDialog``; ``None`` means the user cancelled.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QColorDialog,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QWidget,
)

__all__ = ["ColorPicker", "HexColorField", "qt_color_dialog_picker"]

ColorPicker = Callable[[str, "QWidget | None"], "str | None"]
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def qt_color_dialog_picker(current: str, parent: QWidget | None) -> str | None:
    """Default picker: the native ``QColorDialog``; ``None`` when cancelled."""
    chosen = QColorDialog.getColor(QColor(current), parent, "Select Color")
    return chosen.name() if chosen.isValid() else None


class HexColorField(QWidget):
    """Line edit plus a "Pick..." button; ``line_edit`` keeps the object name.

    ``colour_changed`` carries a validated lowercase ``#rrggbb``. Typed text is
    only validated when set through :meth:`set_text` or the picker; the host
    reads :meth:`text` (or ``line_edit``) when it applies its edits, exactly as
    for a plain ``QLineEdit``.
    """

    colour_changed = pyqtSignal(str)

    def __init__(
        self,
        value: str = "#000000",
        *,
        object_name: str = "",
        color_picker: ColorPicker | None = None,
        parent: QWidget | None = None,
    ) -> None:
        if not _HEX.match(value):
            raise ValueError(f"colour must be a #rrggbb hex string, got {value!r}")
        super().__init__(parent)
        self._color_picker: ColorPicker = color_picker or qt_color_dialog_picker
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.line_edit = QLineEdit(value)
        if object_name:
            self.line_edit.setObjectName(object_name)
        layout.addWidget(self.line_edit)
        self._button = QPushButton("Pick…")
        layout.addWidget(self._button)
        self._button.clicked.connect(self._pick)

    def text(self) -> str:
        """The current text (whatever is in the edit)."""
        return self.line_edit.text().strip()

    def set_text(self, value: str) -> bool:
        """Show ``value`` if it is ``#rrggbb`` (emits); else keep the old text."""
        if not _HEX.match(value):
            return False
        self.line_edit.setText(value.lower())
        self.colour_changed.emit(value.lower())
        return True

    def _pick(self) -> None:
        chosen = self._color_picker(self.text(), self)
        if chosen is not None:
            self.set_text(chosen)
