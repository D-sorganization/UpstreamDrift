"""Small appearance-setting controls shared by Sidekick settings panels."""

from __future__ import annotations

from .appearance import is_hex_color
from .qt_compat import QtGui, QtWidgets


class _ColorButton(QtWidgets.QPushButton):
    """Swatch button that opens a native colour picker and shows the hex."""

    def __init__(self, color: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._color = color if is_hex_color(color) else "#000000"
        self.clicked.connect(self._choose)
        self._refresh()

    def color(self) -> str:
        """Return the current hex colour."""
        return self._color

    def set_color(self, value: str) -> None:
        """Set the colour from a hex string (ignored if invalid)."""
        if is_hex_color(value):
            self._color = str(value).strip()
            self._refresh()

    def _choose(self) -> None:  # pragma: no cover - opens a modal dialog
        chosen = QtWidgets.QColorDialog.getColor(QtGui.QColor(self._color), self)
        if chosen.isValid():
            self.set_color(chosen.name())

    def _refresh(self) -> None:
        self.setText(self._color)
        # Pick a readable text colour against the swatch.
        readable = "#000000" if _is_light(self._color) else "#ffffff"
        self.setStyleSheet(
            f"background-color: {self._color}; color: {readable}; padding: 4px;"
        )


def _is_light(hex_color: str) -> bool:
    value = hex_color.lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    try:
        r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    except (ValueError, IndexError):
        return False
    # Perceived luminance (ITU-R BT.601).
    return (0.299 * r + 0.587 * g + 0.114 * b) > 140
