"""A QLabel that elides its text with an ellipsis instead of widening its row."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QFontMetrics
from PyQt6.QtWidgets import QLabel


class ElidedLabel(QLabel):
    """Single-line label: full text in ``fullText()`` and the tooltip.

    Postcondition: the displayed text is ``fullText()`` when it fits the
    current width, else its ``ElideRight`` form ending in an ellipsis. The
    minimum size hint never depends on the text length.
    """

    def __init__(self, text: str = "", parent: Any = None) -> None:
        super().__init__(parent)
        self._full = ""
        self.setText(text)

    def fullText(self) -> str:  # noqa: N802 - Qt naming
        """The unelided text."""
        return self._full

    def setText(self, text: str | None) -> None:  # noqa: N802 - Qt naming
        """Store ``text`` as the full text and re-elide to the current width."""
        self._full = "" if text is None else str(text)
        self.setToolTip(self._full)
        self._elide()

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        """Height of one line; width independent of the text."""
        return QSize(1, super().minimumSizeHint().height())

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        """Size of the full text, including stylesheet padding and frame."""
        shown = self.text()
        super().setText(self._full)
        hint = super().sizeHint()
        super().setText(shown)
        return hint

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        """Re-elide when the width changes."""
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        width = self.contentsRect().width()
        shown = self._full
        if width > 1:
            shown = QFontMetrics(self.font()).elidedText(
                self._full, Qt.TextElideMode.ElideRight, width
            )
        super().setText(shown)
