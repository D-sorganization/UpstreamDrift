"""A label that shrinks instead of widening the tile (#9846).

Qt sizes a ``QLabel`` to the whole of its text, so one long string — a
session path in the header, a step title in the rail — becomes a floor under
the entire window. :class:`ElidedLabel` reports no minimum width at all,
draws as much of its text as fits with an ellipsis, and always carries the
whole of it in the tooltip, so nothing is lost by making the tile narrow.
"""

from __future__ import annotations

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QFontMetrics, QResizeEvent
from PyQt6.QtWidgets import QLabel, QSizePolicy, QWidget


class ElidedLabel(QLabel):
    """A label that elides its text rather than widening its parent.

    Invariant: :attr:`full_text` is what the label means; ``text()`` is what
    fits at the current width.
    """

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.full_text = text
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def set_full_text(self, text: str) -> None:
        """Replace the text; the label re-elides at its current width.

        Postcondition: :attr:`full_text` is ``text`` and the tooltip carries
        it whole.
        """
        self.full_text = text
        self.setToolTip(text)
        self._elide()

    def minimumSizeHint(self) -> QSize:  # noqa: N802 (Qt API)
        """Zero width: a floor must come from controls, never from prose."""
        return QSize(0, super().minimumSizeHint().height())

    def resizeEvent(self, a0: QResizeEvent | None) -> None:  # noqa: N802 (Qt API)
        super().resizeEvent(a0)
        self._elide()

    def _elide(self) -> None:
        metrics = QFontMetrics(self.font())
        fitted = metrics.elidedText(
            self.full_text, Qt.TextElideMode.ElideRight, self.width()
        )
        if fitted != self.text():
            self.setText(fitted)
