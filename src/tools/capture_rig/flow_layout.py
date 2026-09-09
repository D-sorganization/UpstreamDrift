"""A wrapping (flow) layout, because Qt does not ship one (#9844).

Items are placed left to right and wrap onto a new row when the next one
would not fit, so the layout's *minimum* width is the widest single item
rather than the sum of them all. That is the whole point: a row of twenty
buttons must not set a floor no screen can satisfy. Height then depends on
width, which is reported honestly through :meth:`FlowLayout.heightForWidth`
so parent layouts can size the widget correctly.

:meth:`FlowLayout.add_widget` accepts ``starts_line`` for items that must
open a fresh row — a section label, so it is never left stranded at the end
of somebody else's row, away from the buttons it names.
"""

from __future__ import annotations

from PyQt6.QtCore import QRect, QSize, Qt
from PyQt6.QtWidgets import QLayout, QLayoutItem, QWidget

from src.shared.python.core.contracts import require


class FlowLayout(QLayout):
    """Left-to-right layout that wraps onto as many rows as the width needs."""

    def __init__(self, parent: QWidget | None = None, *, spacing: int = 0) -> None:
        """Precondition: ``spacing`` is not negative."""
        require(spacing >= 0, "spacing must not be negative", spacing)
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._line_starts: list[bool] = []
        self._next_starts_line = False
        self.setSpacing(spacing)
        self.setContentsMargins(0, 0, 0, 0)

    # -- building -------------------------------------------------------------
    def add_widget(self, widget: QWidget, *, starts_line: bool = False) -> None:
        """Append ``widget``; ``starts_line`` makes it open a fresh row.

        Postcondition: ``count()`` grows by one and ``widget`` is reparented
        onto the layout's widget, as :meth:`QLayout.addWidget` always does.
        """
        self._next_starts_line = starts_line
        self.addWidget(widget)
        self._next_starts_line = False

    # -- QLayout plumbing -----------------------------------------------------
    def addItem(self, item: QLayoutItem | None) -> None:  # noqa: N802 (Qt override)
        """Append ``item``; ``None`` (Qt's nullable signature) is ignored."""
        if item is None:
            return
        self._items.append(item)
        self._line_starts.append(self._next_starts_line)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:  # noqa: N802 (Qt override)
        """The item at ``index``, or ``None`` when ``index`` is out of range."""
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int) -> QLayoutItem | None:  # noqa: N802 (Qt override)
        """Remove and return the item at ``index``, keeping the row flags aligned."""
        if not 0 <= index < len(self._items):
            return None
        self._line_starts.pop(index)
        return self._items.pop(index)

    def expandingDirections(self) -> Qt.Orientation:  # noqa: N802 (Qt override)
        return Qt.Orientation(0)

    # -- sizing ---------------------------------------------------------------
    def hasHeightForWidth(self) -> bool:  # noqa: N802 (Qt override)
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 (Qt override)
        """The height the items need once wrapped into ``width``."""
        return self._lay_out(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802 (Qt override)
        super().setGeometry(rect)
        self._lay_out(rect, apply=True)

    def minimumSize(self) -> QSize:  # noqa: N802 (Qt override)
        """The widest single item, not the sum of them: an honest floor."""
        size = QSize(0, 0)
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        return size.grownBy(self.contentsMargins())

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        """The preferred size: every row unwrapped, rows split only where asked."""
        widest = height = row_width = row_height = 0
        for index, item in enumerate(self._items):
            hint = item.sizeHint()
            if row_width and self._line_starts[index]:
                widest = max(widest, row_width - self.spacing())
                height += row_height + self.spacing()
                row_width = row_height = 0
            row_width += hint.width() + self.spacing()
            row_height = max(row_height, hint.height())
        widest = max(widest, row_width - self.spacing() if row_width else 0)
        return QSize(max(widest, 0), height + row_height).grownBy(
            self.contentsMargins()
        )

    # -- placement ------------------------------------------------------------
    def _lay_out(self, rect: QRect, *, apply: bool) -> int:
        """Place (or merely measure) the items inside ``rect``; return the height."""
        margins = self.contentsMargins()
        area = rect.marginsRemoved(margins)
        spacing = self.spacing()
        x, y, row_height = area.x(), area.y(), 0
        for index, item in enumerate(self._items):
            width = self._item_width(item, area.width())
            height = self._item_height(item, width)
            wrapped = self._line_starts[index] or x + width > area.right() + 1
            if row_height and wrapped:
                x, y, row_height = area.x(), y + row_height + spacing, 0
            if apply:
                item.setGeometry(QRect(x, y, width, height))
            x += width + spacing
            row_height = max(row_height, height)
        return y + row_height - rect.y() + margins.bottom()

    @staticmethod
    def _item_width(item: QLayoutItem, available: int) -> int:
        """``item``'s preferred width, never wider than the row it must fit."""
        floor = item.minimumSize().width()
        return max(floor, min(item.sizeHint().width(), available))

    @staticmethod
    def _item_height(item: QLayoutItem, width: int) -> int:
        if item.hasHeightForWidth():
            return max(item.heightForWidth(width), item.minimumSize().height())
        return item.sizeHint().height()
